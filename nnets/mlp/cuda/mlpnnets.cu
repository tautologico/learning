/*

  mlpnnets.cu
  Implementation of feedforward MLP neural networks in CUDA.

  Andrei de A. Formiga, 2012-05-09

 */

#include <stdio.h>
#include <stdlib.h>

#include "mlpnnets.h"


// --- utility functions --------------------------------------------------
inline float* allocateFloatsDev(int n)
{
    float *res;

    if (cudaMalloc((void**) &res, n * sizeof(float)) != cudaSuccess) {
        return NULL;
    }

    return res;
}

// --- activation functions -----------------------------------------------

// sigmoid activation function
__device__ float asigmoid(float t)
{
    return 1.0f / (1.0f + expf(-t));
}

__device__ float dsigmoid(float output)
{
    return output * (1.0f - output);
}


// --- initialization -----------------------------------------------------

// make randomly generated weights in (0.0, 1.0] be in the
// interval from -max_abs to +max_abs
__global__ void normalize_weights(float *w, float max_abs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    w[tid] = ((w[tid] - 0.5f) / 0.5f) * max_abs;
}

// random initialization for weights
// w must be an array of floats on the device
void RandomWeights(MLPNetwork *net, float max_abs, long seed)
{
    curandGenerator_t gen;

    // create and initialize generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_SEEDED);

    curandGenerateUniform(gen, net->d_weights, net->nWeights);
    normalize_weights<<<1, net->nWeights>>>(net->d_weights, max_abs);
    curandDestroyGenerator(gen);
}

// initialize weights randomly using the supplied generator
// w must be an array of floats on the device
void RandomWeightsGen(MLPNetwork *net, float max_abs, curandGenerator_t gen)
{
    curandGenerateUniform(gen, net->d_weights, net->nWeights);
    normalize_weights<<<1, net->nWeights>>>(net->d_weights, max_abs);
}


// --- network construction and management --------------------------------
void DestroyLayer(MLPLayer *layer)
{
    if (layer->d_outs != NULL)
        cudaFree(layer->d_outs);

    if (layer->d_deltas != NULL)
        cudaFree(layer->d_deltas);

    free(layer);
}

MLPLayer *CreateLayer(int nNeurons, int nNeuronsPrev, int wOffset, int nCases)
{
    MLPLayer *result = (MLPLayer*) calloc(1, sizeof(MLPLayer));

    if (result == NULL)
        return NULL;

    result->nNeurons = nNeurons;

    // allocate outputs and deltas on device
    result->d_outs = allocateFloatsDev(nNeurons * nCases);

    if (result->d_outs == NULL) {
        DestroyLayer(result);
        return NULL;
    }

    // TODO: deltas allocated per case?
    result->d_deltas = allocateFloatsDev(nNeurons * nCases);

    if (result->d_deltas == NULL) {
        DestroyLayer(result);
        return NULL;
    }

    result->weightsPerNeuron = nNeuronsPrev + 1;
    result->weightOffset = wOffset;

    return result;
}

// Create a MLP neural network for execution on the GPU.
// nLayers: number of layers
// neuronsPerLayer: array of ints (size equal to nLayers) with the
//                  number of neurons for each layer
// nCases: Number of input cases to process in parallel
MLPNetwork *CreateNetwork(int nLayers, int *neuronsPerLayer, int nCases)
{
    MLPNetwork *result;

    result = (MLPNetwork*) calloc(1, sizeof(MLPNetwork));
    
    if (result == NULL)
        return NULL;

    result->nCases = nCases;
    
    result->nLayers = nLayers;
    result->layers = (MLPLayer**) calloc(nLayers, sizeof(MLPLayer*));

    if (result->layers == NULL) {
        free(result);
        return NULL;
    }

    // create input layer
    result->layers[0] = CreateLayer(neuronsPerLayer[0], 0, 0, nCases);
    if (result->layers[0] == NULL) {
        DestroyNetwork(result);
        return NULL;
    }

    // create remaining layers, and sum the number of weights
    int nwTotal = 0;
    int nwPrev = neuronsPerLayer[0];        
    for (int i = 1; i < nLayers; ++i) {
        result->layers[i] = CreateLayer(neuronsPerLayer[i], nwPrev, nwTotal, nCases);
        if (result->layers[i] == NULL) {
            DestroyNetwork(result);
            return NULL;
        }

        nwTotal += neuronsPerLayer[i] * (nwPrev + 1);
        nwPrev = neuronsPerLayer[i];        
    }

    result->nWeights = nwTotal;
    result->d_weights = allocateFloatsDev(result->nWeights);

    if (result->d_weights == NULL) {
        DestroyNetwork(result);
        return NULL;
    }        

    return result;
}

void DestroyNetwork(MLPNetwork *net)
{
    if (net->d_weights != NULL)
        cudaFree(net->d_weights);

    if (net->layers != NULL) {
        for (int i = 0; i < net->nLayers; ++i)
            if (net->layers[i] != NULL)
                DestroyLayer(net->layers[i]);

        free(net->layers);
    }

    free(net);
}


// ------------------------------------------------------------------------
// --- forward propagation ------------------------------------------------
// ------------------------------------------------------------------------

// calculate outputs of one layer, assuming the previous
// layer was already calculated; the outputs corresponding to
// all input cases are computed in parallel
//
// grid will be <<<Nc, Nn>>> for Nc input cases and Nn neurons in layer
__global__ void forward_layer(float *d_weights, int weightOffset, int weightsPerNeuron,
                              float *d_ins, int neuronsPrev, float *d_outs)
{
    // weightsPerNeuron is always = to neuronsPrev+1
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ixIn = blockIdx.x * neuronsPrev;
    int wid = weightOffset + (threadIdx.x * weightsPerNeuron);

    // bias input
    float a = d_weights[wid];

    for (int i = 1; i < weightsPerNeuron; ++i)
        a += d_weights[wid + i] * d_ins[ixIn + i-1];

    d_outs[tid] = asigmoid(a);
}

// calculate outputs of one layer using a threshold activation,
// assuming the previous layer was already calculated; the outputs
// corresponding to all input cases are computed in parallel
//
// grid will be <<<Nc, Nn>>> for Nc input cases and Nn neurons in layer
__global__ void forward_layer_threshold(float *d_weights, int weightOffset,
                                        int weightsPerNeuron,
                                        float *d_ins, int neuronsPrev,
                                        float *d_outs)
{
    // weightsPerNeuron is always = to neuronsPrev+1
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ixIn = blockIdx.x * neuronsPrev;
    int wid = weightOffset + (threadIdx.x * weightsPerNeuron);

    // bias input
    float a = d_weights[wid];

    for (int i = 1; i < weightsPerNeuron; ++i)
        a += d_weights[wid + i] * d_ins[ixIn + i-1];

    d_outs[tid] = (a > 0.0f? 1.0f : 0.0f);
}

// present a vector of input cases to the network nnet and do forward propagation.
// inputs is assumed to be in host memory, and of size equal to N * nnet->nCases,
// where N is the number of inputs to the network
void PresentInputs(MLPNetwork *nnet, float *inputs, int actf)
{
    int nInputs = nnet->layers[0]->nNeurons;

    // copy inputs to layer 0 on network
    cudaMemcpy(nnet->layers[0]->d_outs, inputs,
               nInputs * nnet->nCases * sizeof(float),
               cudaMemcpyHostToDevice);

    int nn;
    for (int l = 1; l < nnet->nLayers; ++l) {
        nn = nnet->layers[l]->nNeurons;
        if (actf == ACTF_THRESHOLD)
            forward_layer_threshold<<<nnet->nCases, nn>>>(nnet->d_weights,
                                                nnet->layers[l]->weightOffset,
                                                nnet->layers[l]->weightsPerNeuron,
                                                nnet->layers[l-1]->d_outs,
                                                nnet->layers[l-1]->nNeurons,
                                                nnet->layers[l]->d_outs);
        else
            forward_layer<<<nnet->nCases, nn>>>(nnet->d_weights,
                                                nnet->layers[l]->weightOffset,
                                                nnet->layers[l]->weightsPerNeuron,
                                                nnet->layers[l-1]->d_outs,
                                                nnet->layers[l-1]->nNeurons,
                                                nnet->layers[l]->d_outs);
    }
    
}


// ------------------------------------------------------------------------
// --- backpropagation ----------------------------------------------------
// ------------------------------------------------------------------------

// Calculate the deltas for each neuron in the output layer, and the
// error between the actual and expected outputs
//
// grid should be <<<Nc, Nn>>> for Nc cases and Nn neurons in layer
__global__ void deltas_output(float *outs, float *expected, float *d_deltas,
                              float *err)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    err[tid] = expected[tid] - outs[tid];
    d_deltas[tid] = -err[tid] * dsigmoid(outs[tid]);
}

// Calculate the deltas for each neuron in a hidden layer
//
// grid should be <<<Nc, Nn>>> for Nc cases and Nn neurons in layer
__global__ void deltas_hlayer(float *outs, float *d_weights, float *d_deltas,
                              float *d_dltnext, int neuronsNext,
                              int nxtLayerWOffset, int weightsPerNeuronNxt)
{
    // index for delta being calculated on hidden layer
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // index for first delta on next layer
    int oid = blockIdx.x * neuronsNext;
    // index for relevant weights (neurons in next layer)
    int wid = nxtLayerWOffset + threadIdx.x + 1;  // +1 to account for bias weight

    d_deltas[tid] = 0.0f;
    for (int i = 0; i < neuronsNext; ++i, wid += weightsPerNeuronNxt)
        d_deltas[tid] += d_weights[wid] * d_dltnext[oid+i] * dsigmoid(outs[tid]);
}


// Calculate the derivatives of the error relative to each weight
//
// grid should be <<<Nc, Nw>>> for Nc cases and Nw total weights for layer
__global__ void derivs_layer(float *d_inputs, float *d_deltas, float *d_derivs,
			     int nNeurons, int neuronsPrev, 
			     int nWeights, int weightsPerNeuron, int weightOffset)
{
    // weight index
    int wid = blockIdx.x * nWeights + weightOffset + threadIdx.x;
    // delta index
    int did = blockIdx.x * nNeurons + (threadIdx.x / weightsPerNeuron);
    // input index
    int iid = blockIdx.x * neuronsPrev + (threadIdx.x % weightsPerNeuron) - 1;

    float inp = (threadIdx.x % weightsPerNeuron == 0? 1.0f : d_inputs[iid]);

    d_derivs[wid] = d_deltas[did] * inp;
}

//__global__ sum_derivs(float *d_derivs)
//{
//}

// launch grid: <<<1, NWEIGHTS>>> for number of weights
// TODO: do a proper reduction instead of a set number of sums
__global__ void update_weights_nreduc(float *d_weights, float *d_derivs, float lrate,
                                      int nCases, int nWeights)
{
    float dE = 0.0f;
    int wid = blockIdx.x * blockDim.x + threadIdx.x;

    // sum all derivs for the same weight
    for (int i = 0; i < nCases; ++i)
	dE += d_derivs[i * nWeights + wid];

    // update weight
    d_weights[wid] -= (lrate * dE);
}

void TransferDataSetToDevice(DataSet *data)
{
    if (data->location == LOC_HOST) {
        //int nFloatsIn = data->nCases * data->inputSize;
        int nFloatsOut = data->nCases * data->outputSize;

        // allocate memory for dataset in device
        //data->d_inputs = allocateFloatsDev(nFloatsIn);
        data->d_outputs = allocateFloatsDev(nFloatsOut);

        // copy dataset to device (don't copy input, PresentInputs will copy)
        // TODO: check result of memcpy
        //cudaMemcpy(data->d_inputs, data->inputs,
        //           nFloatsIn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(data->d_outputs, data->outputs,
                   nFloatsOut * sizeof(float), cudaMemcpyHostToDevice);

        // change location specifier
        data->location = LOC_BOTH;
    }
}

float BatchTrainBackprop(MLPNetwork *nnet, DataSet *data, int epochs,
                         float lrate, int calcSSE, int printSSE)
{
    float *err = NULL, *d_err;
    float *d_derivs;
    float sse = 0.0f;
    MLPLayer *outLayer = nnet->layers[nnet->nLayers - 1];
    int nOutputs = outLayer->nNeurons;

    TransferDataSetToDevice(data);

    // allocate space for errors
    d_err = allocateFloatsDev(nOutputs * data->nCases);

    if (calcSSE) {
        err = (float*) malloc(nOutputs * data->nCases * sizeof(float));
        if (err == NULL) {
            fprintf(stderr, "Couldn't allocate memory to store errors\n.");
            exit(-1);
        }
    }

    // allocate memory for derivatives
    d_derivs = allocateFloatsDev(data->nCases * nnet->nWeights);

    for (int e = 0; e < epochs; ++e) {
        // forward propagation of all the cases
        PresentInputs(nnet, data->inputs, ACTF_SIGMOID);
        cudaThreadSynchronize();

        // // print outputs (debug)
        // float *outs = (float*) malloc(data->nCases * nOutputs * sizeof(float));
        // CopyNetworkOutputs(nnet, outs);
        // for (int i = 0; i < data->nCases * nOutputs; ++i)
        //     printf("%5.3f ", outs[i]);
        // printf("|| ");
        // free(outs);
        // // print outputs (debug end)
        
        // backpropagation: calculation of deltas
        deltas_output<<<data->nCases, nOutputs>>>(outLayer->d_outs,
                                                  data->d_outputs,
                                                  outLayer->d_deltas,
                                                  d_err);

        // // print deltas for output layer (debug)
        // float *deltas = (float*) malloc(data->nCases * nOutputs * sizeof(float));
        // cudaMemcpy(deltas, outLayer->d_deltas, data->nCases * nOutputs * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < data->nCases * nOutputs; ++i)
        //     printf("%5.3f ", deltas[i]);
        // printf(" -- ");
        // free(deltas);
        // // (debug end)
        

        MLPLayer *layer;
        MLPLayer *nextLayer = outLayer;
        for (int l = nnet->nLayers-2; l > 0; --l) {
            layer = nnet->layers[l];
            deltas_hlayer<<<data->nCases, layer->nNeurons>>>(layer->d_outs,
                                                             nnet->d_weights,
                                                             layer->d_deltas,
                                                             nextLayer->d_deltas,
                                                             nextLayer->nNeurons,
                                                             nextLayer->weightOffset,
                                                             nextLayer->weightsPerNeuron);
            nextLayer = layer;

            // // print deltas for layer (debug)
            // deltas = (float*) malloc(data->nCases * layer->nNeurons * sizeof(float));
            // cudaMemcpy(deltas, layer->d_deltas, data->nCases * layer->nNeurons * sizeof(float), cudaMemcpyDeviceToHost);
            // for (int i = 0; i < data->nCases * layer->nNeurons; ++i)
            //     printf("%5.3f ", deltas[i]);
            // printf(" -- ");
            // free(deltas);
            // // (debug end)           
        }
        
        // calculate SSE for this epoch
	if (calcSSE) {
	    sse = 0.0f;
	    cudaMemcpy(err, d_err, data->nCases * nOutputs * sizeof(float), cudaMemcpyDeviceToHost);
	    for (int i = 0; i < data->nCases * nOutputs; ++i) {
		//printf("%6.3f ", err[i]);
		sse += (err[i] * err[i]);
	    }

	    if (printSSE)
		printf("- SSE = %5.3f\n", sse);
	}

        // calculate derivatives of the error
        MLPLayer *prevLayer = nnet->layers[0];
        int nw;
        for (int l = 1; l < nnet->nLayers; ++l) {
            layer = nnet->layers[l];
            nw = layer->nNeurons * layer->weightsPerNeuron;
            derivs_layer<<<data->nCases, nw>>>(prevLayer->d_outs,
                                               layer->d_deltas,
                                               d_derivs,
                                               layer->nNeurons,
                                               prevLayer->nNeurons,
                                               nnet->nWeights,
                                               layer->weightsPerNeuron,
                                               layer->weightOffset);
            prevLayer = layer;
        }

        // update weights based on derivatives
        update_weights_nreduc<<<1, nnet->nWeights>>>(nnet->d_weights, d_derivs,
                                                     lrate, data->nCases,
                                                     nnet->nWeights);
    }

    if (err != NULL)
        free(err);
    
    // cleanup
    cudaFree(d_err);
    cudaFree(d_derivs);

    return sse;
}

// ------------------------------------------------------------------------
// --- utility functions --------------------------------------------------
// ------------------------------------------------------------------------

// Copy the outputs for network nnet, stored in device memory, to
// host memory pointed to by outs. outs must have size equal to N * nnet->nCases,
// where N is the number of output neurons in the network
void CopyNetworkOutputs(MLPNetwork *nnet, float *outs)
{
    MLPLayer *last = nnet->layers[nnet->nLayers-1];
    
    cudaMemcpy(outs, last->d_outs,
               last->nNeurons * nnet->nCases * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void PrintWeights(MLPNetwork *nnet)
{
    float *h_weights;

    h_weights = (float*) malloc(nnet->nWeights * sizeof(float));

    if (h_weights == NULL) {
        printf("Error allocating host memory to copy weights.\n");
    }
    else {
        // TODO: check cudaMemcpy for errors
        cudaMemcpy(h_weights, nnet->d_weights, nnet->nWeights * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < nnet->nWeights; ++i) {
            printf("%4.5f ", h_weights[i]);
        }
        printf("\n");        
    }

    free(h_weights);
}

// return an array of floats with the outputs for layer with index ixLayer
float *GetLayerOutputs(MLPNetwork *nnet, int ixLayer)
{
    int   length = nnet->layers[ixLayer]->nNeurons * nnet->nCases;
    float *result = (float*) malloc(length * sizeof(float));

    if (result == NULL)
        return NULL;

    // TODO: check cudaMemcpy for errors
    cudaMemcpy(result, nnet->layers[ixLayer]->d_outs,
               length * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}
