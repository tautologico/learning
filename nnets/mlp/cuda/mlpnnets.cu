/*

  mlpnnets.cu
  Implementation of feedforward MLP neural networks in CUDA.

  Andrei de A. Formiga, 2012-05-09

 */

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>

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

MLPLayer *CreateLayer(int nNeurons, int nNeuronsPrev, int wOffset)
{
    MLPLayer *result = (MLPLayer*) calloc(1, sizeof(MLPLayer));

    if (result == NULL)
        return NULL;

    result->nNeurons = nNeurons;

    // allocate outputs and deltas on device
    result->d_outs = allocateFloatsDev(nNeurons);

    if (result->d_outs == NULL) {
        DestroyLayer(result);
        return NULL;
    }

    result->d_deltas = allocateFloatsDev(nNeurons);

    if (result->d_deltas == NULL) {
        DestroyLayer(result);
        return NULL;
    }

    result->weightsPerNeuron = nNeuronsPrev + 1;
    result->weightOffset = wOffset;

    return result;
}

MLPNetwork *CreateNetwork(int nLayers, int *neuronsPerLayer)
{
    MLPNetwork *result;

    result = (MLPNetwork*) calloc(1, sizeof(MLPNetwork));
    
    if (result == NULL)
        return NULL;

    result->nLayers = nLayers;
    result->layers = (MLPLayer**) calloc(nLayers, sizeof(MLPLayer*));

    if (result->layers == NULL) {
        free(result);
        return NULL;
    }

    // create input layer
    result->layers[0] = CreateLayer(neuronsPerLayer[0], 0, 0);
    if (result->layers[0] == NULL) {
        DestroyNetwork(result);
        return NULL;
    }

    // create remaining layers, and sum the number of weights
    int nwTotal = 0;
    int nwPrev = neuronsPerLayer[0];        
    for (int i = 1; i < nLayers; ++i) {
        result->layers[i] = CreateLayer(neuronsPerLayer[i], nwPrev, nwTotal);
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


// --- forward propagation ------------------------------------------------

// calculate outputs of one layer, assuming the previous
// layer was already calculated; the outputs corresponding to
// all input cases are computed in parallel
//
// grid will be <<<Nc, Nn>>> for Nc input cases and Nn neurons in layer
__global__ void forward_layer(MLPNetwork *net, int ixLayer)  // FIX: *net is not in device memory!
{
    int neuronsPrev = net->layers[ixLayer-1]->nNeurons; // assume layer 0 is input, never called
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ixPrev = (blockIdx.x * blockDim.x) * neuronsPrev;
    int toff = net->layers[ixLayer]->weightOffset +
        (threadIdx.x * net->layers[ixLayer]->weightsPerNeuron); // TODO: constants???

    float a = net->d_weights[toff];

    for (int i = 1; i < net->layers[ixLayer]->weightsPerNeuron; ++i)
        a += net->d_weights[toff+i] * net->layers[ixLayer-1]->d_outs[ixPrev];

    // TODO: make it possible to use other activation functions?
    // (maybe using templates)
    net->layers[ixLayer]->d_outs[tid] = asigmoid(a);
}
