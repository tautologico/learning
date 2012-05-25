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
MLPNetwork *CreateNetwork(int nLayers, int *neuronsPerLayer)
{
    MLPNetwork *result;

    result = (MLPNetwork*) malloc(sizeof(MLPNetwork));
    
    if (result == NULL)
        return NULL;

    result->nLayers = nLayers;
    result->layers = (MLPLayer**) malloc(sizeof(MLPLayer*) * nLayers);

    if (result->layers == NULL) {
        free(result);
        return NULL;
    }
    
    for (int i = 0; i < nLayers; ++i) {
        // TODO: check for allocation failure?
        result->layers[i] = (MLPLayer*) malloc(sizeof(MLPLayer));
    }

    int nwTotal = 0;
    int nwPrev = neuronsPerLayer[0];
    for (int i = 1; i < nLayers; ++i) {
        nwTotal += neuronsPerLayer[i] * (nwPrev + 1);
        nwPrev = neuronsPerLayer[i];
    }
    result->nWeights = nwTotal;
    result->d_weights = allocateFloatsDev(result->nWeights);
    
    // TODO: check for the allocation of d_weights

    return result;
}

void DestroyNetwork(MLPNetwork *net)
{
    cudaFree(net->d_weights);
    for (int i = 0; i < net->nLayers; ++i)
        free(net->layers[i]);
    free(net->layers);
}


// --- forward propagation ------------------------------------------------

