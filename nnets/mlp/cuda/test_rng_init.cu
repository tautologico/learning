
/* 
   Testing the random initialization of weights
*/

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>

// constant for the RNG seed
#define SEED        419217ULL

// maximum absolute value for random initialization of weights
#define MAX_ABS     1.5f

#define NWEIGHTS    10

// the network weights on the device
float *d_weights;

// make randomly generated weights in (0.0, 1.0] be in the
// interval from -max_abs to +max_abs
__global__ void normalize_weights(float *w, float max_abs)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    w[tid] = ((w[tid] - 0.5f) / 0.5f) * max_abs;
}

// random initialization for weights
// w must be an array of floats on the device
void random_initialize_weights(float *w, float max_abs, int nweights)
{
    curandGenerator_t gen;

    // create and initialize generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(gen, SEED);
    curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_SEEDED);

    curandGenerateUniform(gen, w, nweights);
    normalize_weights<<<1, nweights>>>(w, max_abs);
    curandDestroyGenerator(gen);
}


int main(int argc, char **argv)
{
    //int nweights = 10;
    float weights[NWEIGHTS];

    if (cudaMalloc((void**) &d_weights, NWEIGHTS * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Could not allocate memory on the device\n");
        exit(1);
    }

    random_initialize_weights(d_weights, MAX_ABS, NWEIGHTS);

    // print the generated weights
    cudaMemcpy(weights, d_weights, NWEIGHTS * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Randomly generated weights on the device:\n");
    for (int i = 0; i < NWEIGHTS; ++i)
        printf("%6.4f ", weights[i]);
    printf("\n");

    cudaFree(d_weights);

    return 0;
}