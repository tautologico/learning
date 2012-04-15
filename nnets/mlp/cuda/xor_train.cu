/* 
   xor_train.cu
   Implementation of a XOR neural network in CUDA, including
   network training using backpropagation. 

   Andrei de A. Formiga, 2012-03-31
*/

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>

// constant for the RNG seed
#define SEED        419217ULL

// maximum absolute value for random initialization of weights
#define MAX_ABS     1.5f

// total number of weights
#define NWEIGHTS    9

// number of active neurons
#define NEURONS     3

// number of deltas (= number of neurons)
#define NDELTAS     NEURONS

// the network weights on the device
float *dev_weights;
const int l1w_off = 0;   // offset into the weight array for layer 1 weights
const int l2w_off = 6;   // offset into the weight array for layer 2 weights

// the random number generator
curandGenerator_t gen;

// device input
float *dev_in;

// hidden outputs and activations (on device)
float *dev_hidden;
//float *dev_hidden_a;

// outputs and activations for final layer (on device)
float *dev_out;
//float *dev_out_a;

// inputs
float inputs[] = { 0.0f, 0.0f, 0.0f, 1.0f,
                   1.0f, 0.0f, 1.0f, 1.0f };

int ncases = 4;
int input_size = 2;

int hidden_size = 2;

// desired outputs
float outputs[] = { 0.0f, 1.0f, 1.0f, 0.0f };

// deltas and derivatives (on the device)
float *dev_delta;
float *dev_deriv;

const int l1delta_off = 0;
const int l2delta_off = 2;

// sigmoid activation function
__device__ float sigmoid(float t)
{
    return 1.0 / (1.0 + expf(-t));
}

__device__ float dsigmoid(float output)
{
    return output * (1 - output);
}

// --- initialization kernels ---------------------------------------------
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

// --- forward propagation kernels ----------------------------------------

// kernel for hidden layer
__global__ void forward_hidden(float *w, float *input, float *hidden)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int input_ix = blockIdx.x * blockDim.x;
    int toff = threadIdx.x + l1w_off;
    float h;

    h = w[toff * 3] * 1.0f +
        w[toff * 3 + 1] * input[input_ix] +
        w[toff * 3 + 2] * input[input_ix+1];

    // threshold
    if (h > 0.0f)
        hidden[tid] = 1.0f;
    else
        hidden[tid] = 0.0;
}

// kernel for output layer
__global__ void forward_output(float *w, float *hidden, float *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden_ix = blockIdx.x * blockDim.x;
    int toff = threadIdx.x + l2w_off;
    float o;

    o = w[toff] * 1.0f +
        w[toff+1] * hidden[2*hidden_ix] +
        w[toff+2] * hidden[2*hidden_ix+1];

    // threshold
    if (o > 0.0f)
        output[tid] = 1.0f;
    else
        output[tid] = 0.0f;
}


// --- kernels for backpropagation ----------------------------------------
__global__ void deltas_output(float *output, float *expected_out, float *delta, int ndeltas)
{
    int oid = blockIdx.x;  // one case per block, one output per case

    // TODO: check 

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int did = blockIdx.x * ndeltas + l2delta_off + threadIdx.x;
    float err;

    err = output[oid] - expected_out[oid];
    delta[did] = err * err * dsigmoid(output[oid]);
    // TODO: calculate derivatives of neuron weights?
    // index for weight: blockIdx.x * NWEIGHTS + l2w_off + threadIdx.x
    // must sum contributions from all example cases to derivatives, 
    // possibly needs a reduction
}

__global__ void deltas_hidden(float *hidden, float *output, float *w, float *delta, int ndeltas)
{
    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int did1 = blockIdx.x * ndeltas + l1delta_off + threadIdx.x;
    int did2 = blockIdx.x * ndeltas + l2delta_off;
    int wid = l2w_off + threadIdx.x + 1;

    delta[did1] = w[wid] * delta[did2] * dsigmoid(output[blockIdx.x]);
    // TODO: calculate derivatives of neuron weights?
}

// --- main ---------------------------------------------------------------
int main(int argc, char **argv)
{

    if (cudaMalloc((void**) &dev_weights, NWEIGHTS * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Could not allocate memory on the device\n");
        exit(1);
    }

    random_initialize_weights(dev_weights, MAX_ABS, NWEIGHTS);

}