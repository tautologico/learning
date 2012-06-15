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
//#define SEED        419229ULL

// maximum absolute value for random initialization of weights
#define MAX_ABS     1.5f

// learning rate
#define LRATE       0.75f

// total number of weights
#define NWEIGHTS    9

// number of active neurons
#define NEURONS          3
#define NEURONS_HIDDEN   2
#define NEURONS_OUT      1

// number of deltas (= number of neurons)
#define NDELTAS          NEURONS
#define DELTAS_HIDDEN    NEURONS_HIDDEN
#define DELTAS_OUT       NEURONS_OUT

#define NCASES           4
#define INPUT_SIZE       2
#define HIDDEN_SIZE      2
#define OUTPUT_SIZE      1

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

// outputs and activations for final layer (on device)
float *dev_out;

// inputs
float inputs[] = { 0.0f, 0.0f, 0.0f, 1.0f,
                   1.0f, 0.0f, 1.0f, 1.0f };

const int ncases = 4;
const int input_size = 2;
const int hidden_size = 2;
const int out_size = 1;

// desired outputs
float outputs[] = { 0.1f, 0.9f, 0.9f, 0.1f };
float *dev_dout;  // for the device

// deltas and derivatives (on the device)
float *dev_delta_h;
float *dev_delta_o;
float *dev_deriv;

// errors (device)
float *dev_err;


// sigmoid activation function
__device__ float asigmoid(float t)
{
    return 1.0f / (1.0f + expf(-t));
}

__device__ float dsigmoid(float output)
{
    return output * (1.0f - output);
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
    int input_ix = blockIdx.x * 2;           // 2 neurons in the previous layer
    int toff = l1w_off + threadIdx.x * 3;    // 3 weights per neuron in hidden layer
    float h;

    h = w[toff] * 1.0f +
        w[toff + 1] * input[input_ix] +
        w[toff + 2] * input[input_ix+1];

    hidden[tid] = asigmoid(h);
}

// kernel for output layer
__global__ void forward_output(float *w, float *hidden, float *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int hidden_ix = blockIdx.x * 2;          // 2 neurons in the previous layer
    int toff = l2w_off + threadIdx.x;        // 3 weights per neuron, but only 1 neuron
    float o;

    o = w[toff] * 1.0f +
        w[toff+1] * hidden[hidden_ix] +
        w[toff+2] * hidden[hidden_ix+1];

    output[tid] = asigmoid(o);
}


// --- kernels for backpropagation ----------------------------------------

// launch grid: <<<N, 1>>> for N = number of cases, 1 output neuron
__global__ void deltas_output(float *output, float *expected_out, float *deltao, float *err)
{
    // there's one delta for each output node
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    err[tid] = expected_out[tid] - output[tid];
    deltao[tid] = -err[tid] * dsigmoid(output[tid]);
}

// launch grid: <<<N, 2>>> for N = number of cases, 2 hidden neurons
__global__ void deltas_hidden(float *hidden, float *w, float *deltah, float *deltao)
{
    // tid is the index for deltah and hidden
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // oid is the corresponding index in the output layer
    // there's only one node in output so 1 per block
    int oid = blockIdx.x;
    // wid is the index into the weights, taking into account the bias weight
    int wid = l2w_off + threadIdx.x + 1;

    deltah[tid] = w[wid] * deltao[oid] * dsigmoid(hidden[tid]);
}

// launch grid: <<<N, 6>>> for N cases, 6 weights for hidden layer
__global__ void derivs_hidden(float *input, float *deltah, float *deriv)
{    
    // weights per node (2 inputs + bias)
    const int wpn = INPUT_SIZE + 1;

    // weight index
    int wid = blockIdx.x * NWEIGHTS + l1w_off + threadIdx.x;
    // delta index (3 weights per node: 2 inputs + bias)
    int did = blockIdx.x * DELTAS_HIDDEN + (threadIdx.x / wpn);
    // input index (3 weights per node)
    int iid = blockIdx.x * INPUT_SIZE + (threadIdx.x % wpn) - 1;

    // divergence due to bias weight
    float in = (threadIdx.x % wpn == 0? 1.0f : input[iid]);

    deriv[wid] = deltah[did] * in;
}

// launch grid: <<<N, 3>>> for N cases, 3 weights for output layer
__global__ void derivs_output(float *hidden, float *deltao, float *deriv)
{
    // weights per node (2 hidden neurons + bias)
    const int wpn = NEURONS_HIDDEN + 1;

    // weight index
    int wid = blockIdx.x * NWEIGHTS + l2w_off + threadIdx.x;
    // delta index (3 weights per node)
    int did = blockIdx.x * DELTAS_OUT + (threadIdx.x / wpn);
    // hidden index (3 weights per node)
    int hid = blockIdx.x * HIDDEN_SIZE + (threadIdx.x % wpn) - 1;

    // divergence due to bias weight
    float h = (threadIdx.x % wpn == 0? 1.0f : hidden[hid]);

    deriv[wid] = deltao[did] * h;
}

// <<<N, 9>>> for N cases, 9 derivs per case?
__global__ void sum_derivs(float *deriv)
{
}

// launch grid: <<<1, NWEIGHTS>>> for number of weights
__global__ void update_weights_nreduc(float *ws, float *deriv, float lrate)
{
    float dE = 0.0f;
    int wid = blockIdx.x * blockDim.x + threadIdx.x;

    // sum all derivs for the same weight
    for (int i = 0; i < NCASES; ++i)
	dE += deriv[i * NWEIGHTS + wid];

    // update weight
    ws[wid] -= (lrate * dE);
}

// --- memory allocations and initialization ------------------------------
inline float* allocateFloatsDev(int n)
{
    float *res;

    if (cudaMalloc((void**) &res, n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Could not allocate memory on the device\n");
        exit(1);
    }

    return res;
}

void allocateDev(void)
{
    // weights
    dev_weights = allocateFloatsDev(NWEIGHTS);
    
    // node values
    dev_in = allocateFloatsDev(ncases * input_size);
    dev_hidden = allocateFloatsDev(ncases * hidden_size);
    dev_out = allocateFloatsDev(ncases * out_size);

    // desired outputs and errors on device
    dev_dout = allocateFloatsDev(ncases * out_size);
    dev_err = allocateFloatsDev(ncases * out_size);
    
    // deltas and derivatives
    dev_delta_h = allocateFloatsDev(NEURONS_HIDDEN);
    dev_delta_o = allocateFloatsDev(NEURONS_OUT);
    dev_deriv = allocateFloatsDev(NWEIGHTS);
}

void freeDev(void)
{
    // weights
    cudaFree(dev_weights);

    // node values
    cudaFree(dev_in);
    cudaFree(dev_hidden);
    cudaFree(dev_out);

    // desired outputs and errors
    cudaFree(dev_dout);
    cudaFree(dev_err);
    
    // deltas and derivatives
    cudaFree(dev_delta_h);
    cudaFree(dev_deriv);
    cudaFree(dev_delta_o);
}

// initialize memory on the device to run kernels
void memorySetup(void)
{
    allocateDev();

    // initialize weights
    random_initialize_weights(dev_weights, MAX_ABS, NWEIGHTS);	
    
    // copy inputs and desired outputs
    cudaMemcpy(dev_in, inputs, ncases * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dout, outputs, ncases * out_size * sizeof(float), cudaMemcpyHostToDevice);
}

void printDevArray(float *devA, int length)
{
    float *hostA;

    hostA = (float*) malloc(length * sizeof(float));
    cudaMemcpy(hostA, devA, length * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < length; ++i)
	printf("%6.3f ", hostA[i]);

    printf("\n");
}

// --- training -----------------------------------------------------------
float batch_train(int epochs, int calc_sse, int print_sse)
{
    float err[NCASES * OUTPUT_SIZE];    
    float sse = 0.0f;

    for (int e = 0; e < epochs; ++e) {
	// forward propagation for all input cases
	forward_hidden<<<4, 2>>>(dev_weights, dev_in, dev_hidden);
	forward_output<<<4, 1>>>(dev_weights, dev_hidden, dev_out);

	// printf("Outputs: ");
	// printDevArray(dev_out, NCASES * OUTPUT_SIZE);
    
	// backprop
	deltas_output<<<4, 1>>>(dev_out, dev_dout, dev_delta_o, dev_err);
	deltas_hidden<<<4, 2>>>(dev_hidden, dev_weights, dev_delta_h, dev_delta_o);

	// printf("Deltas (hidden): ");
	// printDevArray(dev_delta_h, DELTAS_HIDDEN);
	// printf("Deltas (output): ");
	// printDevArray(dev_delta_o, DELTAS_OUT);

	// calculate SSE for this trial
	if (calc_sse) {
	    sse = 0.0f;
	    cudaMemcpy(err, dev_err, NCASES * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	    for (int i = 0; i < NCASES * OUTPUT_SIZE; ++i) {
		//printf("%6.3f ", err[i]);
		sse += (err[i] * err[i]);
	    }

	    if (print_sse)
		printf("SSE = %5.3f\n", sse);
	}

	// calculate derivatives
	derivs_hidden<<<4, 6>>>(dev_in, dev_delta_h, dev_deriv);
	derivs_output<<<4, 3>>>(dev_hidden, dev_delta_o, dev_deriv);
	
	// update weights
	update_weights_nreduc<<<1, NWEIGHTS>>>(dev_weights, dev_deriv, LRATE);
    }

    return sse;
}

void print_weights(void)
{
    float weights[NWEIGHTS];

    cudaMemcpy(weights, dev_weights, NWEIGHTS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NWEIGHTS; ++i)
        printf("%6.4f ", weights[i]);
    printf("\n");
}


// --- main ---------------------------------------------------------------
int main(int argc, char **argv)
{
    float sse;
    
    memorySetup();
    
    // print the generated weights
    printf("Randomly generated weights on the device:\n");
    print_weights();

    // do training
    printf("Batch training with 5000 epochs...\n");
    sse = batch_train(8000, 1, 0);

    printf("Final SSE: %6.3f\n", sse);
    printf("Outputs: ");
    printDevArray(dev_out, NCASES * OUTPUT_SIZE);
    
    // weights after training
    printf("Weights after training:\n");
    print_weights();
    
    freeDev();

    return 0;    
}
