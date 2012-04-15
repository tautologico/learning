
/* Implementation of a XOR neural network in CUDA */

#include <stdio.h>


// weights for the hidden layer
float weights_h[] = { 0.5f, -1.0f, -1.0f,
                      -1.5f, 1.0f, 1.0f };

// weights for the output layer
float weights_o[] = { 0.5f, -1.0f, -1.0f };

// weight arrays for the device
float *dev_hw;
float *dev_ow;

// device input
float *dev_in;

// device output
float *dev_out;

// inputs
float inputs[4][2] = { { 0.0f, 0.0f }, { 0.0f, 1.0f },
                       { 1.0f, 0.0f }, { 1.0f, 1.0f }};

float outputs[] = { 0.0f, 1.0f, 1.0f, 0.0f };

// a forward propagation pass, calculating outputs
__global__ void calculate_output(float *dev_hw, float *dev_ow, float *input, float *output)
{
    int tid = threadIdx.x;
    __shared__ float hidden_out[2];
    __shared__ float o;

    // hidden layer
    if (tid < 2) {
        hidden_out[tid] = dev_hw[tid * 3] * 1.0f +
            dev_hw[tid * 3 + 1] * input[0] +
            dev_hw[tid * 3 + 2] * input[1];

        // threshold
        if (hidden_out[tid] > 0.0f)
            hidden_out[tid] = 1.0f;
        else
            hidden_out[tid] = 0.0;
    }
    __syncthreads();

    if (tid < 1) {
        o = dev_ow[0] * 1.0f +
            dev_ow[1] * hidden_out[0] +
            dev_ow[2] * hidden_out[1];

        // threshold
        if (o > 0.0f)
            *output = 1.0f;
        else
            *output = 0.0f;
    }
}


int main(int argc, char **argv)
{
    float out;

    printf("### XOR test (forward propagation)\n");
    
    cudaMalloc((void**) &dev_hw, 6 * sizeof(float));
    cudaMalloc((void**) &dev_ow, 3 * sizeof(float));
    cudaMalloc((void**) &dev_in, 2 * sizeof(float));
    cudaMalloc((void**) &dev_out, sizeof(float));
    
    cudaMemcpy(dev_hw, weights_h, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ow, weights_o, 3 * sizeof(float), cudaMemcpyHostToDevice);

    // try input 0, 0
    cudaMemcpy(dev_in, inputs[0], 2 * sizeof(float), cudaMemcpyHostToDevice);
    calculate_output<<<1, 2>>>(dev_hw, dev_ow, dev_in, dev_out);
    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output for (0, 0) = %f\n", out);

    // try input 0, 1
    cudaMemcpy(dev_in, inputs[1], 2 * sizeof(float), cudaMemcpyHostToDevice);
    calculate_output<<<1, 2>>>(dev_hw, dev_ow, dev_in, dev_out);
    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);    

    printf("Output for (0, 1) = %f\n", out);

    // try input 1, 0
    cudaMemcpy(dev_in, inputs[2], 2 * sizeof(float), cudaMemcpyHostToDevice);
    calculate_output<<<1, 2>>>(dev_hw, dev_ow, dev_in, dev_out);
    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);    

    printf("Output for (1, 0) = %f\n", out);

    // try input 1, 1
    cudaMemcpy(dev_in, inputs[3], 2 * sizeof(float), cudaMemcpyHostToDevice);
    calculate_output<<<1, 2>>>(dev_hw, dev_ow, dev_in, dev_out);
    cudaMemcpy(&out, dev_out, sizeof(float), cudaMemcpyDeviceToHost);    

    printf("Output for (1, 1) = %f\n", out);

    cudaFree(dev_hw);
    cudaFree(dev_ow);
    cudaFree(dev_in);
    cudaFree(dev_out);
    
    return 0;
}
