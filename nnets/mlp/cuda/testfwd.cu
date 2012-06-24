/*

  testfwd.cu
  Test forward propagation

  Andrei de A. Formiga, 2012-06-11

 */

#include <stdio.h>

#include "mlpnnets.h"

// constant for the RNG seed
#define SEED        419217ULL

// maximum absolute value for random initialization of weights
#define MAX_ABS     1.5f

// weights for network
float weights[] = { 0.5f, -1.0f, -1.0f,   // hidden layer
                    -1.5f, 1.0f, 1.0f,    // hidden layer
                    0.5f, -1.0f, -1.0f }; // output layer

// inputs for all cases
float inputs[] = { 0.0f, 0.0f, 0.0f, 1.0f,
                   1.0f, 0.0f, 1.0f, 1.0f };
const int ncases = 4;

// neurons per layer
int neuronsPerLayer[] = { 2, 2, 1 };

// array for copying outputs
float outputs[] = { 0.0f, 0.0f, 0.0f, 0.0f };

// the network
MLPNetwork *xornn;


// --- main ----------------------------------------------------------
int main(int argc, char **argv)
{
    // create network
    xornn = CreateNetwork(3, neuronsPerLayer);

    if (xornn == NULL) {
        fprintf(stderr, "Error creating XOR network\n");
        return -1;
    }

    // initialize weights
    printf("* Initializing weights\n");
    cudaMemcpy(xornn->d_weights, weights, xornn->nWeights * sizeof(float),
               cudaMemcpyHostToDevice);

    // print weights
    printf("* Weights for network:\n# ");
    PrintWeights(xornn);
    
    // present inputs and do forward propagation
    printf("* Calculating outputs for input cases\n");
    if (!PrepareForTesting(xornn, ncases)) {
        fprintf(stderr, "! Could not allocate memory for outputs on device\n");
        return -1;
    }
        
    PresentInputsFromHost(xornn, inputs, ACTF_THRESHOLD);

    float *outs;
    for (int i = 0; i < 3; ++i) {
        printf("* Outputs for layer %d (off=%d, wPN=%d):\n", i,
               xornn->layers[i]->weightOffset, xornn->layers[i]->weightsPerNeuron);
        outs = GetLayerOutputs(xornn, i);
        if (outs == NULL)
            printf("! Couldn't get outputs for layer %d\n", i);
        else {
            for (int j = 0; j < xornn->layers[i]->nNeurons * xornn->nCases; ++j) {
                printf("%5.3f ", outs[j]);
            }
            printf("\n");
        }
        free(outs);
    }
    
    // copy outputs to host memory
    CopyNetworkOutputs(xornn, outputs);

    // display results
    printf("* Results: \n");
    for (int i = 0; i < ncases; ++i) {
        printf("- Output for case (%f, %f) = %f\n",
               inputs[i*2], inputs[i*2+1], outputs[i]);
    }

    DestroyNetwork(xornn);
    
    return 0;
}
