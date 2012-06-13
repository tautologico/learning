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
    xornn = CreateNetwork(3, neuronsPerLayer, ncases);

    if (xornn == NULL) {
        fprintf(stderr, "Error creating XOR network\n");
        return -1;
    }

    // initialize weights
    printf("* Initializing weights\n");
    RandomWeights(xornn, MAX_ABS, SEED);

    // print weights
    printf("* Weights for network:\n# ");
    PrintWeights(xornn);
    
    // present inputs and do forward propagation
    printf("* Calculating outputs for input cases\n");
    PresentInputs(xornn, inputs);

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
