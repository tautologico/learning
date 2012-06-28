/*

  xor_mlp.cu
  XOR network implementation with general MLP CUDA code. 

  Andrei de A. Formiga, 2012-06-19

*/

#include <stdio.h>

#include "mlpnnets.h"

// constant for the RNG seed
#define SEED        419217ULL
//#define SEED          149317ULL
//#define SEED        27ULL

// maximum absolute value for random initialization of weights
#define MAX_ABS     1.5f

// inputs for all cases
float inputs[] = { 0.0f, 0.0f, 0.0f, 1.0f,
                   1.0f, 0.0f, 1.0f, 1.0f };
const int ncases = 4;

// neurons per layer
int neuronsPerLayer[] = { 2, 2, 1 };

// array for expected outputs
float expected[] = { 0.1f, 0.9f, 0.9f, 0.1f };

// array to store calculated outputs
float *outputs;

// the network
MLPNetwork *xornn;

// training dataset
DataSet *trainData;


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
    RandomWeights(xornn, MAX_ABS, SEED);

    // print weights
    printf("* Random initial weights for network:\n# ");
    PrintWeights(xornn);

    // create dataset
    trainData = (DataSet*) malloc(sizeof(DataSet));

    if (trainData == NULL) {
        fprintf(stderr, "Could not allocate memory for dataset structure\n");
        return -1;
    }

    trainData->nCases = ncases;
    trainData->inputSize = 2;
    trainData->outputSize = 1;
    trainData->inputs = inputs;
    trainData->outputs = expected;
    trainData->location = LOC_HOST;

    // train the network
    int epochs = 6000;
    printf("* Training network by backpropagation with %d epochs\n", epochs);
    float sse;
    sse = BatchTrainBackprop(xornn, trainData, epochs, 0.75f, 1, 0);
    printf("* Final SSE after training: %7.9f\n", sse);
    
    // print weights after training
    printf("* Weights for network after training:\n# ");
    PrintWeights(xornn);
    
    // test trained networks with known inputs (assume outputs are already allocated)
    printf("* Calculating outputs for input cases\n");
    PresentInputsFromDataSet(xornn, trainData, ACTF_SIGMOID);

    // // print outputs per layer (debug)
    // float *outs;
    // for (int i = 0; i < 3; ++i) {
    //     printf("* Outputs for layer %d (off=%d, wPN=%d):\n", i,
    //            xornn->layers[i]->weightOffset, xornn->layers[i]->weightsPerNeuron);
    //     outs = GetLayerOutputs(xornn, i);
    //     if (outs == NULL)
    //         printf("! Couldn't get outputs for layer %d\n", i);
    //     else {
    //         for (int j = 0; j < xornn->layers[i]->nNeurons * xornn->nCases; ++j) {
    //             printf("%5.3f ", outs[j]);
    //         }
    //         printf("\n");
    //     }
    //     free(outs);
    // }

    cudaThreadSynchronize();
    // copy outputs to host memory
    outputs = (float*) malloc(4 * sizeof(float));
    CopyNetworkOutputs(xornn, outputs);

    // display results
    printf("* Results: \n");
    for (int i = 0; i < ncases; ++i) {
        printf("- Output for case (%f, %f) = %f\n",
               inputs[i*2], inputs[i*2+1], outputs[i]);
    }

    free(outputs);
    DestroyNetwork(xornn);
    
    return 0;
}
