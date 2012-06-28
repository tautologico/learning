/*

  mlpnnets.h
  Header file for FFMLP implementation in CUDA

  Andrei de A. Formiga, 2012-05-09

 */

#ifndef __MLPNNETS_H

#define __MLPNNETS_H

#include <curand.h>

// constants for activation functions
#define ACTF_THRESHOLD            0
#define ACTF_SIGMOID              1


struct MLPLayer
{
    int   nNeurons;
    int   weightsPerNeuron;
    int   weightOffset;
    float *d_outs;
    float *d_deltas;
};

struct MLPNetwork
{
    int      nLayers;
    MLPLayer **layers;
    float    *d_weights;
    int      nWeights;
    int      nCases;        // number of input cases stored on device
};

enum DataLocation { LOC_HOST, LOC_DEVICE, LOC_BOTH };

struct DataSet
{
    int          nCases;        // number of cases
    int          inputSize;     // size of input in each case
    int          outputSize;    // size of output in each case
    float        *inputs;       // inputs
    float        *outputs;      // outputs
    float        *d_inputs;     // inputs on device
    float        *d_outputs;    // outputs on device
    DataLocation location;      // where the data is available
};


// network functions
MLPNetwork *CreateNetwork(int nLayers, int *neuronsPerLayer);
void DestroyNetwork(MLPNetwork *net);
void RandomWeights(MLPNetwork *net, float max_abs, long seed);
void RandomWeightsGen(MLPNetwork *net, float max_abs, curandGenerator_t gen);
void PresentInputsFromDataSet(MLPNetwork *nnet, DataSet *dset, int actf);
void PresentInputs(MLPNetwork *nnet, float *d_inputs, int actf);
bool PrepareForTesting(MLPNetwork *nnet, int nCases);
bool CopyNetworkOutputs(MLPNetwork *nnet, float *outs);
float *GetLayerOutputs(MLPNetwork *nnet, int ixLayer);
void PrintWeights(MLPNetwork *nnet);
float BatchTrainBackprop(MLPNetwork *nnet, DataSet *data, int epochs,
                         float lrate, int calcSSE, int printSSE);

// dataset functions
DataSet* CreateDataSet(int nCases, int inputSize, int outputSize);
void DestroyDataSet(DataSet *dset);


#endif                                                      /* __MLPNNETS_H */
