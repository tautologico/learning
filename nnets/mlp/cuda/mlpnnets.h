/*

  mlpnnets.h
  Header file for FFMLP implementation in CUDA

  Andrei de A. Formiga, 2012-05-09

 */

#ifndef __MLPNNETS_H

#define __MLPNNETS_H

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
    int      nCases;      // number of input cases stored on device
};

// network functions
MLPNetwork *CreateNetwork(int nLayers, int *neuronsPerLayer, int nCases);
void DestroyNetwork(MLPNetwork *net);
void RandomWeights(MLPNetwork *net, float max_abs, long seed);
void RandomWeightsGen(MLPNetwork *net, float max_abs, curandGenerator_t gen);
void PresentInputs(MLPNetwork *nnet, float *inputs);
void CopyNetworkOutputs(MLPNetwork *nnet, float *outs);

#endif                                                      /* __MLPNNETS_H */
