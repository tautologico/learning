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
    float *d_outs;
    float *d_deltas;
};

struct MLPNetwork
{
    int      nLayers;
    MLPLayer **layers;
    float    *d_weights;
    int      nWeights;
};

// network functions
MLPNetwork *CreateNetwork(int nLayers, int *neuronsPerLayer);
void DestroyNetwork(MLPNetwork *net);
void RandomWeights(MLPNetwork *net, float max_abs, long seed);
void RandomWeightsGen(MLPNetwork *net, float max_abs, curandGenerator_t gen);

#endif                                                      /* __MLPNNETS_H */
