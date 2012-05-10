/*

  mlpnnets.cu
  Header file for FFMLP implementation in CUDA

  Andrei de A. Formiga, 2012-05-09

 */

#ifndef __MLPNNETS_H

#define __MLPNNETS_H

struct MLPLayer
{
    int n_neurons;
}

struct MLPNet
{
    int n_layers;
    Layer[] layers;
}

#endif                                                      /* __MLPNNETS_H */
