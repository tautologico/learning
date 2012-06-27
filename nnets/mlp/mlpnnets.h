/*
 * mlpnnets.h
 * Feed-forward Multi-Layer Perceptron Neural Networks.
 *
 * Andrei de A. Formiga, 2012-03-31
 *
 */

#ifndef __MLPNNETS_H

#define __MLPNNETS_H

// macro to access weight ij in layer l
#define W(l, i, j)       ( l->w[i * (l->prev->n_neurons+1) + j] )

// structure representing a layer in the neural net
typedef struct tagLayer
{
    int    n_neurons;                   // number of neurons
    double *w;                          // input weights for the layer
    struct tagLayer *prev;              // previous layer
    struct tagLayer *next;              // next layer
    double *a;                          // activations
    double *y;                          // node outputs
} Layer;

// structure representing a neural network
typedef struct tagNetwork
{
    int   n_layers;       // number of layers in network
    Layer *input_layer;   // the input layer
    Layer *output_layer;  // the output (last) layer
} Network;

// structure for a dataset
typedef struct tagDataSet
{
    int    n_cases;       // number of cases
    int    input_size;    // size of input in each case
    int    output_size;   // size of output in each case
    double **input;       // inputs
    double **output;      // outputs
} DataSet;

// activation functions
double sigmoid(double t);
double dsigmoid(double t);
double threshold(double t);

// network functions
Network *create_network(int n_inputs);
Layer* add_layer(Network *nnet, int n_neurons);
void destroy_network(Network *nnet);
void print_network_structure(Network *nnet);
void initialize_weights(Network *nnet, unsigned int seed);
void forward_prop(Network *nnet, double (*activf)(double), double *input);
double batch_train(Network *nnet, DataSet *dset, double lrate, int epochs,
                   double (*actf)(double), double (*dactf)(double));
void allocate_dataset_arrays(DataSet *dset);
void free_dataset(DataSet *dset);

#endif                                    /* __MLPNETS_H */
