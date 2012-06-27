/*
 * mlpnnets.c
 * Feed-forward Multi-Layer Perceptron Neural Networks.
 *
 * Andrei de A. Formiga, 2012-03-31
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mlpnnets.h"

// maximum absolute value of initial weight
#define MAX_ABS_WEIGHT   1.6

// module variable that controls showing of debug information during
// training TODO: implement functions to change it?
int debug_train = 0;

// sigmoid activation function
double sigmoid(double t)
{
    return 1.0 / (1.0 + exp(-t));
}

// derivative of the logistic sigmoid function
// --
// this can be directly calculated
// from the node outputs, which is faster
double dsigmoid(double t)
{
    double sigt = sigmoid(t);
    return sigt * (1.0 - sigt);
}

double threshold(double t)
{
    return (t > 0? 1.0 : 0.0);
}


Layer *create_layer(Layer *prev, int n_neurons)
{
    Layer *res = (Layer*) malloc(sizeof(Layer));

    prev->next = res;

    res->n_neurons = n_neurons;
    res->prev = prev;
    res->a = (double*) malloc(sizeof(double) * n_neurons);
    res->y = (double*) malloc(sizeof(double) * n_neurons);

    // allocate weights (including one for bias for each neuron)
    res->w = (double*) malloc(sizeof(double) * n_neurons * (prev->n_neurons+1));
    res->next = NULL;

    return res;
}

void destroy_layer(Layer *layer)
{
    if (layer != NULL) {
        if (layer->w != NULL)
            free(layer->w);

        if (layer->a != NULL)
            free(layer->a);

        if (layer->y != NULL)
            free(layer->y);

        free(layer);
    }
}


Network *create_network(int n_inputs)
{
    Network *res = (Network*) malloc(sizeof(Network));

    res->n_layers = 1;
    res->input_layer = (Layer*) malloc(sizeof(Layer));
    res->input_layer->n_neurons = n_inputs;
    res->input_layer->w = NULL;
    res->input_layer->prev = NULL;
    res->input_layer->next = NULL;
    res->input_layer->a = NULL;
    res->input_layer->y = (double*) malloc(sizeof(double) * n_inputs);

    res->output_layer = res->input_layer;

    return res;
}

Layer* add_layer(Network *nnet, int n_neurons)
{
    Layer *new_layer = create_layer(nnet->output_layer, n_neurons);
    nnet->output_layer = new_layer;
    nnet->n_layers++;

    return new_layer;
}

void destroy_network(Network *nnet)
{
    Layer *l1, *l2;

    l1 = nnet->input_layer;

    while (l1 != NULL) {
        l2 = l1->next;
        destroy_layer(l1);
        l1 = l2;
    }

    free(nnet);
}

void print_network_structure(Network *nnet)
{
    int   i, j;
    Layer *l = nnet->input_layer->next;

    printf("--- Neural network structure ---\n");
    printf("Layers: %d\n", nnet->n_layers);
    printf("Inputs: %d\n", nnet->input_layer->n_neurons);
    printf("Outputs: %d\n", nnet->output_layer->n_neurons);

    i = 0;
    while (l != NULL) {
        printf("Weights[%d]: ", i);
        for (j = 0; j < l->n_neurons * (l->prev->n_neurons+1); ++j)
            printf("%2.1f ", l->w[j]);
        printf("\n");
        l = l->next;
        ++i;
    }

    printf("--------------------------------\n");
}

void initialize_weights(Network *nnet, unsigned int seed)
{
    int r;
    int i, j;
    Layer *l = nnet->input_layer->next;

    srand(seed);

    while (l != NULL) {
        for (i = 0; i < l->n_neurons; ++i)
            for (j = 0; j < l->prev->n_neurons+1; ++j) {
                r = rand();
                W(l, i, j) = (r - (RAND_MAX / 2.0)) *
                             (2.0 * MAX_ABS_WEIGHT / RAND_MAX);
            }
        l = l->next;
    }
}

// propagates the input forward, calculating the network outputs
// activf is the activation function to use on the outputs
// (assumes input is the same size as the number of neurons in input layer)
void forward_prop(Network *nnet, double (*activf)(double), double *input)
{
    int   i, j;
    Layer *prev_layer = nnet->input_layer;
    Layer *layer = prev_layer->next;

    // copy inputs to the input layer neurons
    for (i = 0; i < nnet->input_layer->n_neurons; ++i)
        nnet->input_layer->y[i] = input[i];

    while (layer != NULL) {
        for (i = 0; i < layer->n_neurons; ++i) {
            // compute the bias
            layer->a[i] = W(layer, i, 0) * 1.0;
            // add weights * inputs
            for (j = 1; j < (prev_layer->n_neurons+1); ++j)
                layer->a[i] += W(layer, i, j) * prev_layer->y[j-1];
            // apply activation function
            layer->y[i] = activf(layer->a[i]);
        }
        prev_layer = layer;
        layer = prev_layer->next;
    }
}

double **create_weight_derivatives_matrix(Network *nnet)
{
    int    i;
    int    n_weights;  // number of weights per layer
    double **deriv;
    Layer  *l = nnet->input_layer->next;

    deriv = (double**) malloc(sizeof(double*) * (nnet->n_layers-1));
    if (deriv == NULL)
        return NULL;

    for (i = 0; i < nnet->n_layers-1; ++i, l = l->next) {
        n_weights = l->n_neurons * (l->prev->n_neurons+1);
        deriv[i] = (double*) calloc(n_weights, sizeof(double));
        if (deriv[i] == NULL)
            return NULL;
    }

    return deriv;
}

// create a matrix for deltas
// (could reduce space to keep deltas for only 2 layers at a time)
double **create_delta_matrix(Network *nnet)
{
    int    i;
    double **deltas;
    Layer  *l = nnet->input_layer->next;

    deltas = (double**) malloc(sizeof(double*) * (nnet->n_layers-1));
    if (deltas == NULL)
        return NULL;

    for (i = 0; i < nnet->n_layers-1; ++i, l = l->next) {
        deltas[i] = (double*) malloc(sizeof(double) * l->n_neurons);
        if (deltas[i] == NULL)
            return NULL;
    }

    return deltas;
}

// calculate deltas for the nodes given the desired outputs
// return the sum of square errors for these outputs
double calculate_deltas(Network *nnet, double **delta, double *d_output,
                        double (*dactf)(double))
{
    int    i, j;
    int    ln;     // layer number
    double d, err = 0.0;
    Layer  *l = nnet->output_layer;

    // calculate deltas for output layer
    ln = nnet->n_layers-2;
    for (i = 0; i < l->n_neurons; ++i) {
        d = d_output[i] - l->y[i];
        err += d * d;
        delta[ln][i] = -d * dactf(l->a[i]);
    }

    // calculate deltas for hidden layers
    for (ln--, l=l->prev; ln >=0; ln--, l=l->prev) {
        for (i = 0; i < l->n_neurons; ++i) {
            delta[ln][i] = 0.0;
            for (j = 0; j < l->next->n_neurons; ++j)
                // use i+1-th weight to compensate for bias node
                delta[ln][i] += W(l->next, j, i+1) * delta[ln+1][j];
            delta[ln][i] *= dactf(l->a[i]);
        }
    }

    return err;
}

// given deltas for the nodes,
// calculate the derivatives in relation to the weights
void calculate_derivatives(Network *nnet, double **delta, double **deriv)
{
    int   ln;   // layer number
    int   nn;   // neuron (node) number
    int   wn;   // weight number
    int   ws;   // number of weights per node in layer
    Layer *l = nnet->input_layer->next;

    for (ln = 0; ln < nnet->n_layers-1; ++ln, l=l->next) {
        ws = l->prev->n_neurons + 1;
        for (nn = 0; nn < l->n_neurons; ++nn) {
            // dE_p/dw_{n0}^l = delta_n^l (bias weight)
            deriv[ln][nn*ws] += delta[ln][nn];
            for (wn = 1; wn < ws; ++wn)
                // dE_p/dw_{nw}^l = delta_n^l * y_w^{l-1}
                deriv[ln][nn*ws+wn] += delta[ln][nn] * l->prev->y[wn-1];
        }
    }
}

// train network nnet in batch mode with backpropagation,
// using the provided dataset, for a number of epochs,
// using lrate as the learning rate and actf as
// activation function (dactf is its derivative)
// ---
// return an approximation to the
// final sum-of-squares error for network after training
double batch_train(Network *nnet, DataSet *dset, double lrate, int epochs,
                   double (*actf)(double), double (*dactf)(double))
{
    int    i, j;
    int    ln;
    int    wn, ws;
    double err;
    double **deriv;   // per-weight derivatives
    double **delta;   // per-node delta for derivative calculation
    Layer  *l;

    // check to see if the dataset matches the network
    if (dset->input_size != nnet->input_layer->n_neurons) {
        fprintf(stderr, "batch_train: Size of input in dataset is different \
                         from input layer in neural net\n");
        return 0.0;
    }

    if (dset->output_size != nnet->output_layer->n_neurons) {
        fprintf(stderr, "batch_train: Size of output in dataset is different \
                         from output layer in neural net\n");
        return 0.0;
    }

    deriv = create_weight_derivatives_matrix(nnet);
    delta = create_delta_matrix(nnet);
    // TODO: verify if allocations failed

    for (j = 0; j < epochs; ++j) {
        err = 0.0;
        // clear the deriv matrix
        l = nnet->input_layer->next;
        for (ln = 0; ln < nnet->n_layers-1; ++ln, l=l->next) {
            for (i = 0; i < l->n_neurons * (l->prev->n_neurons+1); ++i)
                deriv[ln][i] = 0.0;
        }

        for (i = 0; i < dset->n_cases; ++i) {
            // propagate forward and calculate network outputs
            forward_prop(nnet, actf, dset->input[i]);

            // calculate deltas for nodes
            err += calculate_deltas(nnet, delta, dset->output[i], dactf);

            // calculate error derivatives for the current input
            calculate_derivatives(nnet, delta, deriv);
        }

        // now derivatives for this epoch have been calculated
        // update weights based on derivatives
        l = nnet->input_layer->next;
        for (ln = 0; ln < nnet->n_layers-1; ++ln, l=l->next) {
            ws = l->prev->n_neurons + 1;
            for (i = 0; i < l->n_neurons; ++i)
                for (wn = 0; wn < ws; ++wn)
                    W(l, i, wn) -= lrate * deriv[ln][i*ws+wn];
        }

        if (debug_train)
            printf("Epoch: %d - Total SSE for epoch: %6.4f\n", j, err);
    }

    return err;
}

// allocate dataset arrays based on the number of cases
// and input/output sizes
void allocate_dataset_arrays(DataSet *dset)
{
    int i;

    // TODO: verify allocation results
    dset->input = (double**) malloc(sizeof(double*) * dset->n_cases);
    dset->output = (double**) malloc(sizeof(double*) * dset->n_cases);

    for (i = 0; i < dset->n_cases; ++i) {
        dset->input[i] = (double*) malloc(sizeof(double) * dset->input_size);
        dset->output[i] = (double*) malloc(sizeof(double) * dset->output_size);
    }
}

// free the double arrays in an array of arrays
void free_arrayarray(double **array, int rows)
{
    int i;

    for (i = 0; i < rows; ++i)
        if (array[i] != NULL)
            free(array[i]);
}

// free the memory occupied by a dataset
void free_dataset(DataSet *dset)
{
    if (dset->input != NULL) {
        free_arrayarray(dset->input, dset->n_cases);
        free(dset->input);
    }

    if (dset->output != NULL) {
        free_arrayarray(dset->output, dset->n_cases);
        free(dset->output);
    }

    free(dset);
}
