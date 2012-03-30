/*
 * ffbackprop.c
 * Implementation of a feed-forward MLP neural network trained
 * with backpropagation.
 *
 * Andrei de A. Formiga, 2012-03-20
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// macro to access weight ij in layer l
#define W(l, i, j)       ( l->w[i * (l->prev->n_neurons+1) + j] )

// constant to use as a seed (for reproducibility)
#define SEED             1439021

// maximum absolute value of initial weight
#define MAX_ABS_WEIGHT   1.6


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


// sigmoid activation function
double sigmoid(double t)
{
    return 1.0 / (1.0 + exp(-t));
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

void basic_test(void)
{
    int   i;
    Layer *l1, *l2;
    Network *nnet = create_network(3);

    // test
    l1 = add_layer(nnet, 2);
    l2 = add_layer(nnet, 3);

    W(l1, 0, 0) = 1.0;
    W(l1, 0, 1) = 2.0;
    W(l1, 0, 2) = 3.0;
    W(l1, 0, 3) = 3.5;
    W(l1, 1, 0) = 4.0;
    W(l1, 1, 1) = 5.0;
    W(l1, 1, 2) = 6.0;
    W(l1, 1, 3) = 6.5;

    printf("### Basic layer test\n");
    printf("Weights for layer 1:\n");
    for (i = 0; i < 8; ++i)
        printf("%5.2f ", l1->w[i]);

    printf("\n\n");

    W(l2, 0, 0) = 1.0;
    W(l2, 0, 1) = 2.0;
    W(l2, 1, 0) = 3.0;
    W(l2, 1, 1) = 4.0;
    W(l2, 2, 0) = 5.0;
    W(l2, 2, 1) = 6.0;

    printf("Weights for layer 2:\n");
    for (i = 0; i < 9; ++i)
        printf("%5.2f ", l2->w[i]);

    printf("\n");

    destroy_network(nnet);
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

void xor_test(void)
{
    Network *xor_nn = create_network(2);  // 2 inputs
    Layer   *l1, *l2;    // layers for manually adjusting weights
    double  inputs[2];

    // middle layer
    l1 = add_layer(xor_nn, 2);

    // output layer
    l2 = add_layer(xor_nn, 1);

    // set weights for layer 1
    W(l1, 0, 0) = 0.5;
    W(l1, 0, 1) = -1.0;
    W(l1, 0, 2) = -1.0;

    W(l1, 1, 0) = -1.5;
    W(l1, 1, 1) = 1.0;
    W(l1, 1, 2) = 1.0;

    // set weights for layer 2
    W(l2, 0, 0) = 0.5;
    W(l2, 0, 1) = -1.0;
    W(l2, 0, 2) = -1.0;

    printf("\n### XOR network (forward propagation test)\n");

    // test case (0, 0)
    inputs[0] = 0.0;
    inputs[1] = 0.0;

    forward_prop(xor_nn, threshold, inputs);

    printf("Output for (0, 0) = %2.1f\n", xor_nn->output_layer->y[0]);

    // test case (0, 1)
    inputs[0] = 0.0;
    inputs[1] = 1.0;

    forward_prop(xor_nn, threshold, inputs);

    printf("Output for (0, 1) = %2.1f\n", xor_nn->output_layer->y[0]);

    // test case (1, 0)
    inputs[0] = 1.0;
    inputs[1] = 0.0;

    forward_prop(xor_nn, threshold, inputs);

    printf("Output for (1, 0) = %2.1f\n", xor_nn->output_layer->y[0]);

    // test case (1, 1)
    inputs[0] = 1.0;
    inputs[1] = 1.0;

    forward_prop(xor_nn, threshold, inputs);

    printf("Output for (1, 1) = %2.1f\n\n", xor_nn->output_layer->y[0]);
}

void random_init_test(void)
{
    int   i;
    Layer *l1, *l2;
    Network *nnet = create_network(3);

    // test
    l1 = add_layer(nnet, 2);
    l2 = add_layer(nnet, 3);

    initialize_weights(nnet, SEED);

    printf("### Random initialization test\n");
    printf("Weights for layer 1:\n");
    for (i = 0; i < 8; ++i)
        printf("%5.2f ", l1->w[i]);

    printf("\n\n");

    printf("Weights for layer 2:\n");
    for (i = 0; i < 9; ++i)
        printf("%5.2f ", l2->w[i]);

    printf("\n\n");

    destroy_network(nnet);
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
        deriv[i] = (double*) malloc(sizeof(double) * n_weights);
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

// train network nnet in batch mode,
// using the provided dataset, for a number of epochs,
// using lrate as the learning rate and actf as
// activation function (dactf is its derivative)
void batch_train(Network *nnet, DataSet *dset, double lrate, int epochs,
                 double (*actf)(double), double (*dactf)(double))
{
    int    i, j, k;
    int    lnum;
    double **deriv;   // per-weight derivatives
    double **delta;   // per-node delta for derivative calculation
    double d;
    Layer *l;

    // check to see if the dataset matches the network
    if (dset->input_size != nnet->input_layer->n_neurons) {
        fprintf(stderr, "batch_train: Size of input in dataset is different \
                         from input layer in neural net\n");
        return;
    }

    if (dset->output_size != nnet->output_layer->n_neurons) {
        fprintf(stderr, "batch_train: Size of output in dataset is different \
                         from output layer in neural net\n");
        return;
    }

    deriv = create_weight_derivatives_matrix(nnet);
    delta = create_delta_matrix(nnet);

    for (j = 0; j < epochs; ++j) {
        for (i = 0; i < dset->n_cases; ++i) {
            // propagate forward and calculate network outputs
            forward_prop(nnet, actf, dset->input[i]);

            // calculate deltas for output layer
            l = nnet->output_layer;
            lnum = nnet->n_layers-2;
            for (k = 0; k < l->n_neurons; ++k) {
                d = dset->output[i][k] - l->y[k];
                delta[lnum][k] = -d * dactf(l->a[k]);
            }

            // calculate deltas for hidden layers
            for (lnum--, l=l->prev; lnum >=0; lnum--, l=l->prev) {
                // TODO: calculate deltas for hidden layers
                for (k = 0; k < l->n_neurons; ++k) {
                }
            }

            // TODO: calculate error derivatives from deltas
        }
    }

}

int main(int argc, char **argv)
{
    basic_test();
    xor_test();
    random_init_test();

    return 0;
}
