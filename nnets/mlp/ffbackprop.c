/*
 * ffbackprop.c
 * Implementation of a feed-forward MLP neural network trained
 * with backpropagation.
 *
 * Andrei de A. Formiga, 2012-03-20
 *
 */

// TODO: add bias neurons in addition to the ones created in layers?

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// macro to access weight ij in layer l
#define W(l, i, j)       ( l->w[i * l->prev->n_neurons + j] )

// structure representing a layer in the neural net
typedef struct tagLayer
{
    int    n_neurons;                   // number of neurons
    double *w;                          // input weights for the layer
    struct tagLayer *prev;              // previous layer
    struct tagLayer *next;              // next layer
    double *y;                          // activations
} Layer;

// structure representing a neural network
typedef struct tagNetwork
{
    int   n_layers;       // number of layers in network
    Layer *input_layer;   // the input layer
    Layer *output_layer;  // the output (last) layer
} Network;


// the input layer to the network
Layer *input_layer;


// sigmoid activation function
double sigmoid(double t)
{
    return 1.0 / (1.0 + exp(-t));
}


// initialize structures, given the number of inputs to the network
void init(int n_inputs)
{
    // initialize input layer
    input_layer = malloc(sizeof(Layer));
    input_layer->n_neurons = n_inputs;
    input_layer->w = NULL;
    input_layer->prev = NULL;
    input_layer->next = NULL;
    input_layer->y = (double*) malloc(sizeof(double) * n_inputs);
}

Layer *create_layer(Layer *prev, int n_neurons)
{
    Layer *res = (Layer*) malloc(sizeof(Layer));

    prev->next = res;

    res->n_neurons = n_neurons;
    res->prev = prev;
    res->y = (double*) malloc(sizeof(double) * n_neurons);
    res->w = (double*) malloc(sizeof(double) * n_neurons * prev->n_neurons);
    res->next = NULL;

    // TODO: assign weight to bias neuron? (neuron 0)

    return res;
}

void destroy_layer(Layer *layer)
{
    if (layer != NULL) {
        if (layer->w != NULL)
            free(layer->w);

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
    res->input_layer->y = (double*) malloc(sizeof(double) * n_inputs);

    res->output_layer = res->input_layer;

    return res;
}

void add_layer(Network *nnet, int n_neurons)
{
    Layer *new_layer = create_layer(nnet->output_layer, n_neurons);
    nnet->output_layer = new_layer;
    nnet->n_layers++;
}

void destroy_network(Network *nnet)
{
    Layer *l1, *l2;

    l1 = nnet->input_layer;
    l2 = l1->next;

    while (l1 != NULL) {
        destroy_layer(l1);
        l1 = l2;
        l2 = l2->next;
    }

    free(nnet);
}

void basic_test(void)
{
    int   i;
    Layer *l1, *l2;

    // test
    init(3);
    l1 = create_layer(input_layer, 2);
    l2 = create_layer(l1, 3);

    W(l1, 0, 0) = 1.0;
    W(l1, 0, 1) = 2.0;
    W(l1, 0, 2) = 3.0;
    W(l1, 1, 0) = 4.0;
    W(l1, 1, 1) = 5.0;
    W(l1, 1, 2) = 6.0;

    printf("### Basic layer test\n");
    printf("Weights for layer 1:\n");
    for (i = 0; i < 6; ++i)
        printf("%5.2f ", l1->w[i]);

    printf("\n\n");

    W(l2, 0, 0) = 1.0;
    W(l2, 0, 1) = 2.0;
    W(l2, 1, 0) = 3.0;
    W(l2, 1, 1) = 4.0;
    W(l2, 2, 0) = 5.0;
    W(l2, 2, 1) = 6.0;

    printf("Weights for layer 2:\n");
    for (i = 0; i < 6; ++i)
        printf("%5.2f ", l2->w[i]);

    printf("\n");

    destroy_layer(l1);
    destroy_layer(l2);
    free(input_layer);
}

// propagates the inputs forward, calculating the network outputs
// activf is the activation function to use on the outputs
void forward_prop(Network *nnet, double (*activf)(double))
{
    int   i, j;
    Layer *prev_layer = nnet->input_layer;
    Layer *current_layer = prev_layer->next;
    double a;

    while (current_layer != NULL) {
        a = 0.0;
        for (i = 0; i < current_layer->n_neurons; ++i) {
            for (j = 0; j < prev_layer->n_neurons; ++j)
                a += W(current_layer, i, j);
            current_layer->y[i] = activf(a);
        }
        prev_layer = current_layer;
        current_layer = prev_layer->next;
    }
}

void xor_test(void)
{
    // TODO: implement MLP network with XOR function, test forward
    // propagation
    Network *xor_nn = create_network(2);  // 2 inputs
    Layer   *l1, *l2;    // layers for manually adjusting weights

    // middle layer
    add_layer(xor_nn, 2);

    // output layer
    add_layer(xor_nn, 1);

    l1 = xor_nn->input_layer->next;
    l2 = l1->next;

    // set weights for layer 1
    W(l1, 0, 0) = 0.0;
    W(l1, 0, 1) = 0.0;
    W(l1, 1, 0) = 0.0;
    W(l1, 1, 1) = 0.0;

    // set weights for layer 2
    W(l2, 0, 0) = 0.0;
    W(l2, 0, 1) = 0.0;

    printf("\n### XOR network (forward propagation test)\n");

    // set input values
    xor_nn->input_layer->y[0] = 0.0;
    xor_nn->input_layer->y[1] = 0.0;

    forward_prop(xor_nn, sigmoid);
}

int main(int argc, char **argv)
{
    basic_test();
    xor_test();

    return 0;
}