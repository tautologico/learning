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

// macro to access weight ij in layer l
#define W(l, i, j)       ( l->w[i * l->prev->n_neurons + j] )

// structure representing a layer in the neural net
typedef struct tagLayer
{
    int n_neurons;                      // number of neurons
    double *w;                          // input weights for the layer
    struct tagLayer *prev;              // previous layer
    struct tagLayer *next;              // next layer
    double *y;                          // activations
} Layer;

// the input layer to the network
Layer *input_layer;


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
    free(layer->w);
    free(layer->y);
    free(layer);
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

    printf("Weights for layer 1:\n");
    for (i = 0; i < 6; ++i)
        printf("%5.2f ", l1->w[i]);

    printf("\n");

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

int main(int argc, char **argv)
{
    basic_test();

    return 0;
}
