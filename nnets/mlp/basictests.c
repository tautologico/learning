/*
 * basictests.c
 * Basic tests of the Feed-forward MLP neural net implementation.
 *
 * Andrei de A. Formiga, 2012-03-30
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mlpnnets.h"

// constant to use as a seed (for reproducibility)
#define SEED             1439021


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

void xor_train_test(void)
{
    int     i;
    Network *xor_nn = create_network(2);  // 2 inputs
    Layer   *l1, *l2;    // layers for manually adjusting weights
    double  inputs[2];
    double  err;
    DataSet dset;

    // middle layer
    l1 = add_layer(xor_nn, 2);

    // output layer
    l2 = add_layer(xor_nn, 1);

    // initialize weights
    initialize_weights(xor_nn, SEED);

    // prepare dataset
    dset.n_cases = 4;
    dset.input_size = 2;
    dset.output_size = 1;
    dset.input = (double**) malloc(sizeof(double*) * dset.n_cases);
    dset.output = (double**) malloc(sizeof(double*) * dset.n_cases);
    for (i = 0; i < dset.n_cases; ++i) {
        dset.input[i] = (double*) malloc(sizeof(double) * dset.input_size);
        dset.output[i] = (double*) malloc(sizeof(double) * dset.output_size);
    }

    dset.input[0][0] = 0.0;
    dset.input[0][1] = 0.0;
    dset.output[0][0] = 0.0;

    dset.input[1][0] = 0.0;
    dset.input[1][1] = 1.0;
    dset.output[1][0] = 1.0;

    dset.input[2][0] = 1.0;
    dset.input[2][1] = 0.0;
    dset.output[2][0] = 1.0;

    dset.input[3][0] = 1.0;
    dset.input[3][1] = 1.0;
    dset.output[3][0] = 0.0;

    printf("\n### Training a XOR network\n");


    printf("Batch training with backpropagation, using 5000 epochs...\n");
    err = batch_train(xor_nn, &dset, 0.75, 5000, sigmoid, dsigmoid);

    printf("Training concluded, approx. SSE = %f\n", err);
    printf("Testing trained network:\n");

    // test case (0, 0)
    inputs[0] = 0.0;
    inputs[1] = 0.0;

    forward_prop(xor_nn, sigmoid, inputs);

    printf("- Output for (0, 0) = %6.4f\n", xor_nn->output_layer->y[0]);

    // test case (0, 1)
    inputs[0] = 0.0;
    inputs[1] = 1.0;

    forward_prop(xor_nn, sigmoid, inputs);

    printf("- Output for (0, 1) = %6.4f\n", xor_nn->output_layer->y[0]);

    // test case (1, 0)
    inputs[0] = 1.0;
    inputs[1] = 0.0;

    forward_prop(xor_nn, sigmoid, inputs);

    printf("- Output for (1, 0) = %6.4f\n", xor_nn->output_layer->y[0]);

    // test case (1, 1)
    inputs[0] = 1.0;
    inputs[1] = 1.0;

    forward_prop(xor_nn, sigmoid, inputs);

    printf("- Output for (1, 1) = %6.4f\n\n", xor_nn->output_layer->y[0]);

    print_network_structure(xor_nn);
}

int main(int argc, char **argv)
{
    basic_test();
    xor_test();
    random_init_test();
    xor_train_test();

    return 0;
}
