/*
 * testiris.c
 * Test the implementation of MLP neural nets as a classificator,
 * using the famous Iris dataset from Fisher.
 *
 * Andrei de A. Formiga, 2012-03-31
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlpnnets.h"

#define TRUE    1
#define FALSE   0

#define SEED                  631814

#define EPOCHS                7000
#define LEARNING_RATE         0.003

#define MAX(a, b)             (a >= b? a : b)

typedef enum tagClass {
    iris_setosa,
    iris_versicolor,
    iris_virginica
} Class;

DataSet* read_dataset(char *filename)
{
    FILE    *f;
    int     done = FALSE;
    int     i, j;
    double  slen, swid, plen, pwid;
    char    buffer[140];
    DataSet *dset;

    dset = (DataSet*) malloc(sizeof(DataSet));
    if (dset == NULL) {
        fprintf(stderr, "Could not allocate memory\n");
        return NULL;
    }

    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "File not found: %s\n", filename);
        free(dset);
        return NULL;
    }

    // count lines in file to allocate dataset arrays
    i = 0;
    while (fgets(buffer, 140, f) != NULL)
        ++i;

    if (!feof(f) || ferror(f)) {
        fprintf(stderr, "IO error while reading from file\n");
        free(dset);
        fclose(f);
        return NULL;
    }
    fseek(f, 0, SEEK_SET);

    // prepare dataset
    dset->n_cases = i;
    dset->input_size = 4;
    dset->output_size = 3;
    allocate_dataset_arrays(dset);

    i = 0;
    while (!done) {
        j = fscanf(f, "%lf,%lf,%lf,%lf,%s\n", &slen, &swid,
                   &plen, &pwid, buffer);

        if (j != 5)
            done = TRUE;
        else {
            //printf("%f, %f, %f, %f\n", slen, swid, plen, pwid);
            dset->input[i][0] = slen;
            dset->input[i][1] = swid;
            dset->input[i][2] = plen;
            dset->input[i][3] = pwid;

            if (strstr(buffer, "setosa")) {
                dset->output[i][0] = 0.9;
                dset->output[i][1] = 0.1;
                dset->output[i][2] = 0.1;
            } else if (strstr(buffer, "versicolor")) {
                dset->output[i][0] = 0.1;
                dset->output[i][1] = 0.9;
                dset->output[i][2] = 0.1;
            } else { // assume class "virginica"
                dset->output[i][0] = 0.1;
                dset->output[i][1] = 0.1;
                dset->output[i][2] = 0.9;
            }
            ++i;
        }
    }

    fclose(f);

    return dset;
}

void print_dataset(DataSet *dset)
{
    int i, j;

    printf("Number of cases: %d\n", dset->n_cases);
    for (i = 0; i < dset->n_cases; ++i) {
        for (j = 0; j < dset->input_size; ++j)
            printf("%3.2f ", dset->input[i][j]);
        printf(" | ");
        for (j = 0; j < dset->output_size; ++j)
            printf("%3.2f ", dset->output[i][j]);
        printf("\n");
    }
}

Class output_to_class(double *output)
{
    double max;

    max = MAX(output[0], MAX(output[1], output[2]));
    if (output[0] == max)
        return iris_setosa;
    else if (output[1] == max)
        return iris_versicolor;

    return iris_virginica;
}

Class predict_class(Network *nnet, double *input)
{
    forward_prop(nnet, sigmoid, input);
    return output_to_class(nnet->output_layer->y);
}


char    *setosa = "setosa";
char    *versicolor = "versicolor";
char    *virginica = "virginica";

char *class_to_string(Class c)
{
    char *res;

    switch(c) {
    case iris_setosa:
        res = setosa;
        break;

    case iris_versicolor:
        res = versicolor;
        break;

    default:
        res = virginica;
    }

    return res;
}

int main(int argc, char **argv)
{
    int     i;
    int     errors;
    DataSet *train_set;
    DataSet *test_set;
    Network *irisnn = create_network(4);
    double  e;
    double  acc;
    Class   predicted, desired;

    // training
    train_set = read_dataset("iris.train");

    if (train_set == NULL) {
        fprintf(stderr, "Error reading training set\n");
        exit(1);
    }

    add_layer(irisnn, 8);  // hidden layer
    add_layer(irisnn, 3);  // output layer
    initialize_weights(irisnn, SEED);
    print_network_structure(irisnn);

    printf("Training network with %d epochs...\n", EPOCHS);
    e = batch_train(irisnn, train_set, LEARNING_RATE, EPOCHS,
                    sigmoid, dsigmoid);
    printf("Training finished, approximate final SSE: %f\n", e);

    print_network_structure(irisnn);

    // testing
    test_set = read_dataset("iris.test");

    if (test_set == NULL) {
        fprintf(stderr, "Error reading test set\n");
        exit(1);
    }

    errors = 0;
    printf("Testing with %d cases...\n", test_set->n_cases);
    for (i = 0; i < test_set->n_cases; ++i) {
        predicted = predict_class(irisnn, test_set->input[i]);
        desired = output_to_class(test_set->output[i]);
        if (predicted != desired)
            ++errors;
        printf("Case %d | predicted: %s, desired: %s, outputs: %4.3f %4.3f %4.3f\n", i,
               class_to_string(predicted), class_to_string(desired),
               irisnn->output_layer->y[0], irisnn->output_layer->y[1], irisnn->output_layer->y[2]);
    }

    acc = 100.0 - (100.0 * errors / test_set->n_cases);
    printf("Testing accuracy: %f\n", acc);
    printf("Total classificarion errors: %d\n", errors);

    // cleanup
    free_dataset(train_set);
    free_dataset(test_set);
    destroy_network(irisnn);

    return 0;
}
