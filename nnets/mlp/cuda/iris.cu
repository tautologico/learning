/*

  iris.cu
  Classification of the iris dataset from Fisher using neural networks
  implemented in CUDA. 

  Andrei de A. Formiga, 2012-05-21

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlpnnets.h"

#define SEED                  631814ULL

#define MAX_ABS               1.2f

#define EPOCHS                7000
#define LEARNING_RATE         0.003f

#define MAX(a, b)             (a >= b? a : b)

// neurons per layer (4 inputs, 8 hidden, 3 outputs)
int neuronsPerLayer[] = { 4, 8, 3 };

typedef enum tagClass {
    iris_setosa,
    iris_versicolor,
    iris_virginica
} Class;

DataSet* read_dataset(char *filename)
{
    FILE    *f;
    int     i, j;
    double  slen, swid, plen, pwid;
    char    buffer[140];
    DataSet *dset;

    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "File not found: %s\n", filename);
        return NULL;
    }

    // count lines in file to allocate dataset arrays
    i = 0;
    while (fgets(buffer, 140, f) != NULL)
        ++i;

    if (!feof(f) || ferror(f)) {
        fprintf(stderr, "IO error while reading from file\n");
        fclose(f);
        return NULL;
    }
    fseek(f, 0, SEEK_SET);

    dset = CreateDataSet(i, 4, 3);

    if (dset == NULL) {
        fprintf(stderr, "Error creating dataset\n"); 
        return NULL;
    }

    int  iix = 0, oix = 0;
    bool done = false;
    while (!done) {
        j = fscanf(f, "%lf,%lf,%lf,%lf,%s\n", &slen, &swid,
                   &plen, &pwid, buffer);

        if (j != 5)
            done = true;
        else {
            //printf("%f, %f, %f, %f\n", slen, swid, plen, pwid);
            dset->inputs[iix++] = slen;
            dset->inputs[iix++] = swid;
            dset->inputs[iix++] = plen;
            dset->inputs[iix++] = pwid;

            if (strstr(buffer, "setosa")) {
                dset->outputs[oix++] = 0.9;
                dset->outputs[oix++] = 0.1;
                dset->outputs[oix++] = 0.1;
            } else if (strstr(buffer, "versicolor")) {
                dset->outputs[oix++] = 0.1;
                dset->outputs[oix++] = 0.9;
                dset->outputs[oix++] = 0.1;
            } else { // assume class "virginica"
                dset->outputs[oix++] = 0.1;
                dset->outputs[oix++] = 0.1;
                dset->outputs[oix++] = 0.9;
            }
        }
    }

    fclose(f);

    return dset;
}

void print_dataset(DataSet *dset)
{
    int i, j;

    printf("Number of cases: %d\n", dset->nCases);
    for (i = 0; i < dset->nCases; ++i) {
        for (j = 0; j < dset->inputSize; ++j)
            printf("%3.2f ", dset->inputs[i*dset->inputSize+j]);
        printf(" | ");
        for (j = 0; j < dset->outputSize; ++j)
            printf("%3.2f ", dset->outputs[i*dset->outputSize+j]);
        printf("\n");
    }
}

Class output_to_class(float *output)
{
    double max;

    max = MAX(output[0], MAX(output[1], output[2]));
    if (output[0] == max)
        return iris_setosa;
    else if (output[1] == max)
        return iris_versicolor;

    return iris_virginica;
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

void print_network_data(MLPNetwork *net)
{
    printf("nLayers = %d, d_weights = %lu, nWeights = %d, nCases = %d\n",
           net->nLayers, (unsigned long) net->d_weights, net->nWeights, net->nCases);
    printf("output ptr for first layer: %lu\n", (unsigned long) net->layers[0]->d_outs);
    printf("output ptr for last layer: %lu\n", (unsigned long) net->layers[net->nLayers-1]->d_outs);
}

int main(int argc, char **argv)
{
    int     i;
    int     errors;
    DataSet *train_set;
    DataSet *test_set;
    float   e;
    double  acc;
    Class   predicted, desired;

    MLPNetwork *irisnn;

    // training
    train_set = read_dataset("iris.train");

    if (train_set == NULL) {
        fprintf(stderr, "Error reading training set\n");
        exit(1);
    }

    irisnn = CreateNetwork(3, neuronsPerLayer);
    RandomWeights(irisnn, MAX_ABS, SEED);

    printf("Training network with %d epochs...\n", EPOCHS);
    e = BatchTrainBackprop(irisnn, train_set, EPOCHS, LEARNING_RATE, 1, 0);
    printf("Training finished, approximate final SSE: %f\n", e);

    printf("Weights after training:\n");
    PrintWeights(irisnn);

    printf("-----------------------------------------\n");

    // free the training dataset
    cudaThreadSynchronize();
    DestroyDataSet(train_set);

    // testing
    test_set = read_dataset("iris.test");

    if (test_set == NULL) {
        fprintf(stderr, "Error reading test set\n");
        return -1;
    }

    errors = 0;

    if (!PrepareForTesting(irisnn, test_set->nCases)) {
        fprintf(stderr, "Error preparing network for testing\n");
        return -1;
    }

    printf("Testing with %d cases...\n", test_set->nCases);
    PresentInputsFromDataSet(irisnn, test_set, ACTF_SIGMOID);

    cudaThreadSynchronize();

    printf("Weights again:\n");
    PrintWeights(irisnn);    

    float *output = (float*) malloc(sizeof(float) * test_set->nCases * test_set->outputSize);

    if (output == NULL) {
        fprintf(stderr, "Could not allocate memory for copying output to host\n");
        return -1;
    }

    if (!CopyNetworkOutputs(irisnn, output)) {
        fprintf(stderr, "Could not get device outputs\n");
        return -1;
    }

    for (i = 0; i < test_set->nCases; ++i) {
        predicted = output_to_class(output + (i * test_set->outputSize));
        desired = output_to_class(test_set->outputs + (i * test_set->outputSize));
        if (predicted != desired)
            ++errors;
        printf("Case %d | predicted: %s, desired: %s, outputs: %4.3f %4.3f %4.3f\n", i,
               class_to_string(predicted), class_to_string(desired),
               output[i*test_set->outputSize], output[i*test_set->outputSize+1], 
               output[i*test_set->outputSize+2]);
    }

    free(output);

    acc = 100.0 - (100.0 * errors / test_set->nCases);
    printf("Testing accuracy: %f\n", acc);
    printf("Total classificarion errors: %d\n", errors);

    DestroyNetwork(irisnn);
    DestroyDataSet(test_set);

    return 0;
}
