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

int main(int argc, char **argv)
{
    DataSet *training;
    DataSet *test;

    training = read_dataset("iris.train");


    return 0;
}
