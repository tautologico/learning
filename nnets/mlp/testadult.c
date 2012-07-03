/*
 * readAdult.c
 *
 * Sáskya Gurgel, 2012-04-02
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlpnnets.h"

#define TRUE    1
#define FALSE   0
#define INTERROGACAO 0.0

#define SEED                  631814

#define EPOCHS                3000
#define LEARNING_RATE         0.006

#define MAX(a, b)             (a >= b? a : b)

double string_to_double_workclass(char *c);
double string_to_double_education(char *c);
double string_to_double_marital_status(char *c);
double string_to_double_relationship(char *c);
double string_to_double_occupation(char *c);
double string_to_double_race(char *c);
double string_to_double_sex(char *c);
double string_to_double_native_country(char *c);

typedef enum tagClass {
    maior_50k,
    menorIgual_50k
} Class;

void destroy_dataset(DataSet *data)
{
    if (data != NULL) {
        if (data->input != NULL) {
            free(data->input);
            data->input = NULL;
        }

        if (data->output != NULL) {
            free(data->output);
            data->output = NULL;
        }

        free(data);
    }
}

DataSet* read_dataset(char *filename)
{
    FILE    *f;
    int     done = FALSE, i, j, k;
    double  fnlwgt, education_num, capital_gain, capital_loss;
    double  hours_per_week, age;
    char    workclass[40],  education[40],  marital_status[40], occupation[40];
    char    relationship[40], race[40], sex[40], native_country[40];
    char    buffer[240];
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
    while (fgets(buffer, 240, f) != NULL)
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
    dset->input_size = 14;
    dset->output_size = 2;
    allocate_dataset_arrays(dset);

    i = 0;
    while (!done) {
        j = fscanf(f, "%lf, %s %lf, %s %lf, %s %s %s %s %s %lf, %lf, %lf, %s %s\n",
                   &age, &workclass, &fnlwgt, education, &education_num,
                   marital_status, occupation, relationship, &race, &sex,
                   &capital_gain, &capital_loss, &hours_per_week,
                   native_country, buffer);
        /*printf("%3.2lf; %s; %3.2lf; %s; %3.2lf; %s; %s; %s; %s; %s; %3.2lf; %3.2lf; %3.2lf; %s; %s\n", age, workclass, fnlwgt, education, education_num,
                     marital_status, occupation, relationship, race, sex, capital_gain,
                     capital_loss, hours_per_week, native_country, buffer);*/

        if (j != 15)
            done = TRUE;
        else {

            dset->input[i][0] = age;
            dset->input[i][1] = string_to_double_workclass(workclass);
            dset->input[i][2] = fnlwgt;
            dset->input[i][3] = string_to_double_education(education);
            dset->input[i][4] = education_num;
            dset->input[i][5] = string_to_double_marital_status(marital_status);
            dset->input[i][6] = string_to_double_occupation(occupation);
            dset->input[i][7] = string_to_double_relationship(relationship);
            dset->input[i][8] = string_to_double_race(race);
            dset->input[i][9] = string_to_double_sex(sex);
            dset->input[i][10] = capital_gain;
            dset->input[i][11] = capital_loss;
            dset->input[i][12] = hours_per_week;
            dset->input[i][13] = string_to_double_native_country(native_country);

            if (strstr(buffer, "<=50K")) {
                dset->output[i][0] = 0.9;
                dset->output[i][1] = 0.1;
            } else {
                dset->output[i][0] = 0.1;
                dset->output[i][1] = 0.9;
            }
            ++i;

        }

    }

    if (i != dset->n_cases)
        fprintf(stderr, "Error reading dataset: could not read all expected cases. Expected %d, got %d\n",
                dset->n_cases, i);

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

#define FEMALE 1.0
#define MALE 2.0

double string_to_double_sex(char *c)
{
    double res;

    if(strstr(c,"Female,"))
        res = FEMALE;
    else if(strstr(c, "Male,"))
        res = MALE;
    else res = INTERROGACAO;

    return res;
}

#define WHITE 1.0
#define ASIAN_PAC_ISLANDER 2.0
#define AMER_INDIAN_ESKIMO 3.0
#define OTHER 4.0
#define BLACK 5.0

double string_to_double_race(char *c)
{
    double res;

    if(strstr(c,"White,"))
        res = WHITE;
    else if(strstr(c, "Asian-Pac-Islander,"))
        res = ASIAN_PAC_ISLANDER;
    else if(strstr(c, "Amer-Indian-Eskimo,"))
        res = AMER_INDIAN_ESKIMO;
    else if(strstr(c, "Other,"))
        res = OTHER;
    else if(strstr(c, "Black,"))
        res = BLACK;
    else res = INTERROGACAO;

    return res;
}

#define TECH_SUPPORT 1.0
#define CRAFT_REPAIR 2.0
#define OTHER_SERVICE 3.0
#define SALES 4.0
#define EXEC_MANAGERIAL 5.0
#define PROF_SPECIALTY 6.0
#define HANDLERS_CLEANERS 7.0
#define MACHINE_OP_INSPCT 8.0
#define ADM_CLERICAL 9.0
#define FARMING_FISHING 10.0
#define TRANSPORT_MOVING 11.0
#define PRIV_HOUSE_SERV 12.0
#define PROTECTIVE_SERV 13.0
#define ARMED_FORCES 14.0


double string_to_double_occupation(char *c)
{
    double res;

    if(strstr(c,"Tech-support,"))
        res = TECH_SUPPORT;
    else if(strstr(c, "Craft-repair,"))
        res = CRAFT_REPAIR;
    else if(strstr(c, "Other-service,"))
        res = OTHER_SERVICE;
    else if(strstr(c, "Sales,"))
        res = SALES;
    else if(strstr(c, "Exec-managerial,"))
        res = EXEC_MANAGERIAL;
    else if(strstr(c, "Prof-specialty,"))
        res = PROF_SPECIALTY;
    else if(strstr(c, "Handlers-cleaners,"))
        res = HANDLERS_CLEANERS;
    else if(strstr(c, "Machine-op-inspct,"))
        res = MACHINE_OP_INSPCT;
    else if(strstr(c, "Adm-clerical,"))
        res = ADM_CLERICAL;
    else if(strstr(c, "Farming-fishing,"))
        res = FARMING_FISHING;
    else if(strstr(c, "Transport-moving,"))
        res = TRANSPORT_MOVING;
    else if(strstr(c, "Priv-house-serv,"))
        res = PRIV_HOUSE_SERV;
    else if(strstr(c, "Protective-serv,"))
        res = PROTECTIVE_SERV;
    else if(strstr(c, "Armed-Forces,"))
        res = ARMED_FORCES;
    else res = INTERROGACAO;

    return res;
}

#define WIFE 1.0
#define OWN_CHILD 2.0
#define HUSBAND 3.0
#define NOT_IN_FAMILY 4.0
#define OTHER_RELATIVE 5.0
#define UNMARRIED 6.0

double string_to_double_relationship(char *c)
{
    double res;

    if(strstr(c,"Wife,"))
        res = WIFE;
    else if(strstr(c, "Own-child,"))
        res = OWN_CHILD;
    else if(strstr(c, "Husband,"))
        res = HUSBAND;
    else if(strstr(c, "Not-in-family,"))
        res = NOT_IN_FAMILY;
    else if(strstr(c, "Other-relative,"))
        res = OTHER_RELATIVE;
    else if(strstr(c, "Unmarried,"))
        res = UNMARRIED;
    else res = INTERROGACAO;

    return res;
}

#define MARRIED_CIV_SPOUSE 1.0
#define DIVORCED 2.0
#define NEVER_MARRIED 3.0
#define SEPARATED 4.0
#define WIDOWED 5.0
#define MARRIED_SPOUSE_ABSENT 6.0
#define MARRIED_AF_SPOUSE 7.0

double string_to_double_marital_status(char *c)
{
    double res;

    if(strstr(c,"Married-civ-spouse,"))
        res = MARRIED_CIV_SPOUSE;
    else if(strstr(c, "Divorced,"))
        res = DIVORCED;
    else if(strstr(c, "Never-married,"))
        res = NEVER_MARRIED;
    else if(strstr(c, "Separated,"))
        res = SEPARATED;
    else if(strstr(c, "Widowed,"))
        res = WIDOWED;
    else if(strstr(c, "Married-spouse-absent,"))
        res = MARRIED_SPOUSE_ABSENT;
    else if(strstr(c, "Married-AF-spouse,"))
        res = MARRIED_AF_SPOUSE;
    else res = INTERROGACAO;

    return res;
}

#define INTERROGACAO 0.0
#define BACHELORS 1.0
#define SOME_COLLEGE 2.0
#define N11TH 3.0
#define HS_GRAD 4.0
#define PROF_SCHOOL 5.0
#define ASSOC_ACDM 6.0
#define ASSOC_VOC 7.0
#define N9TH 8.0
#define N7TH_8TH 9.0
#define N12TH 10.0
#define MASTERS 11.0
#define N1ST_4TH 12.0
#define N10TH 13.0
#define DOCTORATE 14.0
#define N5TH_6TH 15.0
#define PRESCHOOL 16.0


double string_to_double_education(char *c)
{
    double res;

    if(strstr(c,"Bachelors,"))
        res = BACHELORS;
    else if(strstr(c, "Some-college,"))
        res = SOME_COLLEGE;
    else if(strstr(c, "11th,"))
        res = N11TH;
    else if(strstr(c, "HS-grad,"))
        res = HS_GRAD;
    else if(strstr(c, "Prof-school,"))
        res = PROF_SCHOOL;
    else if(strstr(c, "Assoc-acdm,"))
        res = ASSOC_ACDM;
    else if(strstr(c, "Assoc-voc,"))
        res = ASSOC_VOC;
    else if(strstr(c, "9th,"))
        res = N9TH;
    else if(strstr(c, "7th-8th,"))
        res = N7TH_8TH;
    else if(strstr(c, "12th,"))
        res = N12TH;
    else if(strstr(c, "Masters,"))
        res = MASTERS;
    else if(strstr(c, "1st-4th,"))
        res = N1ST_4TH;
    else if(strstr(c, "10th,"))
        res = N10TH;
    else if(strstr(c, "Doctorate,"))
        res = DOCTORATE;
    else if(strstr(c, "5th-6th,"))
        res = N5TH_6TH;
    else if(strstr(c, "Preschool,"))
        res = PRESCHOOL;
    else res = INTERROGACAO;
    return res;
}

#define PRIVATE 1.0
#define SELF_EMP_NOT_INC 2.0
#define SELF_EMP_INC 3.0
#define FEDERAL_GOV 4.0
#define LOCAL_GOV 5.0
#define STATE_GOV 6.0
#define WITHOUT_PAY 7.0
#define NEVER_WORKED 8.0

double string_to_double_workclass(char *c)
{
    double res;

    if(strstr(c,"Private,"))
        res = PRIVATE;
    else if(strstr(c, "Self-emp-not-inc,"))
        res = SELF_EMP_NOT_INC;
    else if(strstr(c, "Self-emp-inc,"))
        res = SELF_EMP_INC;
    else if(strstr(c, "Federal-gov,"))
        res = FEDERAL_GOV;
    else if(strstr(c, "Local-gov,"))
        res = LOCAL_GOV;
    else if(strstr(c, "State-gov,"))
        res = STATE_GOV;
    else if(strstr(c, "Without-pay,"))
        res = WITHOUT_PAY;
    else if(strstr(c, "Never-worked,"))
        res = NEVER_WORKED;
    else res = INTERROGACAO;

    return res;
}

#define UNITED_STATES 1.0
#define CAMBODIA 2.0
#define ENGLAND 3.0
#define PUERTO_RICO 4.0
#define CANADA 5.0
#define GERMANY 6.0
#define OUTLYING_US__GUAM_USVI_ETC__ 7.0
#define INDIA 8.0
#define JAPAN 9.0
#define GREECE 10.0
#define SOUTH 11.0
#define CHINA 12.0
#define CUBA 13.0
#define IRAN 14.0
#define HONDURAS 15.0
#define PHILIPPINES 16.0
#define ITALY 17.0
#define POLAND 18.0
#define JAMAICA 19.0
#define VIETNAM 20.0
#define MEXICO 21.0
#define PORTUGAL 22.0
#define IRELAND 23.0
#define FRANCE 24.0
#define DOMINICAN_REPUBLIC 25.0
#define LAOS 26.0
#define ECUADOR 27.0
#define TAIWAN 28.0
#define HAITI 29.0
#define COLUMBIA 30.0
#define HUNGARY 31.0
#define GUATEMALA 32.0
#define NICARAGUA 33.0
#define SCOTLAND 34.0
#define THAILAND 35.0
#define YUGOSLAVIA 36.0
#define EL_SALVADOR 37.0
#define TRINADAD_TOBAGO 38.0
#define PERU 39.0
#define HONG 40.0
#define HOLAND_NETHERLANDS 41.0

double string_to_double_native_country(char *c)
{
    double res;

    if(strstr(c,"United-States,"))
        res = UNITED_STATES;
    else if(strstr(c, "Cambodia,"))
        res = CAMBODIA;
    else if(strstr(c, "England,"))
        res = ENGLAND;
    else if(strstr(c, "Puerto-Rico,"))
        res = PUERTO_RICO;
    else if(strstr(c, "Canada,"))
        res = CANADA ;
    else if(strstr(c, "Germany,"))
        res = GERMANY;
    else if(strstr(c, "Outlying-US(Guam-USVI-etc),"))
        res = OUTLYING_US__GUAM_USVI_ETC__;
    else if(strstr(c, "India,"))
        res = INDIA;
    else if(strstr(c, "Japan,"))
        res = JAPAN;
    else if(strstr(c, "Greece,"))
        res = GREECE;
    else if(strstr(c, "South,"))
        res = SOUTH;
    else if(strstr(c, "China,"))
        res = CHINA;
    else if(strstr(c, "Cuba,"))
        res = CUBA;
    else if(strstr(c, "Iran,"))
        res = IRAN;
    else if(strstr(c, "Honduras,"))
        res = HONDURAS;
    else if(strstr(c, "Philippines,"))
        res = PHILIPPINES;
    else if(strstr(c, "Italy,"))
        res = ITALY;
    else if(strstr(c, "Poland,"))
        res = POLAND;
    else if(strstr(c, "Jamaica,"))
        res = JAMAICA;
    else if(strstr(c, "Vietnam,"))
        res = VIETNAM;
    else if(strstr(c, "Mexico,"))
        res = MEXICO;
    else if(strstr(c, "Portugal,"))
        res = PORTUGAL;
    else if(strstr(c, "Ireland,"))
        res = IRELAND;
    else if(strstr(c, "France,"))
        res = FRANCE;
    else if(strstr(c, "Dominican-Republic,"))
        res = DOMINICAN_REPUBLIC;
    else if(strstr(c, "Laos,"))
        res = LAOS;
    else if(strstr(c, "Ecuador,"))
        res = ECUADOR;
    else if(strstr(c, "Taiwan,"))
        res = TAIWAN;
    else if(strstr(c, "Haiti,"))
        res = HAITI;
    else if(strstr(c, "Columbia,"))
        res = COLUMBIA;
    else if(strstr(c, "Hungary,"))
        res = HUNGARY;
    else if(strstr(c, "Guatemala,"))
        res = GUATEMALA;
    else if(strstr(c, "Nicaragua,"))
        res = NICARAGUA;
    else if(strstr(c, "Scotland,"))
        res = SCOTLAND;
    else if(strstr(c, "Thailand,"))
        res = THAILAND;
    else if(strstr(c, "Yugoslavia,"))
        res = YUGOSLAVIA;
    else if(strstr(c, "El-Salvador,"))
        res = EL_SALVADOR;
    else if(strstr(c, "Trinadad&Tobago,"))
        res = TRINADAD_TOBAGO;
    else if(strstr(c, "Peru,"))
        res = PERU;
    else if(strstr(c, "Hong,"))
        res = HONG;
    else if(strstr(c, "Holand-Netherlands,"))
        res = HOLAND_NETHERLANDS;
    else res = INTERROGACAO;

    return res;
}


Class output_to_class(double *output)
{
    double max;

    max = MAX(output[0], output[1]);
    if (output[0] == max)
        return maior_50k;
    return menorIgual_50k;
}

Class predict_class(Network *nnet, double *input)
{
    forward_prop(nnet, sigmoid, input);
    return output_to_class(nnet->output_layer->y);
}


char    *maior50k = ">50k";
char    *menorIgual50k = "<=50k";

char *class_to_string(Class c)
{
    char *res;

    switch(c) {
    case menorIgual_50k:
        res = menorIgual50k;
        break;
    default:
        res = maior50k;
    }

    return res;
}

int main(int argc, char **argv)
{
    int     i;
    int     errors;
    DataSet *train_set;
    DataSet *test_set;
    Network *adultnn = create_network(14);
    double  e;
    double  acc;
    Class   predicted, desired;

    // training
    train_set = read_dataset("adult.train");

    if (train_set == NULL) {
        fprintf(stderr, "Error reading training set\n");
        exit(1);
    }

    add_layer(adultnn, 28);  // hidden layer
    add_layer(adultnn, 2);  // output layer
    initialize_weights(adultnn, SEED);
    print_network_structure(adultnn);

    printf("Training network with %d epochs...\n", EPOCHS);
    e = batch_train(adultnn, train_set, LEARNING_RATE, EPOCHS,
                   sigmoid, dsigmoid);
    printf("Training finished, approximate final SSE: %f\n", e);

    print_network_structure(adultnn);

    // testing
    test_set = read_dataset("adult.test");

    if (test_set == NULL) {
        fprintf(stderr, "Error reading test set\n");
        exit(1);
    }

    errors = 0;
    printf("Testing with %d cases...\n", test_set->n_cases);
    for (i = 0; i < test_set->n_cases; ++i) {
        predicted = predict_class(adultnn, test_set->input[i]);
        desired = output_to_class(test_set->output[i]);
        if (predicted != desired)
            ++errors;
        printf("Case %d | predicted: %s, desired: %s, outputs: %4.3f %4.3f \n", i,
               class_to_string(predicted), class_to_string(desired),
               adultnn->output_layer->y[0], adultnn->output_layer->y[1]);
    }

    acc = 100.0 - (100.0 * errors / test_set->n_cases);
    printf("Testing accuracy: %f\n", acc);
    printf("Total classificarion errors: %d\n", errors);

    destroy_dataset(train_set);
    destroy_dataset(test_set);

    return 0;
}
