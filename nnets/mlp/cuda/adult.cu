/*

  adult.cu
  Classification of the adult dataset from the UCI Machine Learning Repository
  implemented in CUDA. 
  http://archive.ics.uci.edu/ml/datasets/Adult

  Andrei de A. Formiga, 2012-05-23

 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlpnnets.h"

#define SEED                  631814ULL

#define EPOCHS                7000
#define LEARNING_RATE         0.003

#define MAX_ABS               1.2f

#define MAX(a, b)             (a >= b? a : b)

float string_to_float_workclass(char *c);
float string_to_float_education(char *c);
float string_to_float_marital_status(char *c);
float string_to_float_relationship(char *c);
float string_to_float_occupation(char *c);
float string_to_float_race(char *c);
float string_to_float_sex(char *c);
float string_to_float_native_country(char *c);

typedef enum tagClass {
    GT50k,
    LTE50k
} Class;


DataSet* read_dataset(char *filename)
{
    FILE    *f;
    int     i, j;
    float   fnlwgt, education_num, capital_gain, capital_loss;
    float   hours_per_week, age;
    char    workclass[40],  education[40],  marital_status[40];
    char    occupation[40], relationship[40], race[40], sex[40];
    char    native_country[40];
    char    buffer[240];
    DataSet *dset;
                  
    f = fopen(filename, "r");
    if (f == NULL) {
        fprintf(stderr, "File not found: %s\n", filename);        
        return NULL;
    }
       
    // count lines in file to allocate dataset arrays
    i = 0;
    while (fgets(buffer, 240, f) != NULL)
        ++i;

    if (!feof(f) || ferror(f)) {
        fprintf(stderr, "IO error while reading from file\n");
        fclose(f);
        return NULL;
    }
    fseek(f, 0, SEEK_SET);
                
    // prepare dataset
    dset = CreateDataSet(i, 14, 2);
    if (dset == NULL) {
        fprintf(stderr, "Error creating dataset\n");
        return NULL;
    }
            
    i = 0;
    int iix = 0, oix = 0;
    bool done = false;
    while (!done) {
        j = fscanf(f, "%f, %s %f, %s %f, %s %s %s %s %s %f, %f, %f, %s %s\n",
                   &age, &workclass, &fnlwgt, education, &education_num,
                   marital_status, occupation, relationship, &race, &sex,
                   &capital_gain, &capital_loss, &hours_per_week,
                   native_country, buffer);
        /*printf("%3.2f; %s; %3.2f; %s; %3.2f; %s; %s; %s; %s; %s; %3.2f; %3.2f; %3.2f; %s; %s\n", age, workclass, fnlwgt, education, education_num, 
                     marital_status, occupation, relationship, race, sex, capital_gain, 
                     capital_loss, hours_per_week, native_country, buffer);*/

        if (j != 15)
            done = true;
        else {
            dset->inputs[iix++] = age;
            dset->inputs[iix++] = string_to_float_workclass(workclass);
            dset->inputs[iix++] = fnlwgt;
            dset->inputs[iix++] = string_to_float_education(education);
            dset->inputs[iix++] = education_num;            
            dset->inputs[iix++] = string_to_float_marital_status(marital_status);            
            dset->inputs[iix++] = string_to_float_occupation(occupation);            
            dset->inputs[iix++] = string_to_float_relationship(relationship);            
            dset->inputs[iix++] = string_to_float_race(race);            
            dset->inputs[iix++] = string_to_float_sex(sex);            
            dset->inputs[iix++] = capital_gain;                        
            dset->inputs[iix++] = capital_loss;            
            dset->inputs[iix++] = hours_per_week;            
            dset->inputs[iix++] = string_to_float_native_country(native_country);                                    

            if (strstr(buffer, "<=50K")) {
                dset->outputs[oix++] = 0.9f;
                dset->outputs[oix++] = 0.1f;
            } else { // assumes >50k
                dset->outputs[oix++] = 0.1f;
                dset->outputs[oix++] = 0.9f;
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

#define FEMALE  1.0f
#define MALE    2.0f
#define UNKNOWN 0.0f

float string_to_float_sex(char *c)
{
    float res;

    if(strstr(c,"Female,"))
        res = FEMALE;
    else if(strstr(c, "Male,"))
        res = MALE;
    else res = UNKNOWN;
        
    return res;
}

#define WHITE              1.0f
#define ASIAN_PAC_ISLANDER 2.0f
#define AMER_INDIAN_ESKIMO 3.0f
#define OTHER              4.0f
#define BLACK              5.0f

float string_to_float_race(char *c)
{
    float res;

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
    else res = UNKNOWN;
        
    return res;
}

#define TECH_SUPPORT      1.0f
#define CRAFT_REPAIR      2.0f
#define OTHER_SERVICE     3.0f
#define SALES             4.0f
#define EXEC_MANAGERIAL   5.0f
#define PROF_SPECIALTY    6.0f
#define HANDLERS_CLEANERS 7.0f
#define MACHINE_OP_INSPCT 8.0f
#define ADM_CLERICAL      9.0f
#define FARMING_FISHING   10.0f
#define TRANSPORT_MOVING  11.0f
#define PRIV_HOUSE_SERV   12.0f
#define PROTECTIVE_SERV   13.0f
#define ARMED_FORCES      14.0f


float string_to_float_occupation(char *c)
{
    float res;

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
    else res = UNKNOWN;
        
    return res;
}

#define WIFE           1.0f
#define OWN_CHILD      2.0f
#define HUSBAND        3.0f
#define NOT_IN_FAMILY  4.0f
#define OTHER_RELATIVE 5.0f
#define UNMARRIED      6.0f

float string_to_float_relationship(char *c)
{
    float res;

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
    else res = UNKNOWN;
        
    return res;
}

#define MARRIED_CIV_SPOUSE    1.0f
#define DIVORCED              2.0f
#define NEVER_MARRIED         3.0f
#define SEPARATED             4.0f
#define WIDOWED               5.0f
#define MARRIED_SPOUSE_ABSENT 6.0f
#define MARRIED_AF_SPOUSE     7.0f

float string_to_float_marital_status(char *c)
{
    float res;

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
    else res = UNKNOWN;
        
    return res;
}

#define BACHELORS    1.0f
#define SOME_COLLEGE 2.0f
#define N11TH        3.0f
#define HS_GRAD      4.0f
#define PROF_SCHOOL  5.0f
#define ASSOC_ACDM   6.0f
#define ASSOC_VOC    7.0f
#define N9TH         8.0f
#define N7TH_8TH     9.0f
#define N12TH        10.0f
#define MASTERS      11.0f
#define N1ST_4TH     12.0f
#define N10TH        13.0f
#define DOCTORATE    14.0f
#define N5TH_6TH     15.0f
#define PRESCHOOL    16.0f


float string_to_float_education(char *c)
{
    float res;

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
    else res = UNKNOWN;
    return res;
}

#define PRIVATE          1.0f
#define SELF_EMP_NOT_INC 2.0f
#define SELF_EMP_INC     3.0f
#define FEDERAL_GOV      4.0f
#define LOCAL_GOV        5.0f
#define STATE_GOV        6.0f
#define WITHOUT_PAY      7.0f
#define NEVER_WORKED     8.0f

float string_to_float_workclass(char *c)
{
    float res;

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
    else res = UNKNOWN;
        
    return res;
}

#define UNITED_STATES      1.0f
#define CAMBODIA           2.0f
#define ENGLAND            3.0f
#define PUERTO_RICO        4.0f
#define CANADA             5.0f
#define GERMANY            6.0f
#define OUTLYING_US        7.0f
#define INDIA              8.0f
#define JAPAN              9.0f
#define GREECE             10.0f
#define SOUTH              11.0f
#define CHINA              12.0f
#define CUBA               13.0f
#define IRAN               14.0f
#define HONDURAS           15.0f
#define PHILIPPINES        16.0f
#define ITALY              17.0f
#define POLAND             18.0f
#define JAMAICA            19.0f
#define VIETNAM            20.0f
#define MEXICO             21.0f
#define PORTUGAL           22.0f
#define IRELAND            23.0f
#define FRANCE             24.0f
#define DOMINICAN_REPUBLIC 25.0f
#define LAOS               26.0f
#define ECUADOR            27.0f
#define TAIWAN             28.0f
#define HAITI              29.0f
#define COLUMBIA           30.0f
#define HUNGARY            31.0f
#define GUATEMALA          32.0f
#define NICARAGUA          33.0f
#define SCOTLAND           34.0f
#define THAILAND           35.0f
#define YUGOSLAVIA         36.0f
#define EL_SALVADOR        37.0f
#define TRINADAD_TOBAGO    38.0f
#define PERU               39.0f
#define HONG               40.0f
#define HOLAND_NETHERLANDS 41.0f

float string_to_float_native_country(char *c)
{
    float res;

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
        res = OUTLYING_US;
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
    else res = UNKNOWN;
        
    return res;
}


Class output_to_class(float *output)
{
    float max;

    max = MAX(output[0], output[1]);
    if (output[0] == max)
        return GT50k;
    return LTE50k;
}

Class predict_class(MLPNetwork *nnet, float *input)
{
    //forward_prop(nnet, sigmoid, input);
    //return output_to_class(nnet->output_layer->y);
    return GT50k;
}


char    *gt50k = ">50k";
char    *lte50k = "<=50k";

char *class_to_string(Class c)
{
    char *res;

    switch(c) {
    case LTE50k:
        res = lte50k;
        break;
    default:
        res = gt50k;
    }

    return res;
}

int neuronsPerLayer[] = { 14, 28, 2 };

int main(int argc, char **argv)
{
    int     i;
    int     errors;
    DataSet *train_set;
    DataSet *test_set;
    float   e;
    double  acc;
    Class   predicted, desired;

    MLPNetwork *adultnn;

    // training
    train_set = read_dataset("adult.train");

    if (train_set == NULL) {
        fprintf(stderr, "Error reading training set\n");
        exit(1);
    }

    adultnn = CreateNetwork(3, neuronsPerLayer);
    RandomWeights(adultnn, MAX_ABS, SEED);

    printf("Training network with %d epochs...\n", EPOCHS);
    e = BatchTrainBackprop(adultnn, train_set, EPOCHS, LEARNING_RATE, 1, 0);
    printf("Training finished, approximate final SSE: %f\n", e);

    printf("Weights after training:\n");
    PrintWeights(adultnn);

    printf("-----------------------------------------\n");

    // free the training dataset
    cudaThreadSynchronize();
    DestroyDataSet(train_set);

    return 0;
}
