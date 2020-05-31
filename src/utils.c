//
// Created by fvasluia on 2/28/20.
//

#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.h"
#include "../include/file_utils.h"
#include <string.h>

int compare_double(const void* a, const void* b) {
    // simple comparison function to be used in qsort
//    return (*(double*) a - *(double*) b);
    if (*(double*) a > *(double*) b) return 1;
    else if (*(double*) a < *(double*) b) return -1;
    else return 0;
}


double* XmallocMatrixDouble(int n, int m) {
    double* matrix = (double*)malloc(n * m * sizeof(double));

    if (matrix == NULL) {
        fprintf(stderr, "NULL pointer for matrix allocation");
        exit(-1);
    }

    return matrix;
}

double* XmallocMatrixDoubleRandom(int n, int m) {
    double* matrix = (double *) malloc(n * m * sizeof(double ));
    // posix_memalign(&matrix,  8 * sizeof(void*), n * m * sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "NULL pointer for matrix allocation");
        exit(-1);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
            matrix[i * m + j] = scale * 5;
        }
    }
    return matrix;
}

int* XmallocMatrixInt(int n, int m) {
    int* matrix = (int*) malloc(n * m *  sizeof(int));

    if (matrix == NULL) {
        fprintf(stderr, "NULL pointer for matrix allocation");
        exit(-1);
    }

    return matrix;
}

int* XmallocMatrixIntRandom(int n, int m, int max_el) {
    int* matrix = (int*) malloc(n * m  * sizeof(int));

    if (matrix == NULL) {
        fprintf(stderr, "NULL pointer for matrix allocation");
        exit(-1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
            matrix[i * m + j] = rand() % max_el;
        }
    }

    return matrix;
}

double* XmallocVectorDouble(int n) {
    double* vector = (double*) malloc(n * sizeof(double));
    if (vector == NULL) {
        fprintf(stderr, "NULL pointer from vector allocation");
        exit(-1);
    }
    return vector;
}

double* XmallocVectorDoubleRandom(int n) {
    double* vector = (double*) malloc(n * sizeof(double));
    if (vector == NULL) {
        fprintf(stderr, "NULL pointer from vector allocation");
        exit(-1);
    }


    for (int i = 0; i < n; i++) {
        double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
        vector[i] = scale * 5;
    }
    return vector;
}

double* XmallocVectorDoubleFillRandom(int n, int max_num) {
    /**
     * DID NOT SEE YOUR FUNCTION XmallocVectorDoubleRandom FIRST; WILL DELETE / CHANGE MY LATER
     * 
     * small modification of XmallocVectorDoubleFillRandom that fills the vector with 
     * random numbers, currently all the values are = 0, which might be not 
     * optimal for testing some functions
     */
    double* vector = (double*) calloc(n, sizeof(double));
    if (vector == NULL) {
        fprintf(stderr, "NULL pointer from vector allocation");
        exit(-1);
    }
    for (int i = 0; i < n; ++i) {
        vector[i] = rand() % max_num;
        // printf("%f\n", vector[i]);
    }
    return vector;
}

int GetLinearIndex(int i, int j, int n) {
    int index = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
    return index;
}

void CleanTheCache(int n) {
    double* A = XmallocMatrixDoubleRandom(n, n);
    double* B = XmallocMatrixDoubleRandom(n, n);
    double* C = XmallocMatrixDoubleRandom(n, n);
    double* sum = XmallocVectorDoubleRandom(n);
    double s = 0;
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sum[i] += C[i * n + j];
        }
    }

    for (i = 0; i < n; i++) {
        s += sum[i];
    }
    srandom(s);

}

char* concat(const char* s1, const char* s2) {
    char* result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

