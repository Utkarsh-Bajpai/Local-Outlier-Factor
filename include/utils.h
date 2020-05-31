//
// Created by fvasluia on 2/28/20.
//
#include <stdio.h>

#ifndef FASTLOF_UTILS_H
#define FASTLOF_UTILS_H

enum Mode {
    BASIC = 0, TESTING = 1, BENCHMARK = 2
};

double* XmallocSymmetricVectorDouble(int n);

double* XmallocMatrixDouble(int n, int m);

double* XmallocMatrixDoubleRandom(int n, int m);

double* XmallocVectorDouble(int n);

double* XmallocVectorDoubleRandom(int n);

double* XmallocVectorDoubleFillRandom(int n, int max_num);

int* XmallocMatrixInt(int n, int m);

int* XmallocMatrixIntRandom(int n, int m, int max_val);

int compare_double(const void* a, const void* b);

char* concat(const char* s1, const char* s2);

int GetLinearIndex(int i, int j, int n);

void CleanTheCache(int n);

#endif //FASTLOF_UTILS_H
