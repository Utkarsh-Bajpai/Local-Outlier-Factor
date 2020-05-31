//
// Created by fvasluia on 5/12/20.
//

#ifndef FASTLOF_SORT_H
#define FASTLOF_SORT_H
#include "../include/ComputeTopologyInfo.h"

void swap(RECORD* a, RECORD* b);
int partition (RECORD* arr, int low, int high);
void quickSort(RECORD* arr, int low, int high);

void swapArrayDbl(double* a, double* b);
void swapArrayInt(int* a, int* b);
int partitionArray (double* arr_dist, int* arr_idx, int low, int high);
void quickSortArray(double* arr_dist,int* arr_idx,  int low, int high);

#endif //FASTLOF_SORT_H
