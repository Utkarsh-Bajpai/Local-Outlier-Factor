//
// Created by fvasluia on 5/12/20.
//
# include "../include/sort.h"

void swap(RECORD* a, RECORD* b)
{
    RECORD t = *a;
    *a = *b;
    *b = t;
}
int partition (RECORD* arr, int low, int high)
{
    RECORD pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high- 1; j++)
    {
        if (arr[j].distance < pivot.distance)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(RECORD* arr, int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void swapArrayDbl(double* a, double* b) {
    double t = *a;
    *a = *b;
    *b = t;
}
void swapArrayInt(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}
int partitionArray(double* arr_dist, int* arr_idx, int low, int high) {
    double pivot = arr_dist[high];
    int i = (low - 1);

    for (int j = low; j <= high- 1; j++)
    {
        if (arr_dist[j] < pivot)
        {
            i++;
            swapArrayDbl(&arr_dist[i], &arr_dist[j]);
            swapArrayInt(&arr_idx[i], &arr_idx[j]);
        }
    }
    swapArrayDbl(&arr_dist[i + 1], &arr_dist[high]);
    swapArrayInt(&arr_idx[i + 1], &arr_idx[high]);
    return (i + 1);

}
void quickSortArray(double* arr_dist,int* arr_idx,  int low, int high) {
    if (low < high)
    {
        int pi = partitionArray(arr_dist, arr_idx, low, high);

        quickSortArray(arr_dist, arr_idx,  low, pi - 1);
        quickSortArray(arr_dist, arr_idx, pi + 1, high);
    }
}