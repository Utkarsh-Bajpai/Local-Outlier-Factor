//
// Created by fvasluia on 5/18/20.
//

#ifndef FASTLOF_AVXMETRICS_H
#define FASTLOF_AVXMETRICS_H
# include "immintrin.h"
//double sum_double_avx(__m256d v);
double AVXEuclideanDistance(const double *v1_ptr, const double *v2_ptr, int n_dim);
double AVXCosineSimilarity(const double *v1_ptr, const double *v2_ptr, int dim);
int unrolled_euclid_test(double* points, double* dists, int n, int dim);
int avx_euclid_test(double* points, double* dists, int n, int dim);
void metrics_testbench(int num_pts,int dim, int num_reps);
#endif //FASTLOF_AVXMETRICS_H
