//
// Created by pasca on 26.04.2020.
//
#include "lof_baseline.h"

#ifndef FASTLOF_TESTS_H
#define FASTLOF_TESTS_H

int test_neigh_ind(int num_points, int k, const int* neigh_ind_results, const int* neigh_ind_true);

int test_neigh_dist(int num_points, int k, double tol, double* neigh_dist_results, double* neigh_dist_true);

int test_double_arrays(int num_points, double tol, double* lrd_results, double* lrd_true);

int test_double_arrays_precise(int num_points, double tol, double* lrd_results, double* lrd_true);

void print_matrices(int n, double* mm_true, double* mm_results);

int test_double_matrices(int num_points, double tol, double* lrd_results, double* lrd_true);

void test_pipeline_2( int num_pts, int k, double tol, int isolate_lof );

void test_pipeline_2_ver2( int dim, int num_pts, int k, double tol, my_lrdm2_fnc  lrdm2_fnc);
#endif //FASTLOF_TESTS_H
