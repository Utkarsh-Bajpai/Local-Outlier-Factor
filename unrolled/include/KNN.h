//
// Created by fvasluia on 5/11/20.
//

#ifndef FASTLOF_KNN_H
#define FASTLOF_KNN_H

typedef double (* my_knn_fnc)(int, int, int*, const double*, double*);

double KNN(int num_pts, int k, int* k_neighborhood_index, const double* pairwise_dist, double* k_dist_index);

double KNN_fastest(int n, int k, int* k_neighborhood_index, const double* pairwise_dist, double* k_dist_index);

void KNN4x4(int n, int k, int* k_neighborhood_index, const double* pairwise_dist, double* k_dist_index);

void knn_driver(int num_pts, int k, int dim, int num_reps);

double test_knn_baseline(int num_pts, int k, int dim, double* dist_matrix, int* k_index_ptr, int* hash_index,
                         double* k_dist_ptr,
                         double* points_matrix);

void KNN_4(int n, int k, int dim, double* points_matrix, int* k_neighborhood_index, double* dist_array,
           double* k_dist_index);

void
KNN_hash(int n, int k, int dim, double* points_matrix, int* k_neighborhood_index, int* hash_index, double* dist_array,
         double* k_dist_index);

#endif //FASTLOF_KNN_H
