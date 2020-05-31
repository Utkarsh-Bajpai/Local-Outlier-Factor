//
// Created by fvasluia on 5/8/20.
//

#ifndef FASTLOF_COMPUTETOPOLOGYINFO_H
#define FASTLOF_COMPUTETOPOLOGYINFO_H

# include "lattice.h"

typedef struct record_ {
    int point_idx;
    double distance;
} RECORD;

typedef long (* my_topolofy_fnc)(MULTI_LATTICE*, double*, int*, double*, int, int, int, int);

int compare_record(const void * a, const void * b);

long ComputeTopologyInfo(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                         double* k_dist_table_ptr, int k, int num_points, int split_dim,  int dim);

// use permutations to produce neighbors
long ComputeTopologyInfoWithPermutations(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                                         double* k_dist_table_ptr, int k, int num_points, int split_dim,  int dim);

/// !!! with the regular distance metric
long BASEComputeTopologyInfo(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                             double* k_dist_table_ptr, int k, int num_points, int split_dim,  int dim);

int topology_info_driver(int num_pts, int k, int num_reps);

#endif //FASTLOF_COMPUTETOPOLOGYINFO_H
