//
// Created by fvasluia on 5/20/20.
//

#ifndef FASTLOF_AVXTOPOINFO_H
#define FASTLOF_AVXTOPOINFO_H
# include "../../include/ComputeTopologyInfo.h"

long AVXTopologyInfo(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                     double* neighborhood_distance_table_ptr, int k, int num_points, int split_dim, int dim);

long AVXDistanceComputeTopologyInfo(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                                    double* k_dist_table_ptr, int k, int num_points, int split_dim,  int dim);

void test_avx_topo_info(int num_pts, int k);
void avx_topology_info_driver(int num_pts, int k, int dim, int num_reps);

#endif //FASTLOF_AVXTOPOINFO_H
