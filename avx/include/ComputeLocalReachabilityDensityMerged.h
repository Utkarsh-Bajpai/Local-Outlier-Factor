//
// Created by pasca on 08.05.2020.
//

#ifndef FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H
#define FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H

#include "../../include/utils.h"
#include "../../include/lof_baseline.h"

//____________________________ UTILS / POSTPROCESSING __________________________________

void fast_test_improvement_lrdm_avx( int k , int num_pts, int num_reps);
int lrdm_driver_avx( int k_ref, int num_pts_ref );
void check_intrinsics();
//____________________________ IMPROVEMENTS ____________________________________________

void ComputeLocalReachabilityDensityMerged_OUTER4_INNER1_AVX_128(int k, int num_pts, const double* distances_indexed_ptr,
                                                                 const double* k_distances_indexed_ptr,
                                                                 const int* neighborhood_index_table_ptr,
                                                                 double* lrd_score_table_ptr);

void ComputeLocalReachabilityDensityMerged_OUTER1_INNER4_AVX_128(int k, int num_pts, const double* distances_indexed_ptr,
                                                                 const double* k_distances_indexed_ptr,
                                                                 const int* neighborhood_index_table_ptr,
                                                                 double* lrd_score_table_ptr);

double ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128(int k, int num_pts, const double* distances_indexed_ptr,
                                                                 const double* k_distances_indexed_ptr,
                                                                 const int* neighborhood_index_table_ptr,
                                                                 double* lrd_score_table_ptr);

void ComputeLocalReachabilityDensityMerged_OUTER4_INNER8_AVX_128(int k, int num_pts, const double* distances_indexed_ptr,
                                                                 const double* k_distances_indexed_ptr,
                                                                 const int* neighborhood_index_table_ptr,
                                                                 double* lrd_score_table_ptr);

#endif //FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H
