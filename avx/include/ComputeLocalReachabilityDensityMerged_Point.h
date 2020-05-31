//
// Created by Sycheva  Anastasia on 21.05.20.
//

#include "../../include/lof_baseline.h"

#ifndef FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_POINT_H
#define FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_POINT_H


typedef void ( * my_fun )(int, int, const double*, const double*, const int*, double*);

void verify_function( int num_pts, int pnt_idx, int k, my_fun fun_to_verify );

void lrdm_point_perf_driver(int num_pts, int k, int num_reps);
// ------------------------------------------------- AVX improvements

void ComputeLocalReachabilityDensityMerged_Point_AVX4( int pnt_idx, int k,
                                                 const double* dist_k_neighborhood_index,
                                                 const double* k_distance_index,
                                                 const int* k_neighborhood_index,
                                                 double* lrd_score_table_ptr );

void ComputeLocalReachabilityDensityMerged_Point_AVX4_V2( int pnt_idx, int k,
                                                          const double* dist_k_neighborhood_index,
                                                          const double* k_distance_index,
                                                          const int* k_neighborhood_index,
                                                          double* lrd_score_table_ptr );

void ComputeLocalReachabilityDensityMerged_Point_AVX8( int pnt_idx, int k,
                                                       const double* dist_k_neighborhood_index,
                                                       const double* k_distance_index,
                                                       const int* k_neighborhood_index,
                                                       double* lrd_score_table_ptr );

void ComputeLocalReachabilityDensityMerged_Point_AVX16( int pnt_idx, int k,
                                                         const double* dist_k_neighborhood_index,
                                                         const double* k_distance_index,
                                                         const int* k_neighborhood_index,
                                                         double* lrd_score_table_ptr );

void ComputeLocalReachabilityDensityMerged_Point_AVX32_FASTEST( int pnt_idx, int k,
                                                        const double* dist_k_neighborhood_index,
                                                        const double* k_distance_index,
                                                        const int* k_neighborhood_index,
                                                        double* lrd_score_table_ptr );

#endif //FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_POINT_H
