//
// Created by Sycheva  Anastasia on 21.05.20.
//

#ifndef FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_POINT_H
#define FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_POINT_H

void verify_function( int num_pts, int pnt_idx, int k );

//-------------------------------------------------- IMPROVEMENTS

void ComputeLocalReachabilityDensityMerged_Unroll4( int pnt_idx, int k,
                                                    const double* dist_k_neighborhood_index,
                                                    const double* k_distance_index,
                                                    const int* k_neighborhood_index,
                                                    double* lrd_score_table_ptr );

void ComputeLocalReachabilityDensityMerged_Unroll8( int pnt_idx, int k,
                                                    const double* dist_k_neighborhood_index,
                                                    const double* k_distance_index,
                                                    const int* k_neighborhood_index,
                                                    double* lrd_score_table_ptr );

void ComputeLocalReachabilityDensityMergedPointUnroll_Fastest(int k, int num_pts, const double* distances_indexed_ptr,
                                                           const double* k_distances_indexed_ptr,
                                                           const int* neighborhood_index_table_ptr,
                                                           double* lrd_score_table_ptr);
#endif //FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_POINT_H
