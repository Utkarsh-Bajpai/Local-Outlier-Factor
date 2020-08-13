//
// Created by pasca on 08.05.2020.
//

#ifndef FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H
#define FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H

#include "../../include/utils.h"


int lrdensity_merged_driver_unrolled(int num_pts, int k, int dim, int num_reps);


void ComputeLocalReachabilityDensityMergedHashed(int k, int num_pts, const double* distances_indexed_ptr,
                                                 const double* k_distances_indexed_ptr,
                                                 const int* neighborhood_index_table_ptr,
                                                 const int* hashes_index,
                                                 double* lrd_score_table_ptr);

void ComputeLocalReachabilityDensityMerged_6(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr);

double ComputeLocalReachabilityDensityMergedUnroll_Fastest(int k, int num_pts, const double* distances_indexed_ptr,
                                                           const double* k_distances_indexed_ptr,
                                                           const int* neighborhood_index_table_ptr,
                                                           double* lrd_score_table_ptr);

#endif //FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H
