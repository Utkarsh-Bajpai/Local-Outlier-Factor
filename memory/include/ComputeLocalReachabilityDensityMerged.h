//
// Created by pasca on 08.05.2020.
//

#ifndef FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H
#define FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H

#include "../../include/utils.h"
#include "../../include/lof_baseline.h"

//____________________________ UTILS / POSTPROCESSING __________________________________


int lrdm_driver_memory( int k_ref, int num_pts_ref );


//____________________________ IMPROVEMENTS ____________________________________________

void ComputeLocalReachabilityDensityMerged_Reversed(int k, int num_pts, const double* distances_indexed_ptr,
                                                    const double* k_distances_indexed_ptr,
                                                    const int* neighborhood_index_table_ptr,
                                                    double* lrd_score_table_ptr);

void ComputeLocalReachabilityDensityMerged_Reversed_Layout(int k, int num_pts, const double* distances_indexed_ptr,
                                                           const double* k_distances_indexed_ptr,
                                                           const int* neighborhood_index_table_transposed_ptr,
                                                           double* lrd_score_table_ptr);

#endif //FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_H
