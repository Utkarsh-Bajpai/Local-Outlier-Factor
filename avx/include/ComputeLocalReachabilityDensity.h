//
// Created by pasca on 05.05.2020.
//


#ifndef FASTLOF_LRD_H
#define FASTLOF_LRD_H

#include "../../include/utils.h"


// Definition 6
void ComputeLocalReachabilityDensity(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                     const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr);

// int lrdensity_driver(int num_pts, int k, int num_reps);


#endif //FASTLOF_LRD_H
