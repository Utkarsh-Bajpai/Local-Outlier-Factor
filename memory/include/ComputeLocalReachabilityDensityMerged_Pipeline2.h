//
// Created by Sycheva  Anastasia on 22.05.20.
//

#ifndef FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_PIPELINE2_H
#define FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_PIPELINE2_H

typedef void (* my_lrdm2_fnc) (int, int, const double*, const int*, double*, double*, double*);

void verify_lrdm2_function( int num_pts, int k, my_lrdm2_fnc fun_to_verify );

// -------------------------- EXPERIMENTS ------------------------------------------------

void ComputeLocalReachabilityDensityMerged_Pipeline2_Recursive(int k, int num_pts,
                                                               const double* dist_k_neighborhood_index,
                                                               const int* k_neighborhood_index,
                                                               double* k_distance_index,
                                                               double* lrd_score_table_ptr,
                                                               double* lrd_score_neigh_table_ptr);

#endif //FASTLOF_COMPUTELOCALREACHABILITYDENSITYMERGED_PIPELINE2_H
