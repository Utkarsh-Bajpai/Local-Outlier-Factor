//
// Created by pasca on 15.05.2020.
//

#ifndef FASTLOF_COMPUTEPAIRWISEDISTANCES_H
#define FASTLOF_COMPUTEPAIRWISEDISTANCES_H

int pairwise_distances_driver(int n, int k, int dim, int B0, int B1, int nr_reps);

double ComputePairwiseDistancesMMM_baseline(int num_pts, int dim, int B0, int B1, int BK, const double* input_points_ptr,
                                          double* distances_indexed_ptr);

double
ComputePairwiseDistancesMMMUnroll_fastest(int num_pts, int dim, int B0, int B1, int BK, const double* input_points_ptr,
                                          double* distances_indexed_ptr);

#endif //FASTLOF_COMPUTEPAIRWISEDISTANCES_H
