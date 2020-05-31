//
// Created by pasca on 19.05.2020.
//

#ifndef FASTLOF_COMPUTEPAIRWISEDISTANCEMMMAVX_H
#define FASTLOF_COMPUTEPAIRWISEDISTANCEMMMAVX_H

int avx_pairwise_distances_driver(int n, int k, int dim, int B0, int B1, int BK, int nr_reps);

double ComputePairwiseDistancesMMMAvx_Fastest(int num_pts, int dim, int B0, int B1, int BK, const double* input_points,
                                              double* pairwise_distances);

double AVXComputePairwiseDistancesWrapper(int n, int dim, int B0, int B1, int BK, const double* input_points,
                                          double* pairwise_distances);

int avx_pairwise_distances_driver(int n, int k, int dim, int B0, int B1, int BK, int nr_reps);

#endif //FASTLOF_COMPUTEPAIRWISEDISTANCEMMMAVX_H
