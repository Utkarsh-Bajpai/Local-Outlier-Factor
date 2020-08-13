//
// Created by fvasluia on 2/28/20.
//
#include "lof_baseline.h"
#include "ComputeTopologyInfo.h"
#include "../unrolled/include/KNN.h"
#include "../avx/include/MMMAvx.h"
#include <stdio.h>

// TODO: 1. move KNN to src / include ?
//       2. Separate measurement from running ?

#ifndef FASTLOF_LOF_H
#define FASTLOF_LOF_H


double* algorithm_driver_baseline_with_individual_measurement(int num_pts, int k, int dim, enum Mode mode,
                                                              FILE* results_file, FILE* exec_file,
                                                              my_dist_fnc dist_fnc, my_kdistall_fnc kdist_fnc,
                                                              my_neigh_fnc neigh_fnc, my_rdall_fnc rdall_fnc,
                                                              my_lrdm_fnc lrdm_fnc, my_lof_fnc lof_fnc);

// ComputePairwiseDistances => KNN => ComputeLocalReachabilityDensityMerged_Pipeline2 => ComputeLocalOutlierFactor_Pipeline2
double algorithm_driver_baseline(int num_pts, int k, int dim,
                                 my_metrics_fnc metrics_fnc,
                                 my_dist_fnc dist_fnc,
                                 my_kdistall_fnc kdist_fnc,
                                 my_neigh_fnc neigh_fnc,
                                 my_lrdm_fnc lrdm_fnc,
                                 my_lof_fnc lof_fnc);

/*
double algorithm_driver_baseline_razavn(int num_pts, int k, int dim,
                                        int B0, int B1,
                                        my_dist_block_fnc dist_block_fnc,
                                        my_kdistall_fnc kdist_fnc,
                                        my_neigh_fnc neigh_fnc,
                                        my_lrdm_fnc lrdm_fnc,
                                        my_lof_fnc lof_fnc);
*/

double algorithm_driver_knn_memory_struct(int num_pts, int k, int dim,
                                          my_metrics_fnc metric_fnc,
                                          my_dist_fnc dist_fnc,
                                          my_knn_fnc knn_fnc,
                                          my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                          my_lrdm2_fnc lrdm2_fnc,
                                          my_lof2_fnc lof2_fnc);

double algorithm_driver_knn_mmm_pairwise_dist(int num_pts, int k, int dim,
                                              int B0, int B1,
                                              my_mmm_dist_fnc mmm_dist_fnc,
                                              my_knn_fnc knn_fnc,
                                              my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                              my_lrdm2_fnc lrdm2_fnc,
                                              my_lof2_fnc lof2_fnc);


// ComputeTopologyInfo => ComputeLocalReachabilityDensityMerged_Pipeline2 => ComputeLocalOutlierFactor_Pipeline2

double algorithm_driver_lattice(int num_pts, int k, int dim,
                                int num_splits, int resolution,
                                my_topolofy_fnc topolofy_fnc,
                                my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                my_lrdm2_fnc lrdm2_fnc,
                                my_lof2_fnc lof2_fnc);

double algorithm_driver_baseline_mmm_pairwise_distance(int num_pts, int k, int dim,
                                                       int B0, int B1,
                                                       my_mmm_dist_fnc mmm_dist_fnc,
                                                       my_kdistall_fnc kdist_fnc,
                                                       my_neigh_fnc neigh_fnc,
                                                       my_lrdm_fnc lrdm_fnc,
                                                       my_lof_fnc lof_fnc);

// -------------------------------- MEASURE SECOND PART OF THE PIPELINE

double algorithm_driver_second_part_original(int num_pts, int k, int dim,
                                             my_lrdm_fnc lrdm_fnc,
                                             my_lof_fnc lof_fnc);

double algorithm_driver_second_part_memory_optimization(int num_pts, int k, int dim,
                                                        my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                                        my_lrdm2_fnc lrdm2_fnc,
                                                        my_lof2_fnc lof2_fnc);

// -------------------------------- MEASURE FIRST PART OF THE PIPELINE

// lattice and mmm are currently unmeasured !!!

double algorithm_driver_first_part_lattice(int num_pts, int k, int dim,
                                           int num_splits, int resolution,
                                           my_topolofy_fnc topolofy_fnc);

double algorithm_driver_first_part_knn_mmm_pairwise_dist(int num_pts, int k, int dim,
                                                         int B0, int B1,
                                                         my_mmm_dist_fnc mmm_dist_fnc,
                                                         my_knn_fnc knn_fnc);

double algorithm_driver_first_part_mmm_pairwise_distance(int num_pts, int k, int dim,
                                                         int B0, int B1,
                                                         my_mmm_dist_fnc mmm_dist_fnc,
                                                         my_kdistall_fnc kdist_fnc,
                                                         my_neigh_fnc neigh_fnc);

double algorithm_driver_first_part_baseline(int num_pts, int k, int dim,
                                            my_metrics_fnc metrics_fnc,
                                            my_dist_fnc dist_fnc,
                                            my_kdistall_fnc kdist_fnc,
                                            my_neigh_fnc neigh_fnc);

#endif //FASTLOF_LOF_H

