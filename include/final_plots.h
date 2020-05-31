//
// Created by Sycheva  Anastasia on 23.05.20.
//

#include "lof_baseline.h"

#ifndef FASTLOF_FINAL_PLOTS_H
#define FASTLOF_FINAL_PLOTS_H

// DEFINE CONSTANTS (See discussion on the slides) -------------------------------------------->




// ---------------------------------------------------------------------------------------------->

void performance_plot_algo_num_pts(int k_ref,  int dim_ref,
                                   int* num_pts_grid,
                                   const char* file_name,
                                   const char* function_name, const char* mode,
                                   int B0, int B1, // for blocked MMM
                                   int num_splits, int resolution, // for lattice

                                   // FUNCTIONS FOR baseline
                                   my_metrics_fnc metrics_fnc,
                                   my_dist_fnc dist_fnc,
                                   my_kdistall_fnc kdist_fnc,
                                   my_neigh_fnc neigh_fnc,
                                   my_lrdm_fnc lrdm_fnc,
                                   my_lof_fnc lof_fnc,

                                   // FUNCTIONS FOR knn_memory_struct (additional)
                                   my_knn_fnc knn_fnc,
                                   my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                   my_lrdm2_fnc lrdm2_fnc,
                                   my_lof2_fnc lof2_fnc,

                                   // FUNCTIONS FOR knn_blocked_mmm
                                   my_dist_block_fnc dist_block_fnc,

                                   // FUNCTIONS FOR lattice
                                   my_topolofy_fnc topolofy_fnc );

void driver_runtime_performance_plot_algo_num_pts(int k_ref, int dim_ref,
                                                  int* num_pts_grid,
                                                  int num_rep_measurements,
                                                  const char* file_name,
                                                  const char* function_name, const char* mode,
                                                  int B0, int B1, // for blocked MMM
                                                  int num_splits, int resolution, // for lattice

        // FUNCTIONS FOR baseline
                                                  my_metrics_fnc metrics_fnc,
                                                  my_dist_fnc dist_fnc,
                                                  my_kdistall_fnc kdist_fnc,
                                                  my_neigh_fnc neigh_fnc,
                                                  my_lrdm_fnc lrdm_fnc,
                                                  my_lof_fnc lof_fnc,

        // FUNCTIONS FOR knn_memory_struct (additional)
                                                  my_knn_fnc knn_fnc,
                                                  my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                                  my_lrdm2_fnc lrdm2_fnc,
                                                  my_lof2_fnc lof2_fnc,

        // FUNCTIONS FOR knn_blocked_mmm
                                                  my_dist_block_fnc dist_block_fnc,

        // FUNCTIONS FOR lattice
                                                  my_topolofy_fnc topolofy_fnc);

#endif //FASTLOF_FINAL_PLOTS_H
