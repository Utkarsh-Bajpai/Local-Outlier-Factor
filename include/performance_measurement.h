//
// Created by Sycheva  Anastasia on 16.05.20.
//

#include "lof_baseline.h"

#ifndef FASTLOF_PERFORMANCE_MEASUREMENT_H
#define FASTLOF_PERFORMANCE_MEASUREMENT_H

#define CYCLES_REQUIRED 1e8
// Metrics ------------------------------------------------------------------>

double performance_measure_hot_for_metrics(int dim, double (* fct)(const double*, const double*, int),
                                           double (* fct_correct)(const double*, const double*, int),
                                           int to_verify, int verobse, int compare_with_baseline);

// Compute Pairwise Distance ------------------------------------------------------------->

double performance_measure_fast_for_mini_mmm(int Bp, int Bd, int num_reps, int num_pts, int dim,
                                             void (*fnc)(int, int, int, int, const double*, double*));

// Definition 6 -------------------------------------------------------------------------->

double performance_measure_fast_for_lrd(int k, int num_pts, int num_reps, my_lrdm_fnc lrdf_fct, flops_lrdm_fnc flops_fct);

double performance_measure_hot_for_lrd(int k, int num_pts,
                                       my_lrdm_fnc fct, int to_verify, int verobse, int compare_with_baseline);

void performance_plot_lrdm_to_file_k(int num_pts_ref, const char* file_name,
                                     const char* function_name, const char* mode,
                                     my_lrdm_fnc fct);

void performance_plot_lrdm_to_file_num_pts(int k_ref, const char* file_name,
                                           const char* function_name, const char* mode,
                                           my_lrdm_fnc fct );

// Definition 7 ------------------------------------------------------------->

double performance_measure_hot_for_lof(int k, int num_pts,
                                       my_lof_fnc fct, my_lof_fnc fct_original,
                                       int to_verify, int verobse, int compare_with_baseline);

void performance_plot_lof_to_file(int k, int max_num_pts, int step,
                                  const char* name, const char* mode,
                                  my_lof_fnc fct, my_lof_fnc fct_original);

void performance_plot_lof_to_file_num_pts(int k_ref, const char* file_name,
                                          const char* function_name, const char* mode,
                                          my_lof_fnc fct, my_lof_fnc fct_original);

void performance_plot_lof_to_file_k(int num_pts_ref, const char* file_name,
                                    const char* function_name, const char* mode,
                                    my_lof_fnc fct, my_lof_fnc fct_original);

// *********************************************************************************
// PIPELINE 2
// *********************************************************************************

double performance_measure_fast_for_lrdm_point_pipeline2(int pnt_idx, int num_pts, int k, int num_reps,
                                                         my_lrdm2_pnt_fnc lrdm2_point_fnc);

double performance_measure_fast_for_lrdm_pipeline2(int num_pts, int k, int num_reps,
                                                   my_lrdm2_fnc lrdm2_fnc);

#endif //FASTLOF_PERFORMANCE_MEASUREMENT_H
