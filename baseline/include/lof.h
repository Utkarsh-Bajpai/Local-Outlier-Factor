//
// Created by fvasluia on 2/28/20.
//
#include "../../include/utils.h"

#ifndef FASTLOF_LOF_H
#define FASTLOF_LOF_H


void ComputePairwiseDistances(int dim, int num_pts, const double* input_points_ptr, double (* fct)(const double*,
                                                                                                   const double*, int),
                              double* distances_indexed_ptr);

double ComputeFlopsPairwiseDistances(int dim, int num_pts);


// Definition 3
void ComputeKDistanceObject(int obj_idx, int k, int num_pts, const double* distances_indexed_ptr,
                            double* k_distances_indexed_ptr);

void ComputeKDistanceAll(int k, int num_pts, const double* distances_indexed_ptr, double* k_distances_indexed_ptr);

double ComputeFlopsKDistanceAll(int k, int num_pts);


// Definition 4
void ComputeKDistanceNeighborhoodOjbect(int obj_idx, int k, int num_pts, const double* distances_indexed_ptr,
                                        const double* k_distances_indexed_ptr,
                                        int* neighborhood_index_table_ptr);

void
ComputeKDistanceNeighborhoodAll(int num_pts, int k, const double* k_distances_indexed_ptr,
                                const double* distances_indexed_ptr, int* neighborhood_index_table_ptr);

double ComputeFlopsKNeighborhoodAll(int num_pts, int k);

// Definition 5
void ComputeReachabilityDistanceObject(int obj_from_idx, int obj_to_idx, int k, int num_pts,
                                       const double* distances_indexed_ptr,
                                       const double* k_distances_indexed_ptr,
                                       double* reachability_distances_indexed_ptr);

void ComputeReachabilityDistanceAll(int k, int num_pts, const double* distances_indexed_ptr,
                                    const double* k_distances_indexed_ptr,
                                    double* reachability_distances_indexed_ptr);

double ComputeFlopsReachabilityDistanceAll(int num_pts);

// Definition 6
void ComputeLocalReachabilityDensity(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                     const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr);

double ComputeFlopsLocalReachabilityDensity(int k, int num_pts);


// Definition 7
void ComputeLocalOutlierFactor(int k, int num_pts, const double* lrd_score_table_ptr,
                               const int* neighborhood_index_table_ptr,
                               double* lof_score_table_ptr);

void faster_function_1(int k, int num_pts, const double* lrd_score_table_ptr,
                               const int* neighborhood_index_table_ptr,
                               double* lof_score_table_ptr);

void faster_function_2(int k, int num_pts, const double* lrd_score_table_ptr,
                               const int* neighborhood_index_table_ptr,
                               double* lof_score_table_ptr);

void faster_function_3(int k, int num_pts, const double* lrd_score_table_ptr,
                               const int* neighborhood_index_table_ptr,
                               double* lof_score_table_ptr);

void faster_function_4(int k, int num_pts, const double* lrd_score_table_ptr,
                               const int* neighborhood_index_table_ptr,
                               double* lof_score_table_ptr);

void faster_function_5(int k, int num_pts, const double* lrd_score_table_ptr,
                               const int* neighborhood_index_table_ptr,
                               double* lof_score_table_ptr);


double performance_measure_hot_for_lof (int k, int num_pts, 
                                        double (* fct)(int , int , const double*, const int*, double* ), 
                                        double (* fct_original)(int , int , const double*, const int*, double* ), 
                                        int to_verify, int verobse, int compare_with_baseline);

void performance_plot_lof_to_file( int k, int max_num_pts, int step,
                                   const char* name, const char* mode,
                                   double (* fct)(int , int , const double*, const int*, double* ), 
                                   double (* fct_original)(int , int , const double*, const int*, double* ) );

int exec_lof();

double ComputeFlopsLocalOutlierFactor(int k, int num_pts);

//Compute Local Reachability Density
double* mean_axis_1(double** arr, int n, int m, double* result);

double* array_max(double* arr1, double* arr2, int n, double* result);

double* LofBaseline(int num_pts, int k, int dim, enum Mode mode, FILE* results_file, FILE* exec_file);

#endif //FASTLOF_LOF_H

