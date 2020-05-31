/**
* This function lists all the functions we are working on with
* corresponding typedefs and functions to compute the number of flops
*
* TODO:
* 1. please check the flop computations
* 2. add an algorithm_driver
*/
#include "metrics.h"

#ifndef FASTLOF_LOF_BASELINE_H
#define FASTLOF_LOF_BASELINE_H

typedef double (* my_metrics_fnc)(const double*, const double*, int);

// Distance Matrix ---------------------------------------------------------------->
typedef double (* my_dist_fnc)(int, int, const double*, my_metrics_fnc, double*);

double ComputeFlopsPairwiseDistances(int dim, int num_pts);

double ComputePairwiseDistances(int dim, int num_pts, const double* input_points_ptr,
                              my_metrics_fnc fct,
                              double* distances_indexed_ptr);

// Definition 3 ------------------------------------------------------------------->
typedef void (* my_kdistobj_fnc)(int, int, int, const double*, double*);

typedef double (* my_kdistall_fnc)(int, int, const double*, double*);

double ComputeFlopsKDistanceAll(int k, int num_pts);  // WHERE IS IT ?

void ComputeKDistanceObject(int obj_idx, int k, int num_pts, const double* distances_indexed_ptr,
                            double* k_distances_indexed_ptr);

double ComputeKDistanceAll(int k, int num_pts, const double* distances_indexed_ptr, double* k_distances_indexed_ptr);

// Definition 4 ------------------------------------------------------------------->

typedef double (* my_neigh_fnc)(int, int, const double*, const double*, int*);

double ComputeKDistanceNeighborhoodAll(int num_pts, int k, const double* k_distances_indexed_ptr,
                                     const double* distances_indexed_ptr, int* neighborhood_index_table_ptr);

// Definition 5 ------------------------------------------------------------------->

typedef void (* my_rdall_fnc)(int, int, const double*, const double*, double*);

void ComputeReachabilityDistanceAll(int k, int num_pts, const double* distances_indexed_ptr,
                                    const double* k_distances_indexed_ptr,
                                    double* reachability_distances_indexed_ptr);

// Definition 5 + 6 ------------------------------------------------------------------->

typedef double (* my_lrdm_fnc)(int, int, const double*, const double*, const int*, double*);

// helps if we consider several ways to cout the number of operations (i.e. with / without indexes)
typedef double (* flops_lrdm_fnc)(int, int);

double ComputeFlopsReachabilityDensityMerged(int k, int num_pts);

double ComputeLocalReachabilityDensityMerged(int k, int num_pts, const double* distances_indexed_ptr,
                                           const double* k_distances_indexed_ptr,
                                           const int* neighborhood_index_table_ptr,
                                           double* lrd_score_table_ptr);

// Definition 6 ----------------------------------------------------------------------->

typedef void (* my_lrd_fnc)(int, int, const double*, const double*, double*);

void ComputeLocalReachabilityDensity(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                     const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr);


// Definition 7 ------------------------------------------------------------------->

typedef double (* my_lof_fnc)(int, int, const double*, const int*, double*);

double ComputeFlopsLocalOutlierFactor(int k, int num_pts);

double ComputeLocalOutlierFactor(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

// PIPELINE 2 --------------------------------------------------------------------->
// Definition 5 + 6

typedef void (* my_lrdm2_pnt_fnc)(int, int, const double*, const double*, const int*, double*);

typedef double (* my_lrdm2_fnc)(int, int, my_lrdm2_pnt_fnc , const double*, const int*, double*, double*, double*);

void ComputeLocalReachabilityDensityMerged_Point(int pnt_idx, int k,
                                                 const double* dist_k_neighborhood_index,
                                                 const double* k_distance_index,
                                                 const int* k_neighborhood_index,
                                                 double* lrd_score_table_ptr);

double ComputeLocalReachabilityDensityMerged_Pipeline2(int k, int num_pts,
                                                     my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                                     const double* dist_k_neighborhood_index,
                                                     const int* k_neighborhood_index,
                                                     double* k_distance_index,
                                                     double* lrd_score_table_ptr,
                                                     double* lrd_score_neigh_table_ptr);

// Definition 7
typedef double (* my_lof2_fnc)(int, int, const double*, const double*, double*);

double ComputeLocalOutlierFactor_Pipeline2(int k, int num_pts,
                                         const double* lrd_score_table_ptr_tmp,
                                         const double* lrd_score_neigh_table_tmp_ptr,
                                         double* lof_score_table_ptr);
// ALGO DRIFER -------------------------------------------------------------------->
// See separate file!


#endif //FASTLOF_LOF_BASELINE_H
