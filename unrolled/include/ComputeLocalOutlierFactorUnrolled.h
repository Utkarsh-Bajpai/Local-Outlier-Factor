//
// Created by pasca on 08.05.2020.
//

#ifndef FASTLOF_COMPUTELOCALOUTLIERFACTORUNROLLED_H
#define FASTLOF_COMPUTELOCALOUTLIERFACTOR_H

//BASELINE
/*
typedef void (* my_lof_fnc)(int, int, const double*, const int*, double*);

void ComputeLocalOutlierFactor(int k, int num_pts, const double* lrd_score_table_ptr,
                               const int* neighborhood_index_table_ptr,
                               double* lof_score_table_ptr);
*/
//____________________________ UTILS / POSTPROCESSING __________________________________

int lof_driver_unrolled( int k_ref, int num_pts_ref );

//____________________________ IMPROVEMENTS ____________________________________________

void ComputeLocalOutlierFactor_1_unroll(int k, int num_pts, const double* lrd_score_table_ptr,
                                const int* neighborhood_index_table_ptr,
                                double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_2_unroll(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_3_unroll(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_4_unroll(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

double ComputeLocalOutlierFactorUnroll_Fastest(int k, int num_pts, const double* lrd_score_table_ptr,
                                               const int* neighborhood_index_table_ptr,
                                               double* lof_score_table_ptr);


void ComputeLocalOutlierFactor_6_unroll(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

#endif //FASTLOF_COMPUTELOCALOUTLIERFACTORUNROLLED_H
