//
// Created by pasca on 08.05.2020.
//

#ifndef FASTLOF_COMPUTELOCALOUTLIERFACTOR_H
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

void ComputeLocalOutlierFactor_1(int k, int num_pts, const double* lrd_score_table_ptr,
                                const int* neighborhood_index_table_ptr,
                                double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_2(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_3(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_4(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

double ComputeLocalOutlierFactorUnroll_fastest(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_6(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr);

#endif //FASTLOF_COMPUTELOCALOUTLIERFACTOR_H
