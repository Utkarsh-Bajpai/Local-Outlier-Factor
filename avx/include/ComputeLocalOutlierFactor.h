//
// Created by pasca on 08.05.2020.
//

#ifndef FASTLOF_COMPUTELOCALOUTLIERFACTOR_H
#define FASTLOF_COMPUTELOCALOUTLIERFACTOR_H

// POSTPROCESSING

int  lof_driver_avx( int k_ref, int num_pts_ref );

//--------------------------- < AVX Improvements > ------------------------------------------
double ComputeLocalOutlierFactor_1(int k, int num_pts, const double* lrd_score_table_ptr,
                                     const int* neighborhood_index_table_ptr,
                                     double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_Unrolled(int k, int num_pts, const double* lrd_score_table_ptr,
                                        const int* neighborhood_index_table_ptr,
                                        double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_AVX_Old(int k, int num_pts, const double* lrd_score_table_ptr,
                                       const int* neighborhood_index_table_ptr,
                                       double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_Avx(int k, int num_pts, const double* lrd_score_table_ptr,
                                     const int* neighborhood_index_table_ptr,
                                     double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_2(int k, int num_pts,
                                 const double* lrd_score_table_ptr,
                                 const double* lrd_score_neigh_table_ptr,
                                 double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_2_unroll_AVX(int k, int num_pts,
                                        const double* lrd_score_table_ptr,
                                        const double* lrd_score_neigh_table_ptr,
                                        double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_2_unroll_AVX_fixed(int k, int num_pts,
                                                  const double* lrd_score_table_ptr,
                                                  const double* lrd_score_neigh_table_ptr,
                                                  double* lof_score_table_ptr);

void ComputeLocalOutlierFactor_2_AVX_Faster(int k, int num_pts,
                                            const double* lrd_score_table_ptr,
                                            const double* lrd_score_neigh_table_ptr,
                                            double* lof_score_table_ptr);

double ComputeLocalOutlierFactor_2_AVX_Fastest(int k, int num_pts,
                                             const double* lrd_score_table_ptr,
                                             const double* lrd_score_neigh_table_ptr,
                                             double* lof_score_table_ptr);

void lof_driver( int num_pts, int pnt_idx, int k);

void lof_verify_an( int num_pts, int pnt_idx, int k);


//--------------------------- < /AVX Improvements > ------------------------------------------


#endif //FASTLOF_COMPUTELOCALOUTLIERFACTOR_H
