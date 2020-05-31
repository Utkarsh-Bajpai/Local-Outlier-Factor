//
// Created by Sycheva  Anastasia on 21.05.20.
//
#include <math.h>

#include "../../include/utils.h"
#include "../../include/lof_baseline.h"
#include "../include/ComputeLocalReachabilityDensityMerged_Point.h"



void ComputeLocalReachabilityDensityMerged_Unroll4( int pnt_idx, int k,
                                                    const double* dist_k_neighborhood_index,
                                                    const double* k_distance_index,
                                                    const int* k_neighborhood_index,
                                                    double* lrd_score_table_ptr ){
    /**
     * Compute lrd score for @param pnt_idx
     * See ComputeLocalReachabilityDensityMerged_Pipeline2 for argument description
     */

    int base_idx, base_idx_j, base_idx_j0, base_idx_j1, base_idx_j2, base_idx_j3;
    int neigh_idx, neigh_idx0, neigh_idx1, neigh_idx2, neigh_idx3;

    double sum = 0;
    double kdistO, kdistO_0, kdistO_1, kdistO_2, kdistO_3;
    double distPo, distPo_0, distPo_1, distPo_2, distPo_3;

    base_idx = pnt_idx * k;

    unsigned int j;

    for (j = 0; j + 3 < k; j += 4) {

        base_idx_j0 = base_idx + j;
        base_idx_j1 = base_idx_j0 + 1;
        base_idx_j2 = base_idx_j1 + 1;
        base_idx_j3 = base_idx_j2 + 1;

        neigh_idx0 = k_neighborhood_index[ base_idx_j0 ];
        neigh_idx1 = k_neighborhood_index[ base_idx_j1 ];
        neigh_idx2 = k_neighborhood_index[ base_idx_j2 ];
        neigh_idx3 = k_neighborhood_index[ base_idx_j3 ];

        kdistO_0 = k_distance_index[neigh_idx0];
        kdistO_1 = k_distance_index[neigh_idx1];
        kdistO_2 = k_distance_index[neigh_idx2];
        kdistO_3 = k_distance_index[neigh_idx3];

        distPo_0 = dist_k_neighborhood_index[ base_idx_j0 ];
        distPo_1 = dist_k_neighborhood_index[ base_idx_j1 ];
        distPo_2 = dist_k_neighborhood_index[ base_idx_j2 ];
        distPo_3 = dist_k_neighborhood_index[ base_idx_j3 ];

        sum += kdistO_0 >= distPo_0 ? kdistO_0 : distPo_0;
        sum += kdistO_1 >= distPo_1 ? kdistO_1 : distPo_1;
        sum += kdistO_2 >= distPo_2 ? kdistO_2 : distPo_2;
        sum += kdistO_3 >= distPo_3 ? kdistO_3 : distPo_3;

    }

    for( ; j < k; j++ ){
        base_idx_j = base_idx + j;

        neigh_idx = k_neighborhood_index[ base_idx_j ];
        kdistO = k_distance_index[neigh_idx];   //  dist_k_neighborhood_index[ neigh_idx*k + k-1]
        distPo = dist_k_neighborhood_index[ base_idx_j ];

        sum += kdistO >= distPo ? kdistO : distPo;

    }

    double result = k / sum;
    lrd_score_table_ptr[pnt_idx] = result; // move division for later ? => does not seem worth it ...

}   // ComputeLocalReachabilityDensityMerged_Point


void ComputeLocalReachabilityDensityMergedPointUnroll_Fastest( int pnt_idx, int k,
                                                    const double* dist_k_neighborhood_index,
                                                    const double* k_distance_index,
                                                    const int* k_neighborhood_index,
                                                    double* lrd_score_table_ptr ){
    /**
     * Compute lrd score for @param pnt_idx
     * See ComputeLocalReachabilityDensityMerged_Pipeline2 for argument description
     */

    int base_idx, base_idx_j, base_idx_j0, base_idx_j1, base_idx_j2, base_idx_j3, base_idx_j4, base_idx_j5, base_idx_j6, base_idx_j7;
    int neigh_idx, neigh_idx0, neigh_idx1, neigh_idx2, neigh_idx3, neigh_idx4, neigh_idx5, neigh_idx6, neigh_idx7;

    double sum = 0;
    double kdistO, kdistO_0, kdistO_1, kdistO_2, kdistO_3, kdistO_4, kdistO_5, kdistO_6, kdistO_7;
    double distPo, distPo_0, distPo_1, distPo_2, distPo_3, distPo_4, distPo_5, distPo_6, distPo_7;

    base_idx = pnt_idx * k;

    unsigned int j;

    for (j = 0; j + 7 < k; j += 8) {

        base_idx_j0 = base_idx + j;
        base_idx_j1 = base_idx_j0 + 1;
        base_idx_j2 = base_idx_j1 + 1;
        base_idx_j3 = base_idx_j2 + 1;
        base_idx_j4 = base_idx_j3 + 1;
        base_idx_j5 = base_idx_j4 + 1;
        base_idx_j6 = base_idx_j5 + 1;
        base_idx_j7 = base_idx_j6 + 1;

        neigh_idx0 = k_neighborhood_index[ base_idx_j0 ];
        neigh_idx1 = k_neighborhood_index[ base_idx_j1 ];
        neigh_idx2 = k_neighborhood_index[ base_idx_j2 ];
        neigh_idx3 = k_neighborhood_index[ base_idx_j3 ];
        neigh_idx4 = k_neighborhood_index[ base_idx_j4 ];
        neigh_idx5 = k_neighborhood_index[ base_idx_j5 ];
        neigh_idx6 = k_neighborhood_index[ base_idx_j6 ];
        neigh_idx7 = k_neighborhood_index[ base_idx_j7 ];

        kdistO_0 = k_distance_index[neigh_idx0];
        kdistO_1 = k_distance_index[neigh_idx1];
        kdistO_2 = k_distance_index[neigh_idx2];
        kdistO_3 = k_distance_index[neigh_idx3];
        kdistO_4 = k_distance_index[neigh_idx4];
        kdistO_5 = k_distance_index[neigh_idx5];
        kdistO_6 = k_distance_index[neigh_idx6];
        kdistO_7 = k_distance_index[neigh_idx7];

        distPo_0 = dist_k_neighborhood_index[ base_idx_j0 ];
        distPo_1 = dist_k_neighborhood_index[ base_idx_j1 ];
        distPo_2 = dist_k_neighborhood_index[ base_idx_j2 ];
        distPo_3 = dist_k_neighborhood_index[ base_idx_j3 ];
        distPo_4 = dist_k_neighborhood_index[ base_idx_j4 ];
        distPo_5 = dist_k_neighborhood_index[ base_idx_j5 ];
        distPo_6 = dist_k_neighborhood_index[ base_idx_j6 ];
        distPo_7 = dist_k_neighborhood_index[ base_idx_j7 ];

        sum += kdistO_0 >= distPo_0 ? kdistO_0 : distPo_0;
        sum += kdistO_1 >= distPo_1 ? kdistO_1 : distPo_1;
        sum += kdistO_2 >= distPo_2 ? kdistO_2 : distPo_2;
        sum += kdistO_3 >= distPo_3 ? kdistO_3 : distPo_3;
        sum += kdistO_4 >= distPo_4 ? kdistO_4 : distPo_4;
        sum += kdistO_5 >= distPo_5 ? kdistO_5 : distPo_5;
        sum += kdistO_6 >= distPo_6 ? kdistO_6 : distPo_6;
        sum += kdistO_7 >= distPo_7 ? kdistO_7 : distPo_7;

    }

    for( ; j < k; j++ ){
        base_idx_j = base_idx + j;

        neigh_idx = k_neighborhood_index[ base_idx_j ];
        kdistO = k_distance_index[neigh_idx];   //  dist_k_neighborhood_index[ neigh_idx*k + k-1]
        distPo = dist_k_neighborhood_index[ base_idx_j ];

        sum += kdistO >= distPo ? kdistO : distPo;

    }

    double result = k / sum;
    lrd_score_table_ptr[pnt_idx] = result; // move division for later ? => does not seem worth it ...

}   // ComputeLocalReachabilityDensityMerged_Point



// ---------------------------------------------------- POSTPROCESSING

//void verify_function( int num_pts, int pnt_idx, int k ){
//
//    // INITIALIZE INPUT
//    double* dist_k_neighborhood_index = XmallocMatrixDoubleRandom( num_pts, k );
//    double* k_distance_index = XmallocVectorDoubleRandom( num_pts );
//    int* k_neighborhood_index = XmallocMatrixIntRandom( num_pts, k, num_pts-1 );
//
//    double* lrd_score_table_ptr_true = XmallocVectorDouble( num_pts );
//    double* lrd_score_table_ptr = XmallocVectorDouble( num_pts );
//
//    ComputeLocalReachabilityDensityMerged_Point(pnt_idx, k, dist_k_neighborhood_index, k_distance_index,
//                                                k_neighborhood_index, lrd_score_table_ptr_true);
//
//    ComputeLocalReachabilityDensityMerged_Unroll8(pnt_idx, k, dist_k_neighborhood_index, k_distance_index,
//                                                k_neighborhood_index, lrd_score_table_ptr);
//
//    if(fabs( lrd_score_table_ptr[pnt_idx] - lrd_score_table_ptr_true[pnt_idx] ) < 0.001){
//        printf("PASSED VERIFICATION");
//    } else {
//        printf("DID NOT PASS VERIFICATION: should %lf is %lf", lrd_score_table_ptr_true[pnt_idx], lrd_score_table_ptr[pnt_idx]);
//    }
//}