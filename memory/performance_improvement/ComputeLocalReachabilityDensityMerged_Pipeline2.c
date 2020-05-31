/**
 * Try to improve the memory locality by changing the access order of the elements
 * GOAL: process neighbors one after the other
 * HOPE: reduce the number of cache misses (unless there are conflict misses)
 *
 * HYPOTHESIS 1: use a stack like DFS ? => see https://www.geeksforgeeks.org/stack-data-structure-introduction-program/
 *               already significantly increases memory footproint, might be not that efficient
 *
 * Stack: modification -> decrease memory footprint by decreasing the size of the stack
*/
#include <stdlib.h>å

#include "../../include/lof_baseline.h"
#include "../../include/utils.h"
#include "../../include/tests.h"

#include "../include/ComputeLocalReachabilityDensityMerged_Pipeline2.h"

//---------------------------------------------------------------------------------------->

void process_point_recursive(int pnt_idx, int k, int num_pts,
                             const double* dist_k_neighborhood_index,
                             double* k_distance_index,
                             const int* k_neighborhood_index, double* lrd_score_table_ptr){
    /**
     * @param pnt_idx: index of the point currently being processed
     */
    ComputeLocalReachabilityDensityMerged_Point(pnt_idx, k,
                                                dist_k_neighborhood_index, k_distance_index,
                                                k_neighborhood_index, lrd_score_table_ptr);

    int idx_base = pnt_idx*k;
    for(int j=0; j<k; ++j){

        int idx_j_neighbor = k_neighborhood_index[idx_base + j];
        if( lrd_score_table_ptr[ idx_j_neighbor ] < 0.0 ){
            // ADD TO STACK
            process_point_recursive(idx_j_neighbor, k, num_pts,
                                    dist_k_neighborhood_index, k_distance_index,
                                    k_neighborhood_index, lrd_score_table_ptr);
        } // if
    }  // for(int j

}   // process_point_recursive

void ComputeLocalReachabilityDensityMerged_Pipeline2_Recursive(int k, int num_pts,
                                                               const double* dist_k_neighborhood_index,
                                                               const int* k_neighborhood_index,
                                                               double* k_distance_index,
                                                               double* lrd_score_table_ptr,
                                                               double* lrd_score_neigh_table_ptr) {
    /**
     * Try to make sure that neighborhoods are processed one after the other,
     * maybe this can decrease the number of cache misses ?
     *
     * BAD, use recursion to precompute order ? i.e. stack ...
     */
    int idx_base;

    for(int i=0; i < num_pts; i++){
        double max = -1.0;     // K distance is the max distance among k nearest neighbors
        for( int j=0; j < k; j++ ){
            if( dist_k_neighborhood_index[ i*k + j ] > max ){
                // LAST ELEMENT !
                max = dist_k_neighborhood_index[ i*k + j ];
            }
        } // for j
        k_distance_index[ i ] = max;
    } // for i

    for( int i=0; i < num_pts; i++ ){   // iterate over all points

        process_point_recursive(i , k, num_pts, dist_k_neighborhood_index,
                                k_distance_index, k_neighborhood_index,
                                lrd_score_table_ptr);
        /*
        idx_base = i*k;
        if( lrd_score_table_ptr[ i ] < 0.0 ){
            push(stack, i);
        }

        for( int j=0; j < k; j++ ){
            int idx_j_neighbor = k_neighborhood_index[idx_base + j];
            if( lrd_score_table_ptr[ idx_j_neighbor ] < 0.0 ){
                // ADD TO STACK
                push(stack, idx_j_neighbor);
            } // if
        } // for j

        while( isEmpty(stack) == 1 ){
            int current_el = peek(stack);
            pop(stack);
            ComputeLocalReachabilityDensityMerged_Point(current_el, k,
                                                        dist_k_neighborhood_index, k_distance_index,
                                                        k_neighborhood_index, lrd_score_table_ptr);
        } // while
        */
    } // for i

    // FILL THE TABLE
    for( int i=0; i < num_pts; i++ ){   // iterate over all points
        idx_base = i*k;
        for( int j=0; j < k; j++ ) {      // iterate over all their neighbors
            int idx_j_neighbor = k_neighborhood_index[idx_base + j];
            // lrd_score_table_ptr[ idx_j_neighbor ] ++; // just to check that verification works
            lrd_score_neigh_table_ptr[ idx_base + j ] = lrd_score_table_ptr[ idx_j_neighbor ];
        } // for j
    } // for i

}   // ComputeLocalReachabilityDensityMerged_Pipeline2_Stack2

// POSTPROCESSING  ----------------------------------------------------------->

void verify_lrdm2_function( int num_pts, int k, my_lrdm2_fnc fun_to_verify ){

    // INITIALIZE INPUT
    double* dist_k_neighborhood_index = XmallocMatrixDoubleRandom( num_pts, k );
    double* k_distance_index = XmallocVectorDoubleRandom( num_pts );
    int* k_neighborhood_index = XmallocMatrixIntRandom( num_pts, k, num_pts-1 );

    // INITIALIZE OUTPUT
    double* lrd_score_neigh_table_ptr_true = XmallocMatrixDoubleRandom( num_pts, k );
    double* lrd_score_neigh_table_ptr = XmallocMatrixDoubleRandom( num_pts, k );

    double* lrd_score_table_ptr_true = XmallocVectorDouble( num_pts );
    double* lrd_score_table_ptr = XmallocVectorDouble( num_pts );
    // Initialize with negative values
    for(int i=0; i<num_pts; i++){
        lrd_score_table_ptr_true[i] = -1.0;
        lrd_score_table_ptr[i] = -1.0;
    }

    // VERIFICATION

    ComputeLocalReachabilityDensityMerged_Pipeline2( k, num_pts, dist_k_neighborhood_index, k_neighborhood_index,
                                                     k_distance_index, lrd_score_table_ptr_true, lrd_score_neigh_table_ptr_true);

    fun_to_verify( k, num_pts, dist_k_neighborhood_index, k_neighborhood_index,
                   k_distance_index, lrd_score_table_ptr, lrd_score_neigh_table_ptr);

    int ver = test_double_arrays( num_pts, 0.001, lrd_score_table_ptr, lrd_score_table_ptr_true);
    if( ver == 1 ){
        printf("PASSED VERIFICATION\n");
    } else {
        printf("DID NOT PASS VERIFICATION\n");
    }

    free(dist_k_neighborhood_index);
    free(k_distance_index);
    free(k_neighborhood_index);
    free(lrd_score_neigh_table_ptr_true);
    free(lrd_score_neigh_table_ptr);
    free(lrd_score_table_ptr_true);
    free(lrd_score_table_ptr);
}

// PERFORMANCE MEASUREMENT