//
// Created by pasca on 26.04.2020.
//

#include <stdlib.h>

#include "stdio.h"
#include "math.h"
#include "../include/utils.h"
#include "../include/lof_baseline.h"
#include "../unrolled/include/KNN.h"

int test_neigh_ind(int num_points, int k, const int* neigh_ind_results, const int* neigh_ind_true) {

    for (int i = 0; i < num_points; i++) {
        int ok = 0;
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                if (neigh_ind_true[i * k + j] == neigh_ind_results[i * k + l]) {
                    ok += 1;
                    break;
                }
            }
        }
        if (ok != k) {
            printf("Breaks at %d\n", i);
            return 0;
        }
    }

    return 1;
}


int test_neigh_dist(int num_points, int k, double tol, double* neigh_dist_results, double* neigh_dist_true) {
    for (int i = 0; i < num_points; i++) {
        double diff = fabs(neigh_dist_results[i] - neigh_dist_true[i * k + (k - 1)]);
        if (diff > tol) {
            printf("Breaks at %d\n", i);
            return 0;
        }
    }

    return 1;
}

int test_double_arrays(int num_points, double tol, double* lrd_results, double* lrd_true) {

    int res = 1;
    for (int i = 0; i < num_points; i++) {
        double diff = fabs(lrd_results[i] - lrd_true[i]);
        //printf("%f - %f = %f\n", lrd_results[i],lrd_true[i], diff);
        if (diff > tol) {
            printf("Breaks at %d\n", i);
            res = 0;
        }
    }
    return res;
}

int test_double_arrays_precise(int num_points, double tol, double* lrd_results, double* lrd_true) {
    /**
     * modification of test_double_arrays that shows the magnitude of the difference
     * @note: more suitable for small arrays
     * TODO: incorporate into test_double_arrays ?
     */
    int res = 1;
    for (int i = 0; i < num_points; i++) {
        double diff = fabs(lrd_results[i] - lrd_true[i]);
        if (diff > tol) {
            printf("Breaks at %d with %lf: %lf  vs  %lf\n", i, diff, lrd_results[i], lrd_true[i]);
            res = 0;
        }
    }
    if(res == 0) printf("Total number of points: %d\n", num_points);
    return res;
}



void print_matrices(int n, double* mm_true, double* mm_results) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(mm_true[i * n + j] - mm_results[i * n + j]) > 0.001) {
                printf("Diff at %d %d : %4.3lf-%4.3lf | ", i, j, mm_true[i * n + j], mm_results[i * n + j]);
            }
//            printf("%lf-%lf  ", mm_true[i * n + j], mm_results[i * n + j]);
        }
        printf("\n");
    }
}

int test_double_matrices(int num_points, double tol, double* lrd_results, double* lrd_true) {

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_points; j++) {
            double diff = fabs(lrd_results[i * num_points + j] - lrd_true[i * num_points + j]);
            if (diff > tol && (i > j)) {
                printf("Breaks at %d-%d\n", i, j);
                return 0;
            }
        }
    }
    return 1;
}


//----------------------- Verify Downstream for Pipeline 2 -------------------------->
/**
 * GOAL: verify
 * ComputeLocalReachabilityDensityMerged_Pipeline2 + ComputeLocalOutlierFactor_Pipeline2
 *                                      against
 * ComputeLocalReachabilityDensityMerged => ComputeLocalOutlierFactor
 */

void test_pipeline_2( int num_pts, int k, double tol, int isolate_lof ){
    /**
     * Verify downstream pipeline 2 against pipeline 1
     * @param isolate_lof:
     */
    printf("Testing new pipeline:\n");
    // Initialize input for Pipeline 1
    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDouble(num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixIntRandom(num_pts, k, num_pts-1);

    // LRD NEEDS TO BE FILLED WITH 0 !
    double* lrd_pipeline1 = XmallocVectorDouble( num_pts );
    double* lrd_pipeline2 = XmallocVectorDouble( num_pts );
    // INITIALIZE IT WITH NEGATIVE NUMBERS !
    for( int i=0; i < num_pts; i++ ){
        lrd_pipeline2[ i ] = -1.0;  //
    }

    double* lof_pipeline1 = XmallocVectorDouble( num_pts );
    double* lof_pipeline2 = XmallocVectorDouble( num_pts );

    // TRANSFER INPUTS FOR PIPELINE 1 to INPUT FOR PIPELINE 2 ---------------------------------->
    double* dist_k_neighborhood_index = XmallocMatrixDouble( num_pts, k );
    double* lrd_score_neigh_table_ptr = XmallocMatrixDoubleRandom( num_pts, k );

    double* closest_neighb = (double*) malloc( k * sizeof(double));
    for(int i=0; i < num_pts; i++){

        for(int j=0; j < k; j++){

            int idx_j_neigh = neighborhood_index_table_ptr[ i*k + j ];
            int idx_dist;
            if (i < idx_j_neigh) {
                idx_dist = num_pts * i + idx_j_neigh;
            } else {
                idx_dist = num_pts * idx_j_neigh + i;
            }

            closest_neighb[ j ] = distances_indexed_ptr[idx_dist];
            dist_k_neighborhood_index[ i*k + j ] = distances_indexed_ptr[idx_dist];
        }   // for j

        //qsort(closest_neighb, k, sizeof(double), compare_double);
        //for(int j=0; j < k; j++) {
        //    printf("%d closest: %lf  vs original %lf\n", j, closest_neighb[j], dist_k_neighborhood_index[i * k + j]);
        //    dist_k_neighborhood_index[i * k + j] = closest_neighb[ j ];
        //}

    } // for i

    // fill k_distances_indexed_ptr based on distances_indexed_ptr and neighborhood_index_table_ptr
    for( int i=0; i < num_pts; i++ ){
        double max = -1.0;     // K distance is the max distance among k nearest neighbors
        for( int j=0; j < k; j++ ){
            if( dist_k_neighborhood_index[ i*k + j ] > max ){
                max = dist_k_neighborhood_index[ i*k + j ];
            }
        } // for j
        k_distances_indexed_ptr[ i ] = max;
    } // for i


    // COMPARE OUTPUTS ------------------------------------------------------------------------------>

    ComputeLocalReachabilityDensityMerged(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                          neighborhood_index_table_ptr, lrd_pipeline1);

    ComputeLocalReachabilityDensityMerged_Pipeline2( k, num_pts, ComputeLocalReachabilityDensityMerged_Point,
                                                     dist_k_neighborhood_index, neighborhood_index_table_ptr,
                                                     k_distances_indexed_ptr, lrd_pipeline2, lrd_score_neigh_table_ptr);

    int ver_lrd = test_double_arrays_precise( num_pts, tol, lrd_pipeline1, lrd_pipeline2);

    if( ver_lrd == 0 ){
        printf("Resulting LRD scores are different\n");
    } else {
        printf( "LRD passed verification\n" );
    }

    // Alternatively, compute lrd_score_neigh_table_ptr based on lrd_pipeline1 and
    // see if it passes verification in that case => concentrate effort on LRD
    if( isolate_lof == 1 ){
        for(int i=0; i < num_pts; ++i ){
            for(int j=0; j < k; ++j ){
                int idx_j_neigh = neighborhood_index_table_ptr[ i*k + j ];
                //printf("i = %d, j = %d : access = %d\n", i, j, idx_j_neigh);
                lrd_score_neigh_table_ptr[ i*k + j ] = lrd_pipeline1[ idx_j_neigh ];
            }
        }
    }   // if( isolate_lof == 1 )

    ComputeLocalOutlierFactor(k, num_pts, lrd_pipeline1, neighborhood_index_table_ptr, lof_pipeline1 );
    ComputeLocalOutlierFactor_Pipeline2( k, num_pts, lrd_pipeline2, lrd_score_neigh_table_ptr, lof_pipeline2 );

    int ver_lof = test_double_arrays( num_pts, tol, lof_pipeline1, lof_pipeline2);
    if( ver_lof == 0 ){
        printf("Resulting LOF scores are different\n");
    } else {
        printf( "LOF passed verification\n" );
    }
}


void test_pipeline_2_ver2( int dim, int num_pts, int k, double tol, my_lrdm2_fnc  lrdm2_fnc){

    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* pairwise_dist = XmallocMatrixDouble(num_pts, num_pts);
    double* dist_k_neighborhood_index = XmallocMatrixDoubleRandom(num_pts, k);
    int* k_neighborhood_index = XmallocMatrixInt(num_pts, k);
    double* k_distance_index = XmallocVectorDouble(num_pts);
    double* lrd_score_neigh_table_ptr = XmallocMatrixDouble(num_pts, k);


    double* lrd_score_table_ptr_true = XmallocVectorDouble(num_pts);
    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    for (int i = 0; i < num_pts; i++) {
        lrd_score_table_ptr[i] = -1.0;
    }

    // INITILIZE OUTPUT
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);

    // PIPELINE BASELINE:
    // Step 1: compute pairwise distances
    ComputePairwiseDistances(dim, num_pts, input_points_ptr, EuclideanDistance, pairwise_dist);

    // Step 2: compute k distances
    ComputeKDistanceAll(k, num_pts, pairwise_dist, k_distance_index);

    // Step 3: compute k neighborhood
    ComputeKDistanceNeighborhoodAll(num_pts, k, k_distance_index, pairwise_dist, k_neighborhood_index);

    // Step 4 + 5: compute reachability distance and reachability density
    ComputeLocalReachabilityDensityMerged(k, num_pts, pairwise_dist, k_distance_index, k_neighborhood_index,
             lrd_score_table_ptr_true);



    // PIPELINE 2

    ComputePairwiseDistances(dim, num_pts, input_points_ptr, EuclideanDistance, pairwise_dist);

    KNN_fastest(num_pts, k, k_neighborhood_index, pairwise_dist, dist_k_neighborhood_index);

    lrdm2_fnc(k, num_pts, ComputeLocalReachabilityDensityMerged_Point, dist_k_neighborhood_index, k_neighborhood_index, k_distance_index,
              lrd_score_table_ptr, lrd_score_neigh_table_ptr);

    int ver_lrdm = test_double_arrays( num_pts, 0.0001, lrd_score_table_ptr, lrd_score_table_ptr_true );
    if( ver_lrdm == 0 ){
        printf("Resulting LRD scores are different\n");
    } else {
        printf( "LRD passed verification\n" );
    }

}
