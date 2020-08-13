#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>


#include "../include/tests.h"
#include "../include/utils.h"
#include "../include/tsc_x86.h"
#include "../include/file_utils.h"
#include "../include/lof_baseline.h"
#include "../include/metrics.h"
#include "../include/Algorithm.h"
#include "../avx/include/MMMAvx.h"


#define NUM_RUNS 5

// TODO !!!!! CHECK ORDER OF ARGUMENTS !!!
//******************************************************************************************************//
//                  Pipeline 1
//******************************************************************************************************//

double* algorithm_driver_baseline_with_individual_measurement(int num_pts, int k, int dim, enum Mode mode,
                                                              FILE* results_file, FILE* exec_file,
                                                              my_dist_fnc dist_fnc, my_kdistall_fnc kdist_fnc,
                                                              my_neigh_fnc neigh_fnc, my_rdall_fnc rdall_fnc,
                                                              my_lrdm_fnc lrdm_fnc, my_lof_fnc lof_fnc
) {
    /** Running the entire pipeline potentially with different subfunctions
     *
     *@param num_pts: number of points in the dataset
     *@param k: number of neighbors to consider
     *@param dim: dimensionality of the input
     *@param exec_file: ???
     *@param results_file: ???
     *
     *@param :
     *
     *TODO:
     * 1. FIGURE OUT FILE LOCATIONS
     * 2. WE HAVE BOTH DEF 6 and DEF 5&6 - WHICH ONE TO TAKE ?
     *
    */


    char dir_full[50];
    sprintf(dir_full, "/n%d_k%d_dim%d/", num_pts, k, dim);
    char* input_file_name = concat(dir_full, "dataset.txt");
    char* lof_results_file_name = concat(dir_full, "lof_results.txt");

    // Step 1: read the input
    FILE* input_file = open_with_error_check(input_file_name, "r");
    FILE* lof_results_file = open_with_error_check(lof_results_file_name, "r");

    // just a multidimensional test
    char* input_meta_file_name = concat(dir_full, "metadata.txt");
    FILE* input_meta_file = open_with_error_check(input_meta_file_name, "r");
    double* min_range = NULL;
    double* max_range = NULL;

    double* input_points_ptr = LoadGenericDataFromFile(input_file, input_meta_file,
                                                       min_range, max_range, &num_pts, &dim, &k);

    double* lof_true_ptr = LoadResultsFromFile(lof_results_file, num_pts);


    // Initialize all important matrices / vectors used
    double* distances_indexed_ptr = XmallocMatrixDouble(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDouble(num_pts);
    double* reachability_distances_indexed_ptr = XmallocVectorDouble(num_pts * num_pts);
    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);

    int i, num_runs;
    double cyclesStep2, cyclesStep3, cyclesStep4, cyclesStep5, cyclesStep6, cyclesStep7;
    myInt64 start;
    double temp;
    num_runs = NUM_RUNS;

    //calculating cycles
    start = start_tsc();
    // Step 2: compute pairwise distance between points
    for (i = 1; i < num_runs; ++i) {
        dist_fnc(dim, num_pts, input_points_ptr, UnrolledEuclideanDistance, distances_indexed_ptr);
    }
    cyclesStep2 = stop_tsc(start) / (double) num_runs;


    start = start_tsc();
    // Step 3: compute K distances
    for (i = 0; i < num_runs; ++i) {
        kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    }
    cyclesStep3 = stop_tsc(start) / (double) num_runs;


    //calculating cycles
    start = start_tsc();
    //Step 4:  compute K neighborhoods
    for (i = 0; i < num_runs; ++i) {
        neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);
    }
    cyclesStep4 = stop_tsc(start) / (double) num_runs;


    start = start_tsc();
    // Step 5: compute pairwise Reachability distance
    for (i = 0; i < num_runs; ++i) {
        rdall_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, reachability_distances_indexed_ptr);
    }
    cyclesStep5 = stop_tsc(start) / (double) num_runs;


    start = start_tsc();
    // Step 6: compute reachability density
    for (i = 0; i < num_runs; ++i) {
        lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr,
                 lrd_score_table_ptr);
    }
    cyclesStep6 = stop_tsc(start) / (double) num_runs;

    //calculating cycles
    start = start_tsc();
    // Step 7: compute lof
    for (i = 1; i < num_runs; ++i) {
        lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    }
    cyclesStep7 = stop_tsc(start) / (double) num_runs;

    double dif = 0;
    for (i = 0; i < num_pts; i++) {
        dif += fabs(lof_score_table_ptr[i] - lof_true_ptr[i]);
    }


    switch (mode) {
        case BASIC:
            printf(" hey hey just testing ");
            assert(fabs(dif) < 1e-3);
            break;
        case TESTING: {
            // TODO: put it into a separate test file ?
            FILE* neigh_ind_results_file = open_with_error_check(concat(dir_full, "neigh_ind_results.txt"), "r");
            FILE* neigh_dist_results_file = open_with_error_check(concat(dir_full, "neigh_dist_results.txt"), "r");
            FILE* lrd_results_file = open_with_error_check(concat(dir_full, "lrd_results.txt"), "r");
            int* neigh_ind_true_ptr = LoadNeighIndResultsFromFile(neigh_ind_results_file, num_pts, k);
            double* neigh_dist_true_ptr = LoadNeighDistResultsFromFile(neigh_dist_results_file, num_pts, k);
            double* lrd_true_ptr = LoadLRDResultsFromFile(lrd_results_file, num_pts);

            int test_neigh_dist_result = test_neigh_dist(num_pts, k, 1e-3, k_distances_indexed_ptr,
                                                         neigh_dist_true_ptr);
            assert(test_neigh_dist_result == 1);

            int test_neigh_ind_result = test_neigh_ind(num_pts, k, neighborhood_index_table_ptr, neigh_ind_true_ptr);
            assert(test_neigh_ind_result == 1);

            int test_lrd_result = test_double_arrays(num_pts, 1e-5, lrd_score_table_ptr, lrd_true_ptr);
            assert(test_lrd_result == 1);

            assert(fabs(dif) < 1e-3);
        }
            break;
        case BENCHMARK: {
            double total = cyclesStep2 + cyclesStep3 + cyclesStep4 + cyclesStep5 + cyclesStep6 + cyclesStep7;
            fprintf(results_file, "%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", num_pts, k, dim, cyclesStep2, cyclesStep3,
                    cyclesStep4, cyclesStep5, cyclesStep6, cyclesStep7, total);
            printf("%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", num_pts, k, dim, cyclesStep2, cyclesStep3,
                   cyclesStep4, cyclesStep5, cyclesStep6, cyclesStep7, total, dif);
            double freq = 2.6e9;
            fprintf(exec_file, "%d,%d,%d,%lf, %lf\n", num_pts, k, dim, total / freq, dif);
            //assert(fabs(dif) < 1e-3);
            break;
        }
    }
}


double algorithm_driver_baseline(int num_pts, int k, int dim,
                                 my_metrics_fnc metrics_fnc,
                                 my_dist_fnc dist_fnc,
                                 my_kdistall_fnc kdist_fnc,
                                 my_neigh_fnc neigh_fnc,
                                 my_lrdm_fnc lrdm_fnc,
                                 my_lof_fnc lof_fnc) {
    /**
     * @note: currently there is no verification,
     * all individual functions are assumed to be verified
     *
     * @return: average total runtime in cycles
     *
     * USAGE: see unrolled/main.c
     *
     * TODO: function returns number of
     */

    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* distances_indexed_ptr = XmallocMatrixDouble(num_pts, num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);
    double* k_distances_indexed_ptr = XmallocVectorDouble(num_pts);
    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);

    // INITIALIZE OUTPUT
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);



    // COMPUTE THE TOTAL NUMBER OF FLOPS:
    long long flops_total = 0.0;
    flops_total += (long long) dist_fnc(dim, num_pts, input_points_ptr, metrics_fnc, distances_indexed_ptr);

    flops_total += (long long) kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);

    flops_total += (long long) neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr,
                                         neighborhood_index_table_ptr);

    flops_total += (long long) lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                        neighborhood_index_table_ptr,
                                        lrd_score_table_ptr);
    flops_total += (long long) lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr,
                                       lof_score_table_ptr);

    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;
    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {

        // Step 1: compute pairwise distances
        dist_fnc(dim, num_pts, input_points_ptr, metrics_fnc, distances_indexed_ptr);

        // Step 2: compute k distances
        kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);

        // Step 3: compute k neighborhood
        neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);

        // Step 4 + 5: compute reachability distance and reachability density
        lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr,
                 lrd_score_table_ptr);

        // Step 5: compute local outlier factor
        lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(distances_indexed_ptr);
    free(neighborhood_index_table_ptr);
    free(k_distances_indexed_ptr);
    free(lrd_score_table_ptr);
    free(lof_score_table_ptr);

    return perf;
}

/*
double algorithm_driver_baseline_razavn(int num_pts, int k, int dim,
                                        int B0, int B1,
                                        my_dist_block_fnc dist_block_fnc,
                                        my_kdistall_fnc kdist_fnc,
                                        my_neigh_fnc neigh_fnc,
                                        my_lrdm_fnc lrdm_fnc,
                                        my_lof_fnc lof_fnc){


    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* distances_indexed_ptr = XmallocMatrixDouble(num_pts, num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);
    double* k_distances_indexed_ptr = XmallocVectorDouble(num_pts);
    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);

    // INITIALIZE OUTPUT
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);

    //Choosing block size
    int BK_adj = dim / 4;
    BK_adj *= 4;
    BK_adj = BK_adj == 0 ? dim : BK_adj * 4;

    // COMPUTE THE TOTAL NUMBER OF FLOPS:
    double flops_total = 0.0;
    flops_total += dist_block_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, distances_indexed_ptr);
    //printf("Dist bloc flops: %lf\n", flops_total);
    flops_total += kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    //printf("Kdist bloc flops: %lf\n", flops_total);
    flops_total += neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);
    //printf("Neight bloc flops: %lf\n", flops_total);
    flops_total += lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr,
                            lrd_score_table_ptr);
    //printf("LRDM flops: %lf\n", flops_total);
    flops_total += lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    //printf("LOF flops: %lf\n", flops_total);

    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;
    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {

        // Step 1: compute pairwise distances
        dist_block_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, distances_indexed_ptr);

        // Step 2: compute k distances
        kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);

        // Step 3: compute k neighborhood
        neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);

        // Step 4 + 5: compute reachability distance and reachability density
        lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr,
                 lrd_score_table_ptr);

        // Step 5: compute local outlier factor
        lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = ( 1000.0 * flops_total / cycles_total ) / 1000.0;

    free(input_points_ptr);
    free(distances_indexed_ptr);
    free(neighborhood_index_table_ptr);
    free(k_distances_indexed_ptr);
    free(lrd_score_table_ptr);
    free(lof_score_table_ptr);

    return perf;


}*/

//******************************************************************************************************//
//                  Pipeline 2
//
// KNN_HASH => TODO  => ComputeLocalOutlierFactor_Pipeline2
//******************************************************************************************************//

double algorithm_driver_baseline_mmm_pairwise_distance(int num_pts, int k, int dim,
                                                       int B0, int B1,
                                                       my_mmm_dist_fnc mmm_dist_fnc,
                                                       my_kdistall_fnc kdist_fnc,
                                                       my_neigh_fnc neigh_fnc,
                                                       my_lrdm_fnc lrdm_fnc,
                                                       my_lof_fnc lof_fnc) {


    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* distances_indexed_ptr = XmallocMatrixDouble(num_pts, num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);
    double* k_distances_indexed_ptr = XmallocVectorDouble(num_pts);
    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);

    // INITIALIZE OUTPUT
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);

    //Choosing block size
    int BK_adj = dim / 4;
    BK_adj *= 4;
    BK_adj = BK_adj == 0 ? dim : BK_adj * 4;

    // COMPUTE THE TOTAL NUMBER OF FLOPS:
    double flops_total = 0.0;
    flops_total += mmm_dist_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, distances_indexed_ptr);
    //printf("Dist bloc flops: %lf\n", flops_total);
    flops_total += kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    //printf("Kdist bloc flops: %lf\n", flops_total);
    flops_total += neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);
    //printf("Neight bloc flops: %lf\n", flops_total);

    flops_total += lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr,
                            lrd_score_table_ptr);
    //printf("LRDM flops: %lf\n", flops_total);
    flops_total += lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    //printf("LOF flops: %lf\n", flops_total);

    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;
    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {

        // Step 1: compute pairwise distances
        mmm_dist_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, distances_indexed_ptr);

        // Step 2: compute k distances
        kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);

        // Step 3: compute k neighborhood
        neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);

        // Step 4 + 5: compute reachability distance and reachability density
        lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr,
                 lrd_score_table_ptr);

        // Step 5: compute local outlier factor
        lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(distances_indexed_ptr);
    free(neighborhood_index_table_ptr);
    free(k_distances_indexed_ptr);
    free(lrd_score_table_ptr);
    free(lof_score_table_ptr);

    return perf;

}

double algorithm_driver_knn_memory_struct(int num_pts, int k, int dim,
                                          my_metrics_fnc metric_fnc,
                                          my_dist_fnc dist_fnc,
                                          my_knn_fnc knn_fnc,
                                          my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                          my_lrdm2_fnc lrdm2_fnc,
                                          my_lof2_fnc lof2_fnc) {
    /**
     * @param dist_fnc: function for pairwise distance calculation
     * @param knn_fnc: one of the modifications of KNN
     *
     * @return: average total runtime in cycles
     *
     * TODO: 1. add file output ?
     *       2. try to drop certain elements from cache during the runtime ?
     *
     * USAGE: see unrolled/main.c
     */

    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* pairwise_dist = XmallocMatrixDouble(num_pts, num_pts);
    double* dist_k_neighborhood_index = XmallocMatrixDouble(num_pts, k);
    int* k_neighborhood_index = XmallocMatrixInt(num_pts, k);
    double* k_distance_index = XmallocVectorDouble(num_pts);
    double* lrd_score_neigh_table_ptr = XmallocMatrixDouble(num_pts, k);

    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    for (int i = 0; i < num_pts; i++) {
        lrd_score_table_ptr[i] = -1.0;
    }

    // INITILIZE OUTPUT
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);

    // COMPUTE TOTAL NUMBER OF FLOPS:
    double flops_total = 0;

    flops_total += dist_fnc(dim, num_pts, input_points_ptr, metric_fnc, pairwise_dist);
    flops_total += knn_fnc(num_pts, k, k_neighborhood_index, pairwise_dist, dist_k_neighborhood_index);
    flops_total += lrdm2_fnc(k, num_pts, lrdm2_pnt_fnc, dist_k_neighborhood_index, k_neighborhood_index,
                             k_distance_index,
                             lrd_score_table_ptr, lrd_score_neigh_table_ptr);
    flops_total += lof2_fnc(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;

    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {

        // Compute Pairwise Distance
        dist_fnc(dim, num_pts, input_points_ptr, metric_fnc, pairwise_dist);

        knn_fnc(num_pts, k, k_neighborhood_index, pairwise_dist, dist_k_neighborhood_index);

        lrdm2_fnc(k, num_pts, lrdm2_pnt_fnc, dist_k_neighborhood_index, k_neighborhood_index, k_distance_index,
                  lrd_score_table_ptr, lrd_score_neigh_table_ptr);

        lof2_fnc(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);
    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(pairwise_dist);
    free(dist_k_neighborhood_index);
    free(k_neighborhood_index);
    free(k_distance_index);
    free(lrd_score_neigh_table_ptr);
    free(lrd_score_table_ptr);
    free(lof_score_table_ptr);

    return perf;

}

//******************************************************************************************************//
//                  Pipeline 3: Blocked Memory Structure
//******************************************************************************************************//


double algorithm_driver_knn_mmm_pairwise_dist(int num_pts, int k, int dim,
                                              int B0, int B1,
                                              my_mmm_dist_fnc mmm_dist_fnc,
                                              my_knn_fnc knn_fnc,
                                              my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                              my_lrdm2_fnc lrdm2_fnc,
                                              my_lof2_fnc lof2_fnc) {
    /**
     * @param B0: size of the cache block for num of points
     * @param B1:
     * @param BK: size of the cache block for num of dimensions
     */

    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* pairwise_dist = XmallocMatrixDouble(num_pts, num_pts);
    double* dist_k_neighborhood_index = XmallocMatrixDouble(num_pts, k);
    int* k_neighborhood_index = XmallocMatrixInt(num_pts, k);
    double* k_distance_index = XmallocVectorDouble(num_pts);
    double* lrd_score_neigh_table_ptr = XmallocMatrixDouble(num_pts, k);

    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    for (int i = 0; i < num_pts; i++) {
        lrd_score_table_ptr[i] = -1.0;
    }

    // INITILIZE OUTPUT
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);

    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;

    //Choosing block size
    int BK_adj = dim / 4;
    BK_adj *= 4;
    BK_adj = BK_adj == 0 ? dim : BK_adj * 4;

    // RUN EVERYTHING ONCE TO GET THE NUMBER
    double flops_total = 0;

    flops_total += mmm_dist_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, pairwise_dist);
    flops_total += knn_fnc(num_pts, k, k_neighborhood_index, pairwise_dist, dist_k_neighborhood_index);
    flops_total += lrdm2_fnc(k, num_pts, lrdm2_pnt_fnc, dist_k_neighborhood_index, k_neighborhood_index,
                             k_distance_index,
                             lrd_score_table_ptr, lrd_score_neigh_table_ptr);
    flops_total += lof2_fnc(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {
        // Compute Pairwise Distance

        mmm_dist_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, pairwise_dist);

        knn_fnc(num_pts, k, k_neighborhood_index, pairwise_dist, dist_k_neighborhood_index);

        lrdm2_fnc(k, num_pts, lrdm2_pnt_fnc, dist_k_neighborhood_index, k_neighborhood_index, k_distance_index,
                  lrd_score_table_ptr, lrd_score_neigh_table_ptr);

        lof2_fnc(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);
    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(pairwise_dist);
    free(dist_k_neighborhood_index);
    free(k_neighborhood_index);
    free(k_distance_index);
    free(lrd_score_neigh_table_ptr);
    free(lrd_score_table_ptr);
    free(lof_score_table_ptr);

    return perf;

}

//******************************************************************************************************//
//                  Pipeline 4: Lattice
//******************************************************************************************************//

double algorithm_driver_lattice(int num_pts, int k, int dim,
                                int num_splits, int resolution,
                                my_topolofy_fnc topolofy_fnc,
                                my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                my_lrdm2_fnc lrdm2_fnc,
                                my_lof2_fnc lof2_fnc) {
    /**
     * @param resolution ?
     * @param num_splits ?
     */

    // TIMING INFRASTRUCTURE
    myInt64 start, start_insertion;
    int num_runs = NUM_RUNS;
    double cycles_total, cycles_insertion;

    // INITIALIZE INPUT
    double *input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    MULTI_LATTICE lattice;
    double *min_range_ptr = XmallocVectorDouble(dim);
    double *max_range_ptr = XmallocVectorDouble(dim);
    // compute min and max for each dimension -> should it be included
    double max, min, current;

    for (int d = 0; d < dim; d++) {
        max = -1.0;
        min = 1000000.0; // I think random numbers are between 0 and 1 but anyway
        for (int pnt_idx = 0; pnt_idx < num_pts; pnt_idx++) {
            current = input_points_ptr[pnt_idx * dim + d];      // CHECK THE ORDER !!!
            if (current > max) {
                max = current;
            }
            if (current < min) {
                min = current;
            }
        } // for pnt_idx
        min_range_ptr[d] = min;
        max_range_ptr[d] = max;
    } // for dim

    // INITIALIZE INTERMEDIATE OBJECTS
    double *pairwise_dist = XmallocMatrixDouble(num_pts, num_pts);
    double *dist_k_neighborhood_index = XmallocMatrixDoubleRandom(num_pts, k);
    int *k_neighborhood_index = XmallocMatrixInt(num_pts, k);
    double *k_distance_index = XmallocVectorDouble(num_pts);
    double *lrd_score_neigh_table_ptr = XmallocMatrixDouble(num_pts, k);

    double *lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    for (int i = 0; i < num_pts; i++) {
        lrd_score_table_ptr[i] = -1.0;
    }

    // INITILIZE OUTPUT
    double *lof_score_table_ptr = XmallocVectorDouble(num_pts);

    // RUN EVERYTHING ONCE TO GET THE NUMBER
    double flops_total = 0;

    lattice = BuildLattice(dim, num_splits, resolution, min_range_ptr, max_range_ptr);
    // fill the lattice before expecting it to work
    start_insertion = start_tsc();
    for (int i = 0; i < num_pts; ++i) {
        InsertElement(&lattice, input_points_ptr + i * dim, i, num_splits, dim);
    }
    cycles_insertion = stop_tsc(start_insertion);

    flops_total += 6 * num_splits * num_pts; // FOR INSERTING ELEMENTS TO LATTICE !!!
    flops_total += (double) topolofy_fnc(&lattice, input_points_ptr, k_neighborhood_index, dist_k_neighborhood_index,
                                         k, num_pts, num_splits, dim);
    flops_total += lrdm2_fnc(k, num_pts, lrdm2_pnt_fnc, dist_k_neighborhood_index, k_neighborhood_index,
                             k_distance_index,
                             lrd_score_table_ptr, lrd_score_neigh_table_ptr);
    flops_total += lof2_fnc(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    // ALGORITHM  LoadTopologyInfo
    start = start_tsc();

    for (int t = 0; t < num_runs; t++) {
        topolofy_fnc(&lattice, input_points_ptr, k_neighborhood_index, dist_k_neighborhood_index,
                     k, num_pts, num_splits, dim);

        lrdm2_fnc(k, num_pts, lrdm2_pnt_fnc, dist_k_neighborhood_index, k_neighborhood_index, k_distance_index,
                  lrd_score_table_ptr, lrd_score_neigh_table_ptr);

        lof2_fnc(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);
    }

    cycles_total = cycles_insertion + stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(pairwise_dist);
    free(dist_k_neighborhood_index);
    free(k_neighborhood_index);
    free(k_distance_index);
    free(lrd_score_neigh_table_ptr);
    free(lrd_score_table_ptr);
    free(lof_score_table_ptr);

    return perf;

}

// ********************************************************************************************************************>
//
//      IMPROVEMENT MEASUREMENTS FOR THE SECOND PART OF THE PIPELINE
//
// ********************************************************************************************************************>

double algorithm_driver_second_part_original(int num_pts, int k, int dim,
                                             my_lrdm_fnc lrdm_fnc,
                                             my_lof_fnc lof_fnc) {
    /**
     * @note: currently there is no verification,
     * all individual functions are assumed to be verified
     *
     * @return: average total runtime in cycles
     *
     * USAGE: see unrolled/main.c
     *
     * TODO: function returns number of
     */

    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixIntRandom(num_pts, k, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDoubleRandom(num_pts);

    // INITIALIZE OUTPUT
    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);

    // COMPUTE THE TOTAL NUMBER OF FLOPS:
    long long flops_total = 0.0;

    flops_total += (long long) lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                        neighborhood_index_table_ptr,
                                        lrd_score_table_ptr);
    flops_total += (long long) lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr,
                                       lof_score_table_ptr);

    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;
    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {
        // Step 4 + 5: compute reachability distance and reachability density
        lrdm_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr,
                 lrd_score_table_ptr);

        // Step 5: compute local outlier factor
        lof_fnc(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(distances_indexed_ptr);
    free(neighborhood_index_table_ptr);
    free(k_distances_indexed_ptr);
    free(lrd_score_table_ptr);
    free(lof_score_table_ptr);

    return perf;
}

double algorithm_driver_second_part_memory_optimization(int num_pts, int k, int dim,
                                                        my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                                        my_lrdm2_fnc lrdm2_fnc,
                                                        my_lof2_fnc lof2_fnc) {
    /**
     * @param dist_fnc: function for pairwise distance calculation
     * @param knn_fnc: one of the modifications of KNN
     *
     * @return: average total runtime in cycles
     *
     * TODO: 1. add file output ?
     *       2. try to drop certain elements from cache during the runtime ?
     *
     * USAGE: see unrolled/main.c
     */

    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* pairwise_dist = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* dist_k_neighborhood_index = XmallocMatrixDoubleRandom(num_pts, k);
    int* k_neighborhood_index = XmallocMatrixIntRandom(num_pts, k, num_pts);
    double* k_distance_index = XmallocVectorDoubleRandom(num_pts);

    // INITILIZE OUTPUT
    double* lrd_score_neigh_table_ptr = XmallocMatrixDouble(num_pts, k);
    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);

    for (int i = 0; i < num_pts; i++) {
        lrd_score_table_ptr[i] = -1.0;
    }

    // COMPUTE TOTAL NUMBER OF FLOPS:
    double flops_total = 0;

    flops_total += lrdm2_fnc(k, num_pts, lrdm2_pnt_fnc, dist_k_neighborhood_index, k_neighborhood_index,
                             k_distance_index,
                             lrd_score_table_ptr, lrd_score_neigh_table_ptr);
    flops_total += lof2_fnc(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;

    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {

        lrdm2_fnc(k, num_pts, lrdm2_pnt_fnc, dist_k_neighborhood_index, k_neighborhood_index, k_distance_index,
                  lrd_score_table_ptr, lrd_score_neigh_table_ptr);

        lof2_fnc(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(pairwise_dist);
    free(dist_k_neighborhood_index);
    free(k_neighborhood_index);
    free(k_distance_index);
    free(lrd_score_neigh_table_ptr);
    free(lrd_score_table_ptr);
    free(lof_score_table_ptr);

    return perf;
}

// ********************************************************************************************************************>
//
//      IMPROVEMENT MEASUREMENTS FOR THE FIRST PART OF THE PIPELINE
//
// ********************************************************************************************************************>
// Basically comment out / delete certain parts of the following pipelines:
// 1. algorithm_driver_baseline
// 2. algorithm_driver_baseline_mmm_pairwise_distance
// 3. algorithm_driver_knn_memory_struct
// 4. algorithm_driver_lattice



double algorithm_driver_first_part_baseline(int num_pts, int k, int dim,
                                            my_metrics_fnc metrics_fnc,
                                            my_dist_fnc dist_fnc,
                                            my_kdistall_fnc kdist_fnc,
                                            my_neigh_fnc neigh_fnc) {

    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* distances_indexed_ptr = XmallocMatrixDouble(num_pts, num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);
    double* k_distances_indexed_ptr = XmallocVectorDouble(num_pts);


    // COMPUTE THE TOTAL NUMBER OF FLOPS:
    long long flops_total = 0.0;
    flops_total += (long long) dist_fnc(dim, num_pts, input_points_ptr, metrics_fnc, distances_indexed_ptr);

    flops_total += (long long) kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);

    flops_total += (long long) neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr,
                                         neighborhood_index_table_ptr);


    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;
    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {

        // Step 1: compute pairwise distances
        dist_fnc(dim, num_pts, input_points_ptr, metrics_fnc, distances_indexed_ptr);
        // Step 2: compute k distances
        kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
        // Step 3: compute k neighborhood
        neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);

    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(distances_indexed_ptr);
    free(neighborhood_index_table_ptr);
    free(k_distances_indexed_ptr);

    return perf;
}


double algorithm_driver_first_part_mmm_pairwise_distance(int num_pts, int k, int dim,
                                                         int B0, int B1,
                                                         my_mmm_dist_fnc mmm_dist_fnc,
                                                         my_kdistall_fnc kdist_fnc,
                                                         my_neigh_fnc neigh_fnc) {


    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* distances_indexed_ptr = XmallocMatrixDouble(num_pts, num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);
    double* k_distances_indexed_ptr = XmallocVectorDouble(num_pts);


    //Choosing block size
    int BK_adj = dim / 4;
    BK_adj *= 4;
    BK_adj = BK_adj == 0 ? dim : BK_adj * 4;

    // COMPUTE THE TOTAL NUMBER OF FLOPS:
    double flops_total = 0.0;
    flops_total += mmm_dist_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, distances_indexed_ptr);
    flops_total += kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    flops_total += neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);

    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;
    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {

        // Step 1: compute pairwise distances
        mmm_dist_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, distances_indexed_ptr);
        // Step 2: compute k distances
        kdist_fnc(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
        // Step 3: compute k neighborhood
        neigh_fnc(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr, neighborhood_index_table_ptr);

    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(distances_indexed_ptr);
    free(neighborhood_index_table_ptr);
    free(k_distances_indexed_ptr);

    return perf;

}


double algorithm_driver_first_part_knn_mmm_pairwise_dist(int num_pts, int k, int dim,
                                              int B0, int B1,
                                              my_mmm_dist_fnc mmm_dist_fnc,
                                              my_knn_fnc knn_fnc) {
    /**
     * @param B0: size of the cache block for num of points
     * @param B1:
     * @param BK: size of the cache block for num of dimensions
     */

    // INITIALIZE INPUT
    double* input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    // INITIALIZE INTERMEDIATE OBJECTS
    double* pairwise_dist = XmallocMatrixDouble(num_pts, num_pts);
    double* dist_k_neighborhood_index = XmallocMatrixDouble(num_pts, k);
    int* k_neighborhood_index = XmallocMatrixInt(num_pts, k);
    double* k_distance_index = XmallocVectorDouble(num_pts);


    // TIMING INFRASTRUCTURE
    myInt64 start;
    int num_runs = NUM_RUNS;
    double cycles_total;

    //Choosing block size
    int BK_adj = dim / 4;
    BK_adj *= 4;
    BK_adj = BK_adj == 0 ? dim : BK_adj * 4;

    // RUN EVERYTHING ONCE TO GET THE NUMBER
    double flops_total = 0;

    flops_total += mmm_dist_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, pairwise_dist);
    flops_total += knn_fnc(num_pts, k, k_neighborhood_index, pairwise_dist, dist_k_neighborhood_index);

    // ALGORITHM
    start = start_tsc();
    for (int t = 0; t < num_runs; t++) {
        // Compute Pairwise Distance

        mmm_dist_fnc(num_pts, dim, B0, B1, BK_adj, input_points_ptr, pairwise_dist);

        knn_fnc(num_pts, k, k_neighborhood_index, pairwise_dist, dist_k_neighborhood_index);

    }

    cycles_total = stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(pairwise_dist);
    free(dist_k_neighborhood_index);
    free(k_neighborhood_index);
    free(k_distance_index);

    return perf;

}

//******************************************************************************************************//
//                  Pipeline 4: Lattice
//******************************************************************************************************//

double algorithm_driver_first_part_lattice(int num_pts, int k, int dim,
                                int num_splits, int resolution,
                                my_topolofy_fnc topolofy_fnc) {
    /**
     * @param resolution ?
     * @param num_splits ?
     */

    // TIMING INFRASTRUCTURE
    myInt64 start, start_insertion;
    int num_runs = NUM_RUNS;
    double cycles_total, cycles_insertion;

    // INITIALIZE INPUT
    double *input_points_ptr = XmallocMatrixDoubleRandom(num_pts, dim);

    MULTI_LATTICE lattice;
    double *min_range_ptr = XmallocVectorDouble(dim);
    double *max_range_ptr = XmallocVectorDouble(dim);
    // compute min and max for each dimension -> should it be included
    double max, min, current;

    for (int d = 0; d < dim; d++) {
        max = -1.0;
        min = 1000000.0; // I think random numbers are between 0 and 1 but anyway
        for (int pnt_idx = 0; pnt_idx < num_pts; pnt_idx++) {
            current = input_points_ptr[pnt_idx * dim + d];      // CHECK THE ORDER !!!
            if (current > max) {
                max = current;
            }
            if (current < min) {
                min = current;
            }
        } // for pnt_idx
        min_range_ptr[d] = min;
        max_range_ptr[d] = max;
    } // for dim

    // INITIALIZE INTERMEDIATE OBJECTS
    double *pairwise_dist = XmallocMatrixDouble(num_pts, num_pts);
    double *dist_k_neighborhood_index = XmallocMatrixDoubleRandom(num_pts, k);
    int *k_neighborhood_index = XmallocMatrixInt(num_pts, k);
    double *k_distance_index = XmallocVectorDouble(num_pts);


    // RUN EVERYTHING ONCE TO GET THE NUMBER
    double flops_total = 0;

    lattice = BuildLattice(dim, num_splits, resolution, min_range_ptr, max_range_ptr);
    // fill the lattice before expecting it to work
    start_insertion = start_tsc();
    for (int i = 0; i < num_pts; ++i) {
        InsertElement(&lattice, input_points_ptr + i * dim, i, num_splits, dim);
    }
    cycles_insertion = stop_tsc(start_insertion);

    flops_total += 6 * num_splits * num_pts; // FOR INSERTING ELEMENTS TO LATTICE !!!
    flops_total += (double) topolofy_fnc(&lattice, input_points_ptr, k_neighborhood_index, dist_k_neighborhood_index,
                                         k, num_pts, num_splits, dim);

    // ALGORITHM  LoadTopologyInfo
    start = start_tsc();

    for (int t = 0; t < num_runs; t++) {
        topolofy_fnc(&lattice, input_points_ptr, k_neighborhood_index, dist_k_neighborhood_index,
                     k, num_pts, num_splits, dim);

    }

    cycles_total = cycles_insertion + stop_tsc(start) / (double) num_runs;
    double perf = (1000.0 * flops_total / cycles_total) / 1000.0;

    free(input_points_ptr);
    free(pairwise_dist);
    free(dist_k_neighborhood_index);
    free(k_neighborhood_index);
    free(k_distance_index);

    return perf;

}