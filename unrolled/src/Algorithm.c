#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
//#include <tests.h>

#include "../../include/tests.h"
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "../../include/file_utils.h"
#include "../../include/lof_baseline.h"


#include "../include/metrics.h"
#include "../include/Algorithm.h"
#include "../include/ComputeKDistanceAll.h"
#include "../include/ComputeKDistanceNeighborhood.h"
#include "../include/ComputeReachabilityDistance.h"
#include "../include/ComputeLocalReachabilityDensityMerged.h"
#include "../include/ComputeLocalOutlierFactor.h"
#include "../include/KNN.h"
#include "../include/ComputeLocalReachabilityDensity.h"

typedef int (* my_fun)(int, int, int, double*, double*, double*, double*, double*, double*, int*, int*);


#define CYCLES_REQUIRED 2e8
#define NUM_FUNCTIONS 2

int algo_base(int num_pts, int k, int dim, double* input_points_ptr, double* distances_indexed_ptr,
              double* k_distances_indexed_ptr, double* reachability_distances_indexed_ptr,
              double* lrd_score_table_ptr, double* lof_score_table_ptr,
              int* neighborhood_index_table_ptr, int* hash_index) {

    ComputePairwiseDistances(dim, num_pts, input_points_ptr, UnrolledEuclideanDistance, distances_indexed_ptr);
    int f1 = num_pts * (num_pts - 1) * (4 * dim + 1) / 2;


    ComputeKDistanceAll(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    int f2 = num_pts * (num_pts - 1) * (1 + log(num_pts - 1));

    ComputeKDistanceNeighborhoodAll(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr,
                                    neighborhood_index_table_ptr);
    int f3 = num_pts * (num_pts - 1);

    ComputeReachabilityDistanceAll(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                   reachability_distances_indexed_ptr);
    int f4 = num_pts * (num_pts - 1);

    ComputeLocalReachabilityDensity(k, num_pts, reachability_distances_indexed_ptr, neighborhood_index_table_ptr,
                                    lrd_score_table_ptr);
    int f5 = num_pts * (k + 2);

    ComputeLocalOutlierFactor(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);

    int f6 = num_pts * (k + 2);

    return f1 + f2 + f3 + f4 + f5 + f6;

}

int algo_1(int num_pts, int k, int dim, double* input_points_ptr, double* distances_indexed_ptr,
           double* k_distances_indexed_ptr, double* reachability_distances_indexed_ptr,
           double* lrd_score_table_ptr, double* lof_score_table_ptr,
           int* neighborhood_index_table_ptr, int* hash_index) {

    // Step 2: compute pairwise distance between points
    KNN_hash(num_pts, k, dim, input_points_ptr, neighborhood_index_table_ptr, hash_index, distances_indexed_ptr,
             k_distances_indexed_ptr);

    int f1 = num_pts * (num_pts - 1) / 2 * (4 * dim + 1 + 12) + num_pts * (num_pts - 1) * (1 + log(num_pts - 1)) +
             num_pts * (num_pts - 1);


    // Step 5: compute pairwise Reachability distance
    ComputeLocalReachabilityDensityMergedHashed(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                                neighborhood_index_table_ptr, hash_index, lrd_score_table_ptr);
    int f2 = num_pts * (2 * k + 1);



    // Step 7: compute lof
    ComputeLocalOutlierFactor_4(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr,
                                lof_score_table_ptr);

    int f3 = num_pts * (k + 2);
    return f1 + f2 + f3;
}

#define NUM_FUNCTIONS 2

void algorithm_driver(int num_pts, int k, int dim, enum Mode mode, int num_reps) {

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &algo_base;
    fun_array[1] = &algo_1;

    char buffer[50];
    sprintf(buffer, "python3 python_scripts/generator.py 4 %d %d %d", num_pts, k, dim);
    system(buffer);

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
    int* hash_index = XmallocMatrixInt(num_pts, num_pts);

    int i, j;
    myInt64 start, end;
    double cycles1;


    int flops;

    for (int fun_index = 0; fun_index < NUM_FUNCTIONS; fun_index++) {

        double multiplier = 1;
        double numRuns = 10;

        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (i = 0; i < numRuns; i++) {
                flops = (*fun_array[fun_index])(num_pts, k, dim, input_points_ptr, distances_indexed_ptr,
                                                k_distances_indexed_ptr,
                                                reachability_distances_indexed_ptr, lrd_score_table_ptr,
                                                lof_score_table_ptr,
                                                neighborhood_index_table_ptr, hash_index);
            }
            end = stop_tsc(start);

            cycles1 = (double) end;
            multiplier = (CYCLES_REQUIRED) / (cycles1);

        } while (multiplier > 2);

        double* cyclesPtr = XmallocVectorDouble(num_reps);

        CleanTheCache(500);

        for (j = 0; j < num_reps; j++) {
            start = start_tsc();
            for (i = 0; i < numRuns; ++i) {
                flops = (*fun_array[fun_index])(num_pts, k, dim, input_points_ptr, distances_indexed_ptr,
                                                k_distances_indexed_ptr,
                                                reachability_distances_indexed_ptr, lrd_score_table_ptr,
                                                lof_score_table_ptr,
                                                neighborhood_index_table_ptr, hash_index);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;
        }

        qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[((int) num_reps / 2) + 1];
        free(cyclesPtr);
        double perf = round((1000.0 * flops) / cycles) / 1000.0;

        printf("Fun %d Cycles %lf, Performance %lf\n", fun_index, cycles, perf);

        double dif = 0;
        for (i = 0; i < num_pts; i++) {
            dif += fabs(lof_score_table_ptr[i] - lof_true_ptr[i]);
        }

        switch (mode) {
            case BASIC:
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

                int test_neigh_ind_result = test_neigh_ind(num_pts, k, neighborhood_index_table_ptr,
                                                           neigh_ind_true_ptr);
                assert(test_neigh_ind_result == 1);

                int test_lrd_result = test_double_arrays(num_pts, 1e-5, lrd_score_table_ptr, lrd_true_ptr);
                assert(test_lrd_result == 1);

                assert(fabs(dif) < 1e-3);
            }
                break;
            case BENCHMARK: {

                //assert(fabs(dif) < 1e-3);
                break;
            }
        }
    }
    printf("----------------------\n");

}





