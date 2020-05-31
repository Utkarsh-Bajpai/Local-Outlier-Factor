//
// Created by pasca on 05.05.2020.
// FREE !!!
#include <math.h>
#include <stdlib.h>

#include "../../include/tsc_x86.h"
#include "../../include/tests.h"
#include "../../include/utils.h"
#include "../../include/lof_baseline.h"


#define CYCLES_REQUIRED 1e8
#define NUM_FUNCTIONS 6

typedef void (* my_fun)(int, int, const double*, const int*, double*);


/**
 * Change operation order for divisions
 */
void ComputeLocalReachabilityDensity_1(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                       const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr) {
    for (int i = 0; i < num_pts; i++) {
        double sum = 0;

        for (int j = 0; j < k; j++) {
            int neigh_point = neighborhood_index_table_ptr[i * k + j];


            sum += reachability_distances_indexed_ptr[i * num_pts + neigh_point];
        }
        lrd_score_table_ptr[i] = k / sum;
    }

}

/**
 * Outer by 4 w base
 */
void ComputeLocalReachabilityDensity_3(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                       const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr) {
    int i, j;
    for (i = 0; i + 4 < num_pts; i += 4) {

        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;


        int base_0 = i * num_pts;
        int base_1 = base_0 + num_pts;
        int base_2 = base_1 + num_pts;
        int base_3 = base_2 + num_pts;

        int np_base_0 = i * k;
        int np_base_1 = np_base_0 + k;
        int np_base_2 = np_base_1 + k;
        int np_base_3 = np_base_2 + k;

        for (j = 0; j < k; j++) {
            int np_0 = neighborhood_index_table_ptr[np_base_0 + j];
            int np_1 = neighborhood_index_table_ptr[np_base_1 + j];
            int np_2 = neighborhood_index_table_ptr[np_base_2 + j];
            int np_3 = neighborhood_index_table_ptr[np_base_3 + j];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0];
            sum_1 += reachability_distances_indexed_ptr[base_1 + np_1];
            sum_2 += reachability_distances_indexed_ptr[base_2 + np_2];
            sum_3 += reachability_distances_indexed_ptr[base_3 + np_3];
        }
        lrd_score_table_ptr[i] = k / sum_0;
        lrd_score_table_ptr[i + 1] = k / sum_1;
        lrd_score_table_ptr[i + 2] = k / sum_2;
        lrd_score_table_ptr[i + 3] = k / sum_3;
    }

    for (; i < num_pts; i++) {

        double sum = 0;
        int base = i * num_pts;
        int np_base_0 = i * k;

        for (j = 0; j < k; j++) {
            int neigh_point = neighborhood_index_table_ptr[np_base_0 + j];
            sum += reachability_distances_indexed_ptr[base + neigh_point];
        }
        lrd_score_table_ptr[i] = k / sum;
    }
}

/**
 * Outer by 4 w/o base
 */
void ComputeLocalReachabilityDensity_4(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                       const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr) {
    int i, j;
    for (i = 0; i + 4 < num_pts; i += 4) {

        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;


        int base_0 = i * num_pts;
        int base_1 = base_0 + num_pts;
        int base_2 = base_1 + num_pts;
        int base_3 = base_2 + num_pts;


        for (j = 0; j < k; j++) {
            int np_0 = neighborhood_index_table_ptr[i * k + j];
            int np_1 = neighborhood_index_table_ptr[(i + 1) * k + j];
            int np_2 = neighborhood_index_table_ptr[(i + 2) * k + j];
            int np_3 = neighborhood_index_table_ptr[(i + 3) * k + j];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0];
            sum_1 += reachability_distances_indexed_ptr[base_1 + np_1];
            sum_2 += reachability_distances_indexed_ptr[base_2 + np_2];
            sum_3 += reachability_distances_indexed_ptr[base_3 + np_3];
        }
        lrd_score_table_ptr[i] = k / sum_0;
        lrd_score_table_ptr[i + 1] = k / sum_1;
        lrd_score_table_ptr[i + 2] = k / sum_2;
        lrd_score_table_ptr[i + 3] = k / sum_3;
    }

    for (; i < num_pts; i++) {

        double sum = 0;
        int base = i * num_pts;
        int np_base_0 = i * k;

        for (j = 0; j < k; j++) {
            int neigh_point = neighborhood_index_table_ptr[np_base_0 + j];
            sum += reachability_distances_indexed_ptr[base + neigh_point];
        }
        lrd_score_table_ptr[i] = k / sum;
    }

}


/**
 * Inner by 4
 */
void ComputeLocalReachabilityDensity_5(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                       const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr) {

    int i, j = 0;
    for (i = 0; i < num_pts; i++) {

        double sum = 0;
        int base = i * num_pts;

        for (j = 0; j + 4 < k; j += 4) {
            int np_0 = neighborhood_index_table_ptr[i * k + j];
            sum += reachability_distances_indexed_ptr[base + np_0];
            int np_1 = neighborhood_index_table_ptr[i * k + j + 1];
            sum += reachability_distances_indexed_ptr[base + np_1];
            int np_2 = neighborhood_index_table_ptr[i * k + j + 2];
            sum += reachability_distances_indexed_ptr[base + np_2];
            int np_3 = neighborhood_index_table_ptr[i * k + j + 3];
            sum += reachability_distances_indexed_ptr[base + np_3];
        }

        for (; j < k; j += 2) {
            int np_0 = neighborhood_index_table_ptr[i * k + j];
            sum += reachability_distances_indexed_ptr[base + np_0];
            int np_1 = neighborhood_index_table_ptr[i * k + j + 1];
            sum += reachability_distances_indexed_ptr[base + np_1];
        }

        for (; j < k; j++) {
            int np_0 = neighborhood_index_table_ptr[i * k + j];
            sum += reachability_distances_indexed_ptr[base + np_0];
        }
        lrd_score_table_ptr[i] = k / sum;
    }
}

/**
 * Outer by 4 and inner by 4
 */
void ComputeLocalReachabilityDensity_6(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                       const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr) {
    int i, j;
    for (i = 0; i + 4 < num_pts; i += 4) {
        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;

        int base_0 = i * num_pts;
        int base_1 = base_0 + num_pts;
        int base_2 = base_1 + num_pts;
        int base_3 = base_2 + num_pts;

        int base_np_0 = i * k;
        int base_np_1 = (i + 1) * k;
        int base_np_2 = (i + 2) * k;
        int base_np_3 = (i + 3) * k;

        for (j = 0; j + 4 < k; j += 4) {
            int np_0_0 = neighborhood_index_table_ptr[base_np_0 + j];
            int np_1_0 = neighborhood_index_table_ptr[base_np_1 + j];
            int np_2_0 = neighborhood_index_table_ptr[base_np_2 + j];
            int np_3_0 = neighborhood_index_table_ptr[base_np_3 + j];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_0];
            sum_1 += reachability_distances_indexed_ptr[base_1 + np_1_0];
            sum_2 += reachability_distances_indexed_ptr[base_2 + np_2_0];
            sum_3 += reachability_distances_indexed_ptr[base_3 + np_3_0];

            int np_0_1 = neighborhood_index_table_ptr[base_np_0 + j + 1];
            int np_1_1 = neighborhood_index_table_ptr[base_np_1 + j + 1];
            int np_2_1 = neighborhood_index_table_ptr[base_np_2 + j + 1];
            int np_3_1 = neighborhood_index_table_ptr[base_np_3 + j + 1];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_1];
            sum_1 += reachability_distances_indexed_ptr[base_1 + np_1_1];
            sum_2 += reachability_distances_indexed_ptr[base_2 + np_2_1];
            sum_3 += reachability_distances_indexed_ptr[base_3 + np_3_1];

            int np_0_2 = neighborhood_index_table_ptr[base_np_0 + j + 2];
            int np_1_2 = neighborhood_index_table_ptr[base_np_1 + j + 2];
            int np_2_2 = neighborhood_index_table_ptr[base_np_2 + j + 2];
            int np_3_2 = neighborhood_index_table_ptr[base_np_3 + j + 2];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_2];
            sum_1 += reachability_distances_indexed_ptr[base_1 + np_1_2];
            sum_2 += reachability_distances_indexed_ptr[base_2 + np_2_2];
            sum_3 += reachability_distances_indexed_ptr[base_3 + np_3_2];

            int np_0_3 = neighborhood_index_table_ptr[base_np_0 + j + 3];
            int np_1_3 = neighborhood_index_table_ptr[base_np_1 + j + 3];
            int np_2_3 = neighborhood_index_table_ptr[base_np_2 + j + 3];
            int np_3_3 = neighborhood_index_table_ptr[base_np_3 + j + 3];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_3];
            sum_1 += reachability_distances_indexed_ptr[base_1 + np_1_3];
            sum_2 += reachability_distances_indexed_ptr[base_2 + np_2_3];
            sum_3 += reachability_distances_indexed_ptr[base_3 + np_3_3];
        }

        for (; j < k; j++) {
            int np_0_0 = neighborhood_index_table_ptr[base_np_0 + j];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_0];

            int np_1_0 = neighborhood_index_table_ptr[base_np_1 + j];
            sum_1 += reachability_distances_indexed_ptr[base_1 + np_1_0];

            int np_2_0 = neighborhood_index_table_ptr[base_np_2 + j];
            sum_2 += reachability_distances_indexed_ptr[base_2 + np_2_0];

            int np_3_0 = neighborhood_index_table_ptr[base_np_3 + j];
            sum_3 += reachability_distances_indexed_ptr[base_3 + np_3_0];
        }

        lrd_score_table_ptr[i] = k / sum_0;
        lrd_score_table_ptr[i + 1] = k / sum_1;
        lrd_score_table_ptr[i + 2] = k / sum_2;
        lrd_score_table_ptr[i + 3] = k / sum_3;
    }

    //complete the remaining
    for (; i < num_pts; i++) {

        double sum_0 = 0;
        int base_0 = i * num_pts;
        int base_np_0 = i * k;

        for (j = 0; j + 4 < k; j += 4) {
            int np_0_0 = neighborhood_index_table_ptr[base_np_0 + j];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_0];

            int np_0_1 = neighborhood_index_table_ptr[base_np_0 + j + 1];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_1];

            int np_0_2 = neighborhood_index_table_ptr[base_np_0 + j + 2];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_2];

            int np_0_3 = neighborhood_index_table_ptr[base_np_0 + j + 3];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_3];
        }

        for (; j < k; j++) {
            int np_0_0 = neighborhood_index_table_ptr[base_np_0 + j];
            sum_0 += reachability_distances_indexed_ptr[base_0 + np_0_0];
        }

        lrd_score_table_ptr[i] = k / sum_0;
    }
}


int lrdensity_driver_unroll(int num_pts, int k, int dim, int num_reps) {


    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &ComputeLocalReachabilityDensity;
    fun_array[1] = &ComputeLocalReachabilityDensity_1;
    fun_array[2] = &ComputeLocalReachabilityDensity_3;
    fun_array[3] = &ComputeLocalReachabilityDensity_4;
    fun_array[4] = &ComputeLocalReachabilityDensity_5;
    fun_array[5] = &ComputeLocalReachabilityDensity_6;
    char* fun_names[NUM_FUNCTIONS] = {"V0", "V1", "V3", "V4", "V5", "V6"};

    myInt64 start, end;
    double cycles1;
    double multiplier = 1;
    double numRuns = 10;

    // INITIALIZE RANDOM INPUT
    int* neighborhoodIndexTablePtr = XmallocMatrixIntRandom(num_pts, k, num_pts);
    double* reachabilityDistancesIndexedPtr = XmallocVectorDoubleRandom(num_pts * num_pts);

    double* lrdScoreTablePtr = XmallocVectorDouble(num_pts);
    double* lrdScoreTablePtrTrue = XmallocVectorDouble(num_pts);


    for (int fun_index = 0; fun_index < NUM_FUNCTIONS;
         fun_index++) {

        // VERIFICATION : -------------------------------------------------------------------------
        ComputeLocalReachabilityDensity(k, num_pts, reachabilityDistancesIndexedPtr, neighborhoodIndexTablePtr,
                                        lrdScoreTablePtrTrue);
        (*fun_array[fun_index])(k, num_pts, reachabilityDistancesIndexedPtr, neighborhoodIndexTablePtr,
                                lrdScoreTablePtr);
        int ver = test_double_arrays(num_pts, 1e-3, lrdScoreTablePtr, lrdScoreTablePtrTrue);

        if (ver != 1) {
            printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
            exit(-1);
        }

        // Warm-up phase: we determine a number of executions that allows
        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[fun_index])(k, num_pts, reachabilityDistancesIndexedPtr, neighborhoodIndexTablePtr,
                                        lrdScoreTablePtr);
            }
            end = stop_tsc(start);

            cycles1 = (double) end;
            multiplier = (CYCLES_REQUIRED) / (cycles1);

        } while (multiplier > 2);


        double totalCycles = 0;
        double* cyclesPtr = XmallocVectorDouble(num_reps);

        for (size_t j = 0; j < num_reps; j++) {

            start = start_tsc();
            for (size_t i = 0; i < numRuns; ++i) {
                (*fun_array[fun_index])(k, num_pts, reachabilityDistancesIndexedPtr, neighborhoodIndexTablePtr,
                                        lrdScoreTablePtr);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;

            totalCycles += cycles1;
        }

        qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[(int) num_reps / 2 + 1];

        double flops = num_pts * (2 + k);
        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        free(cyclesPtr);
        printf("%s n:%d cycles:%lf perf:%lf \n", fun_names[fun_index], num_pts, cycles, perf);
    }
    printf("-------------\n");
    free(neighborhoodIndexTablePtr);
    free(reachabilityDistancesIndexedPtr);
    free(lrdScoreTablePtr);
    free(lrdScoreTablePtrTrue);

    return 0;
}
