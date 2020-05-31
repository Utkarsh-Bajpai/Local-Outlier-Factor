//
// Created by pasca on 05.05.2020.
//

#include "stdlib.h"
#include "math.h"

#include "../../include/tsc_x86.h"
#include "../../include/tests.h"
#include "../../include/utils.h"
#include "../../include/lof_baseline.h"

#define CYCLES_REQUIRED 1e8
#define NUM_FUNCTIONS 3

typedef void (* my_fun)(int, int, const double*, const double*, double*);

/**
 *
 * Split the inner loop
 * *
 */

void ComputeReachabilityDistanceAll_1(int k, int num_pts, const double* distances_indexed_ptr,
                                      const double* k_distances_indexed_ptr,
                                      double* reachability_distances_indexed_ptr) {

    int idx_from, idx_to;
    for (idx_from = 0; idx_from < num_pts; ++idx_from) {
        for (idx_to = 0; idx_to < idx_from; ++idx_to) {
            int idxDist = num_pts * idx_to + idx_from;

            int idxReach = idx_from * num_pts + idx_to;

            double kdistO = k_distances_indexed_ptr[idx_to];
            double distPo = distances_indexed_ptr[idxDist];

            reachability_distances_indexed_ptr[idxReach] = kdistO >= distPo ? kdistO : distPo;
        }

        for (idx_to = idx_from + 1; idx_to < num_pts; idx_to += 1) {
            int idxDist1 = num_pts * idx_from + idx_to;

            double kdistO_1 = k_distances_indexed_ptr[idx_to];
            double distPo_1 = distances_indexed_ptr[idxDist1];

            reachability_distances_indexed_ptr[idxDist1] = kdistO_1 >= distPo_1 ? kdistO_1 : distPo_1;
        }
    }
}

/**
 *
 * Unroll inside
 * *
 */

void ComputeReachabilityDistanceAll_2(int k, int num_pts, const double* distances_indexed_ptr,
                                      const double* k_distances_indexed_ptr,
                                      double* reachability_distances_indexed_ptr) {

    int idx_from, idx_to;
    for (idx_from = 0; idx_from < num_pts; ++idx_from) {
        for (idx_to = 0; idx_to < idx_from; ++idx_to) {
            int idxDist = num_pts * idx_to + idx_from;

            int idxReach = idx_from * num_pts + idx_to;

            double kdistO = k_distances_indexed_ptr[idx_to];
            double distPo = distances_indexed_ptr[idxDist];

            reachability_distances_indexed_ptr[idxReach] = kdistO >= distPo ? kdistO : distPo;
        }

        for (idx_to = idx_from + 1; idx_to + 4 < num_pts; idx_to += 4) {
            int idxDist_0 = num_pts * idx_from + idx_to;
            int idxDist_1 = num_pts * idx_from + idx_to + 1;
            int idxDist_2 = num_pts * idx_from + idx_to + 2;
            int idxDist_3 = num_pts * idx_from + idx_to + 3;

//
            double kdistO_0 = k_distances_indexed_ptr[idx_to];
            double distPo_0 = distances_indexed_ptr[idxDist_0];
            double kdistO_1 = k_distances_indexed_ptr[idx_to + 1];
            double distPo_1 = distances_indexed_ptr[idxDist_1];

            double kdistO_2 = k_distances_indexed_ptr[idx_to + 2];
            double distPo_2 = distances_indexed_ptr[idxDist_2];
            double kdistO_3 = k_distances_indexed_ptr[idx_to + 3];
            double distPo_3 = distances_indexed_ptr[idxDist_3];

            reachability_distances_indexed_ptr[idxDist_0] = kdistO_0 >= distPo_0 ? kdistO_0 : distPo_0;
            reachability_distances_indexed_ptr[idxDist_1] = kdistO_1 >= distPo_1 ? kdistO_1 : distPo_1;
            reachability_distances_indexed_ptr[idxDist_2] = kdistO_2 >= distPo_2 ? kdistO_2 : distPo_2;
            reachability_distances_indexed_ptr[idxDist_3] = kdistO_3 >= distPo_3 ? kdistO_3 : distPo_3;
        }

        for (; idx_to < num_pts; idx_to += 1) {
            int idxDist1 = num_pts * idx_from + idx_to;

            double kdistO_1 = k_distances_indexed_ptr[idx_to];
            double distPo_1 = distances_indexed_ptr[idxDist1];

            reachability_distances_indexed_ptr[idxDist1] = kdistO_1 >= distPo_1 ? kdistO_1 : distPo_1;
        }
    }
}

int lrdistance_driver_unrolled(int num_pts, int k, int num_reps) {

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &ComputeReachabilityDistanceAll;
    fun_array[1] = &ComputeReachabilityDistanceAll_1;
    fun_array[2] = &ComputeReachabilityDistanceAll_2;
//    fun_array[2] = &ComputeLocalReachabilityDensity_3;
//    fun_array[3] = &ComputeLocalReachabilityDensity_4;
//    fun_array[4] = &ComputeLocalReachabilityDensity_6;
    char* fun_names[NUM_FUNCTIONS] = {"V0", "V1", "V2", "V4", "V6"};

    myInt64 start, end;
    double cycles1;
    double multiplier = 1;
    double numRuns = 10;

    // INITIALIZE RANDOM INPUT
    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDoubleRandom(num_pts);

    double* reachability_distances_indexed_ptr = XmallocVectorDouble(num_pts * num_pts);
    double* reachability_distances_indexed_ptr_true = XmallocVectorDouble(num_pts * num_pts);


    for (int fun_index = 0; fun_index < NUM_FUNCTIONS;
         fun_index++) {
        ComputeReachabilityDistanceAll(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                       reachability_distances_indexed_ptr_true);
        // VERIFICATION : -------------------------------------------------------------------------

        (*fun_array[fun_index])(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                reachability_distances_indexed_ptr);
        int ver = test_double_arrays(num_pts, 1e-3, reachability_distances_indexed_ptr,
                                     reachability_distances_indexed_ptr_true);

        if (ver != 1) {
            printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
            exit(-1);
        }

        // Warm-up phase: we determine a number of executions that allows
        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[fun_index])(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                        reachability_distances_indexed_ptr);
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
                (*fun_array[fun_index])(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                        reachability_distances_indexed_ptr);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;

            totalCycles += cycles1;
        }

        qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[(int) num_reps / 2 + 1];

        double flops = num_pts * (num_pts - 1);
        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        free(cyclesPtr);
        printf("%s n:%d cycles:%lf perf:%lf \n", fun_names[fun_index], num_pts, cycles, perf);
    }

    printf("-------------\n");
    free(reachability_distances_indexed_ptr);
    free(reachability_distances_indexed_ptr_true);
    free(distances_indexed_ptr);
    free(k_distances_indexed_ptr);

    return 0;
}
