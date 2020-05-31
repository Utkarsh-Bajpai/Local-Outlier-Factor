//
// Created by pasca on 05.05.2020.
//


#include "math.h"
#include "stdlib.h"

#include "../../include/tests.h"
#include "../../include/tsc_x86.h"
#include "../../include/utils.h"
#include "../../include/lof_baseline.h"
#include "../../include/lof_baseline.h"

#define CYCLES_REQUIRED 1e8
#define NUM_FUNCTIONS_RD_AVX 3
// num of functions + 1 for baseline

// ------------------------------------------------------------------------------------- < AVX improvements >



// ------------------------------------------------------------------------------------- < /AVX improvements >

/*
int lrdistance_driver(int num_pts, int k, int num_reps) {

    my_lrd_fnc* fun_array = (my_lrd_fnc*) calloc(NUM_FUNCTIONS_RD_AVX, sizeof(my_lrd_fnc));
    fun_array[0] = &ComputeReachabilityDistanceAll;
    
    //fun_array[1] = &ComputeReachabilityDistanceAll_1;
    //fun_array[2] = &ComputeReachabilityDistanceAll_2;
    //fun_array[2] = &ComputeLocalReachabilityDensity_3;
    //fun_array[3] = &ComputeLocalReachabilityDensity_4;
    //fun_array[4] = &ComputeLocalReachabilityDensity_6;
    
    char* fun_names[NUM_FUNCTIONS_RD_AVX] = {"V0", "V1", "V2", "V4", "V6"};

    myInt64 start, end;
    double cycles1;
    double multiplier = 1;
    double numRuns = 10;

    // INITIALIZE RANDOM INPUT
    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDoubleRandom(num_pts);

    double* reachability_distances_indexed_ptr = XmallocVectorDouble(num_pts * num_pts);
    double* reachability_distances_indexed_ptr_true = XmallocVectorDouble(num_pts * num_pts);


    for (int fun_index = 0; fun_index < NUM_FUNCTIONS_RD_AVX;
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
*/
