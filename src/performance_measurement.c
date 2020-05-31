//
// Created by Sycheva  Anastasia on 16.05.20.
//
/**
 * The goal is to collect most of the functions / variables are reused
 * to evaluate the performance improvements of functions
 *
 * drivers for each function / improvement can be stored in the corresponding source file
 *
 * NB: a lot of code redunduncy due to the absence of templates :(
 *
 * TODO:
 * 1. need to discuss how to incorporate code from lrd_driver here and make sure there is more reuse
 * 2. figure out file paths
 * 3. are there some types of templates for C ? These functions are super similiar
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/tsc_x86.h"
#include "../include/utils.h"
#include "../include/tests.h"
#include "../include/metrics.h"
#include "../include/lof_baseline.h"
#include "../include/performance_measurement.h"

#define NUM_FUNCTIONS 1

// QUESTION: SHOULD I DO ALL THE SAME ?
int LOF_NUM_RUNS = 5000;
int LOF_NUM_SMALL_RUNS = 10;
int LRDM_NUM_RUNS = 5000;
int LRDM_NUM_SMALL_RUNS = 10;
int METRICS_NUM_RUNS = 1000;        // repeat this that many times
int METRICS_NUM_SMALL_RUNS = 10;    // average over 10

// Metrics ----------------------------------------------------------------------------------------------->

double performance_measure_hot_for_metrics(int dim, double (* fct)(const double*, const double*, int),
                                           double (* fct_correct)(const double*, const double*, int),
                                           int to_verify, int verobse, int compare_with_baseline) {
    /**
     * new function for measuring the perfomance of the function (hot cache)
     * data is used to create corresponding box plots
     *
     * :param fct         : function we currently consider
     * :param fct_correct : original correct implementation w.r.t which we verify
     * :param to_verify   : verify the fct against fct_correct
     * :param verbose     : print intermediate output
     */

    myInt64 start;
    double cyclesOriginal, cyclesNew;
    double flopsPerCycleOrig, flopsPerCycleNew;

    // compute the expected number of flops
    double flopsTotal = ComputeFlopsCosineSimilarity(dim);      // computer the

    // INITIALIZE RANDOM INPUT
    double* v1_ptr = XmallocVectorDoubleFillRandom(dim, 10);
    double* v2_ptr = XmallocVectorDoubleFillRandom(dim, 10);

    // VERIFICATION : -------------------------------------------------------------------------

    if (to_verify == 1) {

        double orig_res = fct_correct(v1_ptr, v2_ptr, dim);
        double new_res = fct(v1_ptr, v2_ptr, dim);

        if (fabs(orig_res - new_res) > 0.0001) {
            printf("Results from the functions mismatch by %f", fabs(orig_res - new_res));
            return 0;
        }

    }

    // ------------------------------------- Baseline

    if (compare_with_baseline == 1) { // redundand ?

        start = start_tsc();
        for (int i = 0; i < METRICS_NUM_SMALL_RUNS; ++i) {
            fct_correct(v1_ptr, v2_ptr, dim);
        }
        cyclesOriginal = stop_tsc(start) / (double) METRICS_NUM_SMALL_RUNS;
        flopsPerCycleOrig = flopsTotal / cyclesOriginal;

    }

    //-------------------------------------- Better Function
    start = start_tsc();
    for (int i = 0; i < METRICS_NUM_SMALL_RUNS; ++i) {
        fct(v1_ptr, v2_ptr, dim);
    }
    cyclesNew = stop_tsc(start) / (double) METRICS_NUM_SMALL_RUNS;
    flopsPerCycleNew = flopsTotal / cyclesNew;

    if (verobse == 1) {
        printf("Total number of flops: %f\n", flopsTotal);
        printf("%f (cycles) %f (flops / cycle)\n", cyclesNew, flopsPerCycleNew);


        if (compare_with_baseline == 1) {
            printf("%f (cycles) %f (flops / cycle)\n", cyclesOriginal, flopsPerCycleOrig);
            printf("Speedup = %f\n", flopsPerCycleNew / flopsPerCycleOrig);
        }
    }   // if( verobse == 1 )

    return flopsPerCycleNew;
} // performance_measure



void performance_plot_metrics_to_file(int max_dim, int step,
                                      const char* name, const char* mode,
                                      double (* fct)(const double*, const double*, int),
                                      double (* fct_correct)(const double*, const double*, int)) {
    /**
     * for metrics functions measure performance
     * by varying number of dimensions
     *
     * :param max_dim        largest dimensionality considered on the plots
     * :param step           step size for iteration
     * :param name           name of the function / identifier that will be used in the plots
     * :param mode           expected one of: "w" - override, "a" - append
     * :param fct            function to be measured
     * :param fct_correct    baseline implementation to compare with
     *
     */

    FILE* results_file = fopen("measurements/metrics_results.txt", mode);

    unsigned int dim, rep;
    double res;

    int success = fputs(name, results_file);
    fprintf(results_file, "\n");
    for (dim = 2; dim < max_dim; dim += step) {

        fprintf(results_file, "%d, ", dim);
        for (rep = 0; rep < METRICS_NUM_RUNS; rep++) {
            res = performance_measure_hot_for_metrics(dim, fct, fct_correct, 0, 0, 0);
            fprintf(results_file, "%f, ", res);
        }
        fprintf(results_file, "\n");

    } // iterate over dimensions
    fclose(results_file);
}

// Compute Pairwise Distances ---------------------------------------------------------------------------->



double performance_measure_fast_for_mini_mmm(int Bp, int Bd, int num_reps, int num_pts, int dim,
                                             void (*fnc)(int, int, int, int, const double*, double*)){
    /**
     *@param fnc: function to compute mini mmm
     *            (expected memory/../ComputePairwiseDistance/mini_mmm_lower_triangular)
     *       mini_mmm_lower_triangular(int num_pts, int dim, int Bp, int Bd, const double* input_points_ptr, double* result )
     */
    myInt64 start, end;
    double cycles1;
    double multiplier = 1;
    double numRuns = 10;
    // COMPUTE NUMBER OF FLOPS REQUIRED
    double flopsTotal = 0.5 * num_pts * num_pts * dim;    // CHECK !!!

    // INITIALIZE RANDOM INPUT
    double* results = XmallocMatrixDouble(num_pts, num_pts);
    double* input_points = XmallocMatrixDoubleRandom(num_pts, num_pts);

    // Warm-up phase: we determine a number of executions that allows
    do {
        numRuns = numRuns * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < numRuns; i++) {
            fnc( num_pts, dim, Bp, Bd, input_points, results);
        }
        end = stop_tsc(start);

        cycles1 = (double) end;
        multiplier = (CYCLES_REQUIRED) / (cycles1);

    } while (multiplier > 2);

    // START MEASUREMENTS

    double* cyclesPtr = XmallocVectorDouble(num_reps);
    for (size_t j = 0; j < num_reps; j++) {

        start = start_tsc();
        for (size_t i = 0; i < numRuns; ++i) {
            fnc( num_pts, dim, Bp, Bd, input_points, results);;
        }
        end = stop_tsc(start);

        cycles1 = ((double) end) / numRuns;
        cyclesPtr[j] = cycles1;
    }

    qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
    double cycles = cyclesPtr[(int) num_reps / 2 + 1];
    double perf = round((1000.0 * flopsTotal) / cycles) / 1000.0;

    // printf("n: %d k: %d repetitions: %d cycles: %lf perf: %lf \n", num_pts, k, num_reps, cycles, perf);

    free(cyclesPtr);
    free(results);
    free(input_points);

    //printf("%d, %d, %d, %d, %lf", num_pts, dim, Bd, Bp);

    return perf;

}

// Definition 3 ------------------------------------------------------------------------------------------>

// Definition 4 ------------------------------------------------------------------------------------------>

// Definition 5 ------------------------------------------------------------------------------------------>

// Definition 6 ------------------------------------------------------------------------------------------>

double performance_measure_fast_for_lrd(int k, int num_pts, int num_reps,
                                        my_lrdm_fnc lrdf_fct, flops_lrdm_fnc flops_fct){
    /**
     * wrap around / modification of the function lrdistance_driver that performs the measurements
     * for only one function and returns the median. Can be useful for fast testing which modification is better
     * before moving to producing boxplot measurements which take some time
     *
     * @param lrdf_fct : function to be tested
     * @param flops_fct : function that computes the number of flops depending on the input size
     *                    useful if we experiment with different counting systems
     * @param num_reps : How many times should measurements be repeated
     *                   recomment 1000 ?
     * @param to_verify: compare with baseline implementation ?
     *
     */
    myInt64 start, end;
    double cycles1;
    double multiplier = 1;
    double numRuns = 10;
    // COMPUTE NUMBER OF FLOPS REQUIRED
    double flopsTotal = flops_fct( k, num_pts );

    // INITIALIZE RANDOM INPUT
    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDoubleRandom(num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixIntRandom(num_pts, k, num_pts);
    double* lrd_score_table_ptr = XmallocVectorDoubleRandom(num_pts);
    double* lrd_score_table_ptr_true = XmallocVectorDoubleRandom(num_pts);

    // VERIFICATION:
    ComputeLocalReachabilityDensityMerged(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr_true);
    lrdf_fct(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr);
    int ver = test_double_arrays( num_pts, 1e-3, lrd_score_table_ptr, lrd_score_table_ptr_true );

    if (ver == 0) {
        printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
    }
    // Warm-up phase: we determine a number of executions that allows
    do {
        numRuns = numRuns * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < numRuns; i++) {
            lrdf_fct(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr);
        }
        end = stop_tsc(start);

        cycles1 = (double) end;
        multiplier = (CYCLES_REQUIRED) / (cycles1);

    } while (multiplier > 2);

    // START MEASUREMENTS

    double* cyclesPtr = XmallocVectorDouble(num_reps);
    for (size_t j = 0; j < num_reps; j++) {

        start = start_tsc();
        for (size_t i = 0; i < numRuns; ++i) {
            lrdf_fct(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr);
        }
        end = stop_tsc(start);

        cycles1 = ((double) end) / numRuns;
        cyclesPtr[j] = cycles1;
    }

    qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
    double cycles = cyclesPtr[(int) num_reps / 2 + 1];
    double perf = round((1000.0 * flopsTotal) / cycles) / 1000.0;

    printf("n: %d k: %d repetitions: %d cycles: %lf perf: %lf \n", num_pts, k, num_reps, cycles, perf);

    free(cyclesPtr);
    free(lrd_score_table_ptr);
    free(lrd_score_table_ptr_true);
    free(distances_indexed_ptr);
    free(k_distances_indexed_ptr);
    free(neighborhood_index_table_ptr);

    return perf;

}

double performance_measure_hot_for_lrd(int k, int num_pts, my_lrdm_fnc fct,
                                       int to_verify, int verobse, int compare_with_baseline) {
    myInt64 start, end;
    double cyclesOriginal, cyclesNew;
    double flopsPerCycleOrig, flopsPerCycleNew;
    double flopsTotal = ComputeFlopsReachabilityDensityMerged( k, num_pts );

    // INITIALIZE RANDOM INPUT
    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDoubleRandom(num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixIntRandom(num_pts, k, num_pts);

    double* lrd_score_table_ptr = XmallocVectorDoubleRandom(num_pts);
    double* lrd_score_table_ptr_true = XmallocVectorDoubleRandom(num_pts);

    // VERIFICATION : -------------------------------------------------------------------------

    if (to_verify == 1) {

        ComputeLocalReachabilityDensityMerged(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr_true);
        fct(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr);
        int ver = test_double_arrays(num_pts, 1e-3, lrd_score_table_ptr_true,  lrd_score_table_ptr);

        if (ver == 0) {
            printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
        }

    }

    // ------------------------------------- Baseline

    if (compare_with_baseline == 1) { // redundand ?

        start = start_tsc();
        for (int i = 0; i < LRDM_NUM_SMALL_RUNS; ++i) {
            ComputeLocalReachabilityDensityMerged(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr_true);
        }
        cyclesOriginal = stop_tsc(start) / (double) LRDM_NUM_SMALL_RUNS;
        flopsPerCycleOrig = flopsTotal / cyclesOriginal;

    }

    // Better versions --------------------------------------------------------------------------
    // warm up!
    for (int i = 0; i < LRDM_NUM_SMALL_RUNS; ++i) {
        fct(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr);
    }

    start = start_tsc();
    for (int i = 0; i < LRDM_NUM_SMALL_RUNS; ++i) {
        fct(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr, lrd_score_table_ptr);
    }
    cyclesNew = stop_tsc(start) / (double) LRDM_NUM_SMALL_RUNS;
    flopsPerCycleNew = flopsTotal / cyclesNew;

    if (verobse == 1) {

        printf("Total number of flops: %f\n", flopsTotal);
        printf("%f (cycles) %f (flops / cycle)\n", cyclesNew, flopsPerCycleNew);

        if (compare_with_baseline == 1) {
            printf("%f (cycles) %f (flops / cycle)\n", cyclesOriginal, flopsPerCycleOrig);
            printf("Speedup = %f\n", flopsPerCycleNew / flopsPerCycleOrig);
        }

    }   // if( verobse == 1 )

    // Free memory
    free(distances_indexed_ptr);
    free(k_distances_indexed_ptr);
    free(neighborhood_index_table_ptr);
    free(lrd_score_table_ptr_true);
    free(lrd_score_table_ptr);

    return flopsPerCycleNew;
} // performance_measure



void performance_plot_lrdm_to_file_k(int num_pts_ref, const char* file_name,
                                    const char* function_name, const char* mode,
                                     my_lrdm_fnc fct) {
    /**
     * for metrics functions measure performance
     * by varying number of dimensions
     *
     * :param max_num_pts    maximum input dimension
     * :param step           step size for iteration
     * :param name           name of the function / identifier that will be used in the plots.
     *                       has to be in double quotes !
     * :param mode           expected one of: "w" - override, "a" - append
     * :param fct            function to be measured
     * :param fct_correct    baseline implementation to compare with
     *
     * TODO: 1. add filename as input ?
     *       2. add k to the file
     *       3. add grid to file
     */

    // my local computer
    //FILE* results_file = fopen("/Users/sychevaa/Desktop/MyStudies/ETH/FS2020/ASL/old_team025/measurements/lof_results_k.txt", mode);
    //FILE* results_file = fopen("measurements/lof_results_k.txt", mode);
    FILE* results_file = fopen( file_name , mode);
    // SHOULD BE LESS THAN NUM PTS !!! - add assertion
    int k_grid[6] = {4, 10, 100, 250, 500, 1000};

    if (results_file == NULL) {
        fprintf(stderr, "Error at opening %s", results_file);
        printf("Check if the file was generated or the path in open_with_error_check\n");
    }

    unsigned int k, rep;
    double res;

    fprintf(results_file, "%s\n", function_name);
    fprintf(results_file, "%d\n", num_pts_ref);
    fprintf(results_file, "\n");

    //for (num_pts = 4; num_pts < max_num_pts; num_pts += step) {
    for (int i = 0; i < 6; i++){

        k = k_grid[i];
        if( k <= num_pts_ref ){

            fprintf(results_file, "%d, ", k);
            for (rep = 0; rep < LOF_NUM_RUNS; rep++) {
                res = performance_measure_hot_for_lrd( k, num_pts_ref, fct, 0, 0, 0);
                fprintf(results_file, "%f, ", res);
            }
            fprintf(results_file, "\n");

        }

    } // iterate over dimensions
    fclose(results_file);
}   //  performance_plot_lof_to_file

void performance_plot_lrdm_to_file_num_pts(int k_ref, const char* file_name,
                                           const char* function_name, const char* mode,
                                           my_lrdm_fnc fct ) {
    /**
     * for metrics functions measure performance
     * by varying number of dimensions
     *
     * :param max_num_pts    maximum input dimension
     * :param step           step size for iteration
     * :param name           name of the function / identifier that will be used in the plots.
     *                       has to be in double quotes !
     * :param mode           expected one of: "w" - override, "a" - append
     * :param fct            function to be measured
     * :param fct_correct    baseline implementation to compare with
     *
     * TODO:  1. add grid as input ?
     */

    // my local computer
    //FILE* results_file = fopen("/Users/sychevaa/Desktop/MyStudies/ETH/FS2020/ASL/old_team025/measurements/lof_results_num_pts.txt", mode);
    int num_pts_grid[13] = {10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    //FILE* results_file = fopen("measurements/lof_results_num_pts.txt", mode);
    FILE* results_file = fopen( file_name , mode);

    if (results_file == NULL) {
        fprintf(stderr, "Error at opening %s", results_file);
        printf("Check if the file was generated or the path in open_with_error_check\n");
    }

    unsigned int num_pts, rep;
    double res;

    fprintf(results_file, "%s\n", function_name);
    fprintf(results_file, "%d\n", k_ref);
    fprintf(results_file, "\n");

    //for (num_pts = 4; num_pts < max_num_pts; num_pts += step)
    for (int i = 2; i < 13; i++){

        num_pts = num_pts_grid[i];
        if( num_pts >= k_ref ) {    // number of neighbrs can't be lower than the number of points
            fprintf(results_file, "%d, ", num_pts);
            for (rep = 0; rep < LOF_NUM_RUNS; rep++) {
                res = performance_measure_hot_for_lrd(k_ref, num_pts, fct, 0, 0, 0);
                fprintf(results_file, "%f, ", res);
            }
        }
        fprintf(results_file, "\n");

    } // iterate over dimensions
    fclose(results_file);
}   //  performance_plot_lof_to_file


// Definition 7 -------------------------------------------------------------------------------->

double performance_measure_hot_for_lof(int k, int num_pts,
                                       my_lof_fnc fct, my_lof_fnc fct_original,
                                       int to_verify, int verobse, int compare_with_baseline) {
    myInt64 start;
    double cyclesOriginal, cyclesNew;
    double flopsPerCycleOrig, flopsPerCycleNew;
    double flopsTotal = ComputeFlopsLocalOutlierFactor(k, num_pts);

    // INITIALIZE RANDOM INPUT
    double* lrd_score_table_ptr = XmallocVectorDoubleRandom(num_pts); // Random
    int* neighborhood_index_table_ptr = XmallocMatrixIntRandom(num_pts, k, num_pts);
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);
    double* lof_score_table_ptr_true = XmallocVectorDouble(num_pts);

    // VERIFICATION : -------------------------------------------------------------------------

    if (to_verify == 1) {

        fct_original(k, num_pts, lrd_score_table_ptr,  neighborhood_index_table_ptr, lof_score_table_ptr_true);
        fct(k, num_pts, lrd_score_table_ptr,  neighborhood_index_table_ptr, lof_score_table_ptr);
        int ver = test_double_arrays(num_pts, 1e-3, lof_score_table_ptr, lof_score_table_ptr_true);

        if (ver == 0) {
            printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
        }
    }

    // ------------------------------------- Baseline

    if (compare_with_baseline == 1) { // redundand ?

        start = start_tsc();
        for (int i = 0; i < LOF_NUM_SMALL_RUNS; ++i) {
            fct_original(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
        }
        cyclesOriginal = stop_tsc(start) / (double) LOF_NUM_SMALL_RUNS;
        flopsPerCycleOrig = flopsTotal / cyclesOriginal;

    }

    // Better versions --------------------------------------------------------------------------
    // warm up!
    for (int i = 0; i < LOF_NUM_SMALL_RUNS; ++i) {
        fct(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    }

    start = start_tsc();
    for (int i = 0; i < LOF_NUM_SMALL_RUNS; ++i) {
        fct(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    }
    cyclesNew = stop_tsc(start) / (double) LOF_NUM_SMALL_RUNS;
    flopsPerCycleNew = flopsTotal / cyclesNew;

    if (verobse == 1) {

        printf("Total number of flops: %f\n", flopsTotal);
        printf("%f (cycles) %f (flops / cycle)\n", cyclesNew, flopsPerCycleNew);

        if (compare_with_baseline == 1) {
            printf("%f (cycles) %f (flops / cycle)\n", cyclesOriginal, flopsPerCycleOrig);
            printf("Speedup = %f\n", flopsPerCycleNew / flopsPerCycleOrig);
        }

    }   // if( verobse == 1 )

    free(lrd_score_table_ptr);
    free(lof_score_table_ptr_true);
    free(lof_score_table_ptr);
    free(neighborhood_index_table_ptr);

    return flopsPerCycleNew;
} // performance_measure


void performance_plot_lof_to_file_k(int num_pts_ref, const char* file_name,
                                    const char* function_name, const char* mode,
                                    my_lof_fnc fct, my_lof_fnc fct_original) {
    /**
     * for metrics functions measure performance
     * by varying number of dimensions
     *
     * :param max_num_pts    maximum input dimension
     * :param step           step size for iteration
     * :param name           name of the function / identifier that will be used in the plots.
     *                       has to be in double quotes !
     * :param mode           expected one of: "w" - override, "a" - append
     * :param fct            function to be measured
     * :param fct_correct    baseline implementation to compare with
     *
     * TODO: 1. add filename as input ?
     *       2. add k to the file
     *       3. add grid to file
     */

    // my local computer
    //FILE* results_file = fopen("/Users/sychevaa/Desktop/MyStudies/ETH/FS2020/ASL/old_team025/measurements/lof_results_k.txt", mode);
    //FILE* results_file = fopen("measurements/lof_results_k.txt", mode);
    FILE* results_file = fopen( file_name , mode);
    // SHOULD BE LESS THAN NUM PTS !!! - add assertion
    int k_grid[6] = {4, 10, 100, 250, 500, 1000};

    if (results_file == NULL) {
        fprintf(stderr, "Error at opening %s", results_file);
        printf("Check if the file was generated or the path in open_with_error_check\n");
    }

    unsigned int k, rep;
    double res;

    fprintf(results_file, "%s\n", function_name);
    fprintf(results_file, "%d\n", num_pts_ref);
    fprintf(results_file, "\n");

    //for (num_pts = 4; num_pts < max_num_pts; num_pts += step) {
    for (int i = 0; i < 6; i++){

        k = k_grid[i];
        if( k <= num_pts_ref ){

            fprintf(results_file, "%d, ", k);
            for (rep = 0; rep < LOF_NUM_RUNS; rep++) {
                res = performance_measure_hot_for_lof( k, num_pts_ref, fct, fct_original, 0, 0, 0);
                fprintf(results_file, "%f, ", res);
            }
            fprintf(results_file, "\n");

        }

    } // iterate over dimensions
    fclose(results_file);
}   //  performance_plot_lof_to_file

void performance_plot_lof_to_file_num_pts(int k_ref, const char* file_name,
                                          const char* function_name, const char* mode,
                                          my_lof_fnc fct, my_lof_fnc fct_original) {
    /**
     * for metrics functions measure performance
     * by varying number of dimensions
     *
     * :param max_num_pts    maximum input dimension
     * :param step           step size for iteration
     * :param name           name of the function / identifier that will be used in the plots.
     *                       has to be in double quotes !
     * :param mode           expected one of: "w" - override, "a" - append
     * :param fct            function to be measured
     * :param fct_correct    baseline implementation to compare with
     *
     * TODO:  1. add grid as input ?
     */

    // my local computer
    //FILE* results_file = fopen("/Users/sychevaa/Desktop/MyStudies/ETH/FS2020/ASL/old_team025/measurements/lof_results_num_pts.txt", mode);
    int num_pts_grid[13] = {10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    //FILE* results_file = fopen("measurements/lof_results_num_pts.txt", mode);
    FILE* results_file = fopen( file_name , mode);

    if (results_file == NULL) {
        fprintf(stderr, "Error at opening %s", results_file);
        printf("Check if the file was generated or the path in open_with_error_check\n");
    }

    unsigned int num_pts, rep;
    double res;

    fprintf(results_file, "%s\n", function_name);
    fprintf(results_file, "%d\n", k_ref);
    fprintf(results_file, "\n");

    //for (num_pts = 4; num_pts < max_num_pts; num_pts += step)
    for (int i = 2; i < 13; i++){

        num_pts = num_pts_grid[i];
        if( num_pts >= k_ref ) {    // number of neighbrs can't be lower than the number of points
            fprintf(results_file, "%d, ", num_pts);
            for (rep = 0; rep < LOF_NUM_RUNS; rep++) {
                res = performance_measure_hot_for_lof(k_ref, num_pts, fct, fct_original, 0, 0, 0);
                fprintf(results_file, "%f, ", res);
            }
        }
        fprintf(results_file, "\n");

    } // iterate over dimensions
    fclose(results_file);
}   //  performance_plot_lof_to_file

//*************************************************************************************
//      PIPELINE 2
//-------------------------------------------------------------------------------------

double performance_measure_fast_for_lrdm_pipeline2(int num_pts, int k, int num_reps,
                                                   my_lrdm2_fnc lrdm2_fnc){
    /**
     * wrap around / modification of the function lrdistance_driver that performs the measurements
     * for only one function and returns the median. Can be useful for fast testing which modification is better
     * before moving to producing boxplot measurements which take some time
     *
     * @param lrdf_fct : function to be tested
     * @param flops_fct : function that computes the number of flops depending on the input size
     *                    useful if we experiment with different counting systems
     * @param num_reps : How many times should measurements be repeated
     *                   recomment 1000 ?
     * @param to_verify: compare with baseline implementation ?
     *
     */
    myInt64 start, end;
    double cycles1;
    double multiplier = 1;
    double numRuns = 10;

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

    // Warm-up phase: we determine a number of executions that allows
    do {
        numRuns = numRuns * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < numRuns; i++) {

            lrdm2_fnc(k, num_pts, ComputeLocalReachabilityDensityMerged_Point, dist_k_neighborhood_index, k_neighborhood_index,
                     k_distance_index, lrd_score_table_ptr, lrd_score_neigh_table_ptr);
        }
        end = stop_tsc(start);

        cycles1 = (double) end;
        multiplier = (CYCLES_REQUIRED) / (cycles1);

    } while (multiplier > 2);

    // START MEASUREMENTS

    double* cyclesPtr = XmallocVectorDouble(num_reps);
    for (size_t j = 0; j < num_reps; j++) {

        start = start_tsc();
        for (size_t i = 0; i < numRuns; ++i) {
            lrdm2_fnc(k, num_pts, ComputeLocalReachabilityDensityMerged_Point,
                      dist_k_neighborhood_index, k_neighborhood_index,
                      k_distance_index, lrd_score_table_ptr, lrd_score_neigh_table_ptr);
        }
        end = stop_tsc(start);

        cycles1 = ((double) end) / numRuns;
        cyclesPtr[j] = cycles1;
    }

    qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
    double cycles = cyclesPtr[(int) num_reps / 2 + 1];
    // double perf = round((1000.0 * flopsTotal) / cycles) / 1000.0;

    printf("n: %d k: %d repetitions: %d cycles: %lf\n", num_pts, k, num_reps, cycles );

    free(dist_k_neighborhood_index);
    free(k_distance_index);
    free(k_neighborhood_index);
    free(lrd_score_neigh_table_ptr_true);
    free(lrd_score_neigh_table_ptr);
    free(lrd_score_table_ptr_true);
    free(lrd_score_table_ptr);

    return cycles;

}


double performance_measure_fast_for_lrdm_point_pipeline2(int pnt_idx, int num_pts, int k, int num_reps,
                                                         my_lrdm2_pnt_fnc lrdm2_point_fnc){
    /**
     * wrap around / modification of the function lrdistance_driver that performs the measurements
     * for only one function and returns the median. Can be useful for fast testing which modification is better
     * before moving to producing boxplot measurements which take some time
     *
     * @param lrdf_fct : function to be tested
     * @param flops_fct : function that computes the number of flops depending on the input size
     *                    useful if we experiment with different counting systems
     * @param num_reps : How many times should measurements be repeated
     *                   recomment 1000 ?
     * @param to_verify: compare with baseline implementation ?
     *
     */
    myInt64 start, end;
    double cycles1;
    double multiplier = 1;
    double numRuns = 10;

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

    // Warm-up phase: we determine a number of executions that allows
    do {
        numRuns = numRuns * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < numRuns; i++) {

            lrdm2_point_fnc(pnt_idx, k, dist_k_neighborhood_index, k_distance_index, k_neighborhood_index,
                            lrd_score_table_ptr);
        }
        end = stop_tsc(start);

        cycles1 = (double) end;
        multiplier = (CYCLES_REQUIRED) / (cycles1);

    } while (multiplier > 2);

    // START MEASUREMENTS

    double* cyclesPtr = XmallocVectorDouble(num_reps);
    for (size_t j = 0; j < num_reps; j++) {

        start = start_tsc();
        for (size_t i = 0; i < numRuns; ++i) {
            lrdm2_point_fnc(pnt_idx, k, dist_k_neighborhood_index, k_distance_index, k_neighborhood_index,
                      lrd_score_table_ptr);
        }
        end = stop_tsc(start);

        cycles1 = ((double) end) / numRuns;
        cyclesPtr[j] = cycles1;
    }

    qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
    double cycles = cyclesPtr[(int) num_reps / 2 + 1];
    // double perf = round((1000.0 * flopsTotal) / cycles) / 1000.0;

    printf("n: %d k: %d repetitions: %d cycles: %lf\n", num_pts, k, num_reps, cycles );

    free(dist_k_neighborhood_index);
    free(k_distance_index);
    free(k_neighborhood_index);
    free(lrd_score_neigh_table_ptr_true);
    free(lrd_score_neigh_table_ptr);
    free(lrd_score_table_ptr_true);
    free(lrd_score_table_ptr);

    return cycles;

}