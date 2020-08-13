//
// Created by fvasluia on 5/4/20.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../include/metrics.h"
#include "../../include/tsc_x86.h"
#include "../../include/utils.h"
#include "../../include/lof_baseline.h"

int METRICS_NUM_RUNS_UN = 1000;
int METRICS_NUM_SMALL_RUNS_UN = 10;    // average over 10

double UnrolledEuclideanDistance(const double* v1_ptr, const double* v2_ptr, int n_dim) {
    double dist = 0.0;
    if (n_dim <= 10) {
        for (int i = 0; i < n_dim; ++i) {
            dist += (v1_ptr[i] - v2_ptr[i]) * (v1_ptr[i] - v2_ptr[i]);
        }
    } else {
        double acc_1 = 0.0, acc_2 = 0.0, acc_3 = 0.0, acc_4 = 0.0, acc_5 = 0.0,
                acc_6 = 0.0, acc_7 = 0.0, acc_8 = 0.0, acc_9 = 0.0, acc_10 = 0.0;
        double a1, a2, a3, a4, a5, a6, a7, a8, a9, a10;
        int i;
        int n_runs = 0;
        for (i = 0; (i + 9) < n_dim; i += 10) {
            ++n_runs;
            a1 = (v1_ptr[i] - v2_ptr[i]);
            a2 = (v1_ptr[i + 1] - v2_ptr[i + 1]);
            a3 = (v1_ptr[i + 2] - v2_ptr[i + 2]);
            a4 = (v1_ptr[i + 3] - v2_ptr[i + 3]);
            a5 = (v1_ptr[i + 4] - v2_ptr[i + 4]);
            a6 = (v1_ptr[i + 5] - v2_ptr[i + 5]);
            a7 = (v1_ptr[i + 6] - v2_ptr[i + 6]);
            a8 = (v1_ptr[i + 7] - v2_ptr[i + 7]);
            a9 = (v1_ptr[i + 8] - v2_ptr[i + 8]);
            a10 = (v1_ptr[i + 9] - v2_ptr[i + 9]);

            acc_1 += a1 * a1;
            acc_2 += a2 * a2;
            acc_3 += a3 * a3;
            acc_4 += a4 * a4;
            acc_5 += a5 * a5;
            acc_6 += a6 * a6;
            acc_7 += a7 * a7;
            acc_8 += a8 * a8;
            acc_9 += a9 * a9;
            acc_10 += a10 * a10;

        }
        dist = acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6 + acc_7 + acc_8 + acc_9 + acc_10;
        if (10 * n_runs < n_dim) {
            for (int j = 10 * n_runs; j < n_dim; ++j) {
                dist += (v1_ptr[j] - v2_ptr[j]) * (v1_ptr[j] - v2_ptr[j]);
            }
        }
    }
    return sqrt(dist);
}

// ----------------------------------------------------

double CosineSimilarityFaster1(const double* v1_ptr, const double* v2_ptr, int dim) {
    /** 
    * implementation of cosine similarity
    * see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html#scipy.spatial.distance.cosine
    */

    unsigned int i = 0;
    double dist, denom, numerator = 0;
    double x_2 = 0;
    double y_2 = 0;

    if (dim <= 8) {

        for (; i < dim; ++i) {
            numerator += v1_ptr[i] * v2_ptr[i];
            x_2 += v1_ptr[i] * v1_ptr[i];
            y_2 += v2_ptr[i] * v2_ptr[i];
            // printf("%d  %f : %f\nfrom %f : %f \n", i, x_2, y_2, v1_ptr[i], v2_ptr[i]);
        }

    } else {

        double acc11, acc12, acc13, acc14, acc15, acc16, acc17, acc18;
        double acc21, acc22, acc23, acc24, acc25, acc26, acc27, acc28;
        double numerator1 = 0, numerator2 = 0, numerator3 = 0, numerator4 = 0, numerator5 = 0, numerator6 = 0, numerator7 = 0, numerator8 = 0;

        for (; i + 8 < dim; i += 8) {

            numerator1 += v1_ptr[i] * v2_ptr[i];
            numerator2 += v1_ptr[i + 1] * v2_ptr[i + 1];
            numerator3 += v1_ptr[i + 2] * v2_ptr[i + 2];
            numerator4 += v1_ptr[i + 3] * v2_ptr[i + 3];
            numerator5 += v1_ptr[i + 4] * v2_ptr[i + 4];
            numerator6 += v1_ptr[i + 5] * v2_ptr[i + 5];
            numerator7 += v1_ptr[i + 6] * v2_ptr[i + 6];
            numerator8 += v1_ptr[i + 7] * v2_ptr[i + 7];

            acc11 += v1_ptr[i] * v1_ptr[i];
            acc21 += v2_ptr[i] * v2_ptr[i];
            acc12 += v1_ptr[i + 1] * v1_ptr[i + 1];
            acc22 += v2_ptr[i + 1] * v2_ptr[i + 1];
            acc13 += v1_ptr[i + 2] * v1_ptr[i + 2];
            acc23 += v2_ptr[i + 2] * v2_ptr[i + 2];
            acc14 += v1_ptr[i + 3] * v1_ptr[i + 3];
            acc24 += v2_ptr[i + 3] * v2_ptr[i + 3];
            acc15 += v1_ptr[i + 4] * v1_ptr[i + 4];
            acc25 += v2_ptr[i + 4] * v2_ptr[i + 4];
            acc16 += v1_ptr[i + 5] * v1_ptr[i + 5];
            acc26 += v2_ptr[i + 5] * v2_ptr[i + 5];
            acc17 += v1_ptr[i + 6] * v1_ptr[i + 6];
            acc27 += v2_ptr[i + 6] * v2_ptr[i + 6];
            acc18 += v1_ptr[i + 7] * v1_ptr[i + 7];
            acc28 += v2_ptr[i + 7] * v2_ptr[i + 7];


        } // for

        x_2 = acc11 + acc12 + acc13 + acc14 + acc15 + acc16 + acc17 + acc18;
        y_2 = acc21 + acc22 + acc23 + acc24 + acc25 + acc26 + acc27 + acc28;
        numerator =
                numerator1 + numerator2 + numerator3 + numerator4 + numerator5 + numerator6 + numerator7 + numerator8;

        // remaining i
        for (; i < dim; i++) {
            numerator += v1_ptr[i] * v2_ptr[i];
            x_2 += v1_ptr[i] * v1_ptr[i];
            y_2 += v2_ptr[i] * v2_ptr[i];
        }

    }   // else

    x_2 = sqrt(x_2);
    y_2 = sqrt(y_2);
    denom = x_2 * y_2;
    dist = 1 - numerator / denom;

    return dist;
}


/*
double performance_measure_hot_for_metrics(int dim, double (* fct)(const double*, const double*, int),
                                           double (* fct_correct)(const double*, const double*, int),
                                           int to_verify, int verobse, int compare_with_baseline) {
    //
    // new function for measuring the perfomance of the function (hot cache)
    // data is used to create corresponding box plots
    //
    // :param fct         : function we currently consider
    //:param fct_correct : original correct implementation w.r.t which we verify
    //:param to_verify   : verify the fct against fct_correct
     // :param verbose     : print intermediate output
     //

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
    //
     // for metrics functions measure performance
     // by varying number of dimensions
     //
     // :param max_dim        largest dimensionality considered on the plots
     // :param step           step size for iteration
     // :param name           name of the function / identifier that will be used in the plots
     // :param mode           expected one of: "w" - override, "a" - append
     // :param fct            function to be measured
     // :param fct_correct    baseline implementation to compare with
     //
     // TODO: 1. add filename as input ?
     //

    // gcc -O3 -mavx2 -mfma -march=native FasterMetrics.c -o FasterMetrics
    // ./FasterMetrics


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


int exec_metrics() {

    int max_dim = 550, step = 50;

    performance_plot_metrics_to_file(max_dim, step, "baseline", "w", CosineSimilarity, CosineSimilarity);
    performance_plot_metrics_to_file(max_dim, step, "loop unrolling 8", "a", CosineSimilarityFaster1, CosineSimilarity);

}
 */