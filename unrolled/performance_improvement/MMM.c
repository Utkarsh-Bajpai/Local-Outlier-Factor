//
// Created by Sycheva  Anastasia on 18.05.20.
//
#include <math.h>
#include <stdlib.h>

#include "../../include/utils.h"
#include "../include/MMM.h"
#include "../../include/tsc_x86.h"
#include "../../include/tests.h"

typedef long (* my_fun)(int, int, int, int, const double*, double*);

//_____________________________ BASELINE  _____________________________________

double mmm_baseline(int n, int dim, const double* input_points_ptr, double* result) {
    /**
     * Compute pairwise distances and store them in a lower triangular matrix
     */
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            double sum = 0;
            for (int k = 0; k < dim; k++) {
                sum += input_points_ptr[i * dim + k] * input_points_ptr[j * dim + k];
            }
            result[i * n + j] = sum;
        }
    }
    return 2.0 * n * (n - 1) * dim / 2;

}

//_____________________________ MINI  _____________________________________

long mini_mm_1_block(int n, int dim, int B0, int B1, const double* input, double* result) {
    /**
     * Cache blocking for mmm_baseline
     *
     * @param Bp: block size for points
     * @param Bd: block size for dimensions
     */

    int i, j, i1, j1, k, k1, i2, j2;
    double sum;
    int BK = 2;

    for (i = 0; i < n - B0; i += B0) {
        for (j = 0; j < i - B0; j += B0) {
            for (k = 0; k < dim - BK; k += BK) {

                for (k1 = k; k1 < k + BK; k1++) {
                    for (i1 = i; i1 < i + B0; i1++) {
                        for (j1 = j; j1 < j + B0; j1++) {

                            result[i1 * n + j1] += input[i1 * dim + k1] * input[j1 * dim + k1];
                        }
                    }
                }
            }

            for (; k < dim; k++) {
                // NO BLOCKING - CHANGE !
                for (i1 = i; i1 < i + B0; i1++) {
                    for (j1 = j; j1 < j + B0; j1++) {
                        result[i1 * n + j1] += input[i1 * dim + k] * input[j1 * dim + k];
                    }
                }
            }
        } // j_out
        // NO BLOCKING - CHANGE !
        for (i1 = i; i1 < i + B0; i1++) {
            for (j1 = j; j1 < i1; j1++) {    // CHANGE HERE !
                for (k = 0; k < dim; k++) {
                    result[i1 * n + j1] += input[i1 * dim + k] * input[j1 * dim + k];
                }
            }
        }

    } // i_out

    // Collect remaining I
    for (; i < n; i++) {
        for (j = 0; j < i; j++) {       // CHANGED HERE
            for (k = 0; k < dim; k++) {
                result[i * n + j] += input[i * dim + k] * input[j * dim + k];
            }
        }
    }

    return 2 * n * (n - 1) * dim / 2;
}


long mini_mm_2_blocks(int n, int dim, int B0, int B1, const double* input, double* result) {
    /**
     * Cache blocking for mmm_baseline
     *
     * @param Bp: block size for points
     * @param Bd: block size for dimensions
     */

    int i, j, i1, j1, k, k1, i2, j2;
    double sum;
    int BK = 2;

    for (i = 0; i < n - B0; i += B0) {
        for (j = 0; j < i - B0; j += B0) {      //CHANGE

            for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
                for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                    double s_00 = 0, s_01 = 0, s_10 = 0, s_11 = 0;
                    for (k = 0; k < dim - BK; k += BK) {

                        for (k1 = k; k1 < k + BK; k1++) {
                            s_00 += input[i1 * dim + k1] * input[j1 * dim + k1];
                            s_01 += input[i1 * dim + k1] * input[(j1 + 1) * dim + k1];
                            s_10 += input[(i1 + 1) * dim + k1] * input[j1 * dim + k1];
                            s_11 += input[(i1 + 1) * dim + k1] * input[(j1 + 1) * dim + k1];
                        }
                    }
                    for (; k < dim; k++) {
                        s_00 += input[i1 * dim + k] * input[j1 * dim + k];
                        s_01 += input[i1 * dim + k] * input[(j1 + 1) * dim + k];
                        s_10 += input[(i1 + 1) * dim + k] * input[j1 * dim + k];
                        s_11 += input[(i1 + 1) * dim + k] * input[(j1 + 1) * dim + k];
                    }
                    result[i1 * n + j1] = s_00;
                    result[i1 * n + j1 + 1] = s_01;
                    result[(i1 + 1) * n + j1] = s_10;
                    result[(i1 + 1) * n + j1 + 1] = s_11;
                }
            }


        } // j_out

        //collecting J left up to i1
        for (i1 = i; i1 < i + B0; i1++) {
            for (j1 = j; j1 < i1; j1++) {
                sum = 0;
                for (k = 0; k < dim; k++) {
                    sum += input[i1 * dim + k] * input[j1 * dim + k];
                }
                result[i1 * n + j1] = sum;
            }
        }
    }

    // Collect remaining I
    for (; i < n; i++) {
        for (j = 0; j < i; j++) {       // CHANGED HERE
            sum = 0;
            for (k = 0; k < dim; k++) {
                sum += input[i * dim + k] * input[j * dim + k];
            }
            result[i * n + j] = sum;
        }
    }

    return 2 * n * (n - 1) * dim / 2;
} // mini_mmm_lower_triangular



long micro_mm_2_blocks(int n, int dim, int B0, int B1, const double* input, double* result) {
    /**
     * Cache blocking for mmm_baseline
     *
     * @param Bp: block size for points
     * @param Bd: block size for dimensions
     */

    int i, j, i1, j1, k, k1, i2, j2;
    double sum;
    int BK = 2;


    for (i = 0; i < n - B0; i += B0) {
        for (j = 0; j < i - B0; j += B0) {      //CHANGE

            for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
                for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                    double s_00 = 0, s_01 = 0, s_10 = 0, s_11 = 0;
                    for (k = 0; k <= dim - BK; k += BK) {
                        s_00 += input[i1 * dim + k] * input[j1 * dim + k];
                        s_01 += input[i1 * dim + k] * input[(j1 + 1) * dim + k];
                        s_10 += input[(i1 + 1) * dim + k] * input[j1 * dim + k];
                        s_11 += input[(i1 + 1) * dim + k] * input[(j1 + 1) * dim + k];

                        s_00 += input[i1 * dim + k + 1] * input[j1 * dim + k + 1];
                        s_01 += input[i1 * dim + k + 1] * input[(j1 + 1) * dim + k + 1];
                        s_10 += input[(i1 + 1) * dim + k + 1] * input[j1 * dim + k + 1];
                        s_11 += input[(i1 + 1) * dim + k + 1] * input[(j1 + 1) * dim + k + 1];
                    }
                    for (; k < dim; k++) {
                        s_00 += input[i1 * dim + k] * input[j1 * dim + k];
                        s_01 += input[i1 * dim + k] * input[(j1 + 1) * dim + k];
                        s_10 += input[(i1 + 1) * dim + k] * input[j1 * dim + k];
                        s_11 += input[(i1 + 1) * dim + k] * input[(j1 + 1) * dim + k];
                    }
                    result[i1 * n + j1] = s_00;
                    result[i1 * n + j1 + 1] = s_01;
                    result[(i1 + 1) * n + j1] = s_10;
                    result[(i1 + 1) * n + j1 + 1] = s_11;
                }
            }
        } // j_out

        //collecting J left up to i1
        for (i1 = i; i1 < i + B0; i1++) {
            for (j1 = j; j1 < i1; j1++) {
                double s_00 = 0;
                for (k = 0; k < dim - BK; k += BK) {
                    s_00 += input[i1 * dim + k] * input[j1 * dim + k];
                    s_00 += input[i1 * dim + k + 1] * input[j1 * dim + k + 1];
                }
                for (; k < dim; k++) {
                    s_00 += input[i1 * dim + k] * input[j1 * dim + k];
                }
                result[i1 * n + j1] = s_00;
            }
        }
    }

    // Collect remaining I
    for (; i < n; i++) {
        for (j = 0; j < i; j++) {
            double s_00 = 0;
            for (k = 0; k < dim - BK; k += BK) {
                s_00 += input[i * dim + k] * input[j * dim + k];
                s_00 += input[i * dim + k + 1] * input[j * dim + k + 1];
            }
            for (; k < dim; k++) {
                s_00 += input[i * dim + k] * input[j * dim + k];
            }
            result[i * n + j] = s_00;
        }
    }

    return 2 * n * (n - 1) * dim / 2;
} // mini_mmm_lower_triangular


/**
 * Micro 2x2x4
 **/
double mmm_unroll_fastest(int n, int dim, int B0, int B1, int BK, const double* input, double* result) {
    int i, j, i1, j1, k, k1, i2, j2, k2, i3, j3;
    double sum;
    int BK2 = 4;
    int B2 = 2;

    for (i = 0; i < n - B0; i += B0) {
        for (j = 0; j < i - B0; j += B0) {
            for (k = 0; k < dim - BK; k += BK) {

                for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
                    for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                        for (i2 = i1; i2 + B2 <= i1 + B1; i2 += B2) {
                            for (j2 = j1; j2 + B2 <= j1 + B1; j2 += B2) {
                                double s_00 = 0, s_01 = 0, s_10 = 0, s_11 = 0;

                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    s_00 += input[i2 * dim + k1] * input[j2 * dim + k1];
                                    s_01 += input[i2 * dim + k1] * input[(j2 + 1) * dim + k1];
                                    s_10 += input[(i2 + 1) * dim + k1] * input[j2 * dim + k1];
                                    s_11 += input[(i2 + 1) * dim + k1] * input[(j2 + 1) * dim + k1];

                                    s_00 += input[i2 * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_01 += input[i2 * dim + k1 + 1] * input[(j2 + 1) * dim + k1 + 1];
                                    s_10 += input[(i2 + 1) * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_11 += input[(i2 + 1) * dim + k1 + 1] * input[(j2 + 1) * dim + k1 + 1];

                                    s_00 += input[i2 * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_01 += input[i2 * dim + k1 + 2] * input[(j2 + 1) * dim + k1 + 2];
                                    s_10 += input[(i2 + 1) * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_11 += input[(i2 + 1) * dim + k1 + 2] * input[(j2 + 1) * dim + k1 + 2];

                                    s_00 += input[i2 * dim + k1 + 3] * input[j2 * dim + k1 + 3];
                                    s_01 += input[i2 * dim + k1 + 3] * input[(j2 + 1) * dim + k1 + 3];
                                    s_10 += input[(i2 + 1) * dim + k1 + 3] * input[j2 * dim + k1 + 3];
                                    s_11 += input[(i2 + 1) * dim + k1 + 3] * input[(j2 + 1) * dim + k1 + 3];
                                }
                                result[i2 * n + j2] += s_00;
                                result[i2 * n + j2 + 1] += s_01;
                                result[(i2 + 1) * n + j2] += s_10;
                                result[(i2 + 1) * n + j2 + 1] += s_11;
                            }
                        }
                    }
                }
            }
            for (k1 = k; k1 < dim; k1++) {
                for (i1 = i; i1 < i + B0; i1 += 1) {
                    for (j1 = j; j1 < j + B0; j1 += 1) {
                        result[i1 * n + j1] += input[i1 * dim + k1] * input[j1 * dim + k1];
                    }
                }
            }
        }
        // j_out
        for (i1 = i; i1 < i + B0; i1++) {
            for (j1 = j; j1 < i1; j1++) {
                sum = 0;
                for (k = 0; k < dim; k++) {
                    sum += input[i1 * dim + k] * input[j1 * dim + k];
                }
                result[i1 * n + j1] = sum;
            }
        }
    } // i_out

    for (; i < n; i++) {
        for (j = 0; j < i; j++) {
            sum = 0;
            for (k = 0; k < dim; k++) {
                sum += input[i * dim + k] * input[j * dim + k];
            }
            result[i * n + j] = sum;
        }
    }
    return 2.0 * n * (n - 1) * dim / 2;
}


#define  NUM_FUNCTIONS 4
#define  CYCLES_REQ 1e8

void mmm_driver(int n, int dim, int B0, int B1, int nr_reps) {
    double* input_points = XmallocMatrixDoubleRandom(n, dim);
    double* mm_true = XmallocMatrixDouble(n, n);

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &mmm_baseline;
    fun_array[1] = &mini_mm_1_block;
//    fun_array[1] = &mini_mmm_full;
    fun_array[2] = &mini_mm_2_blocks;
    fun_array[3] = &micro_mm_2_blocks;

    char* fun_names[NUM_FUNCTIONS] = {"baseline_lower", "mini_mm_1", "mini_mm_2", "micro_mm",
                                      "V5", "V6", "V7", "V8",
                                      "V9"};

    myInt64 start, end;
    double cycles1;
    double flops;

//    FILE* results_file = open_with_error_check("../lrd_benchmark_k10_full.txt", "a");

    mmm_baseline(n, dim, input_points, mm_true);

    for (int fun_index = 0; fun_index < NUM_FUNCTIONS; fun_index++) {
        double* mm_results = XmallocMatrixDouble(n, n);

        // VERIFICATION : ------------------------------------------------------------------------
        (*fun_array[fun_index])(n, dim, B0, B1, input_points, mm_results);
        int ver = test_double_arrays(n * n, 1e-3, mm_results, mm_true);
        if (ver != 1) {
            print_matrices(n, mm_true, mm_results);
//            printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
            exit(-1);
        }

        double multiplier = 1;
        double numRuns = 10;

        // Warm-up phase: we determine a number of executions that allows
        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[fun_index])(n, dim, B0, B1, input_points, mm_results);
            }
            end = stop_tsc(start);

            cycles1 = (double) end;
            multiplier = (CYCLES_REQ) / (cycles1);

        } while (multiplier > 2);


        double totalCycles = 0;
        double* cyclesPtr = XmallocVectorDouble(nr_reps);

        for (size_t j = 0; j < nr_reps; j++) {
            CleanTheCache(200);
            start = start_tsc();
            for (size_t i = 0; i < numRuns; ++i) {
                flops = (*fun_array[fun_index])(n, dim, B0, B1, input_points, mm_results);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;

            totalCycles += cycles1;
        }

        qsort(cyclesPtr, nr_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[(int) nr_reps / 2 + 1];


        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        printf("%s n:%d d:%d bp:%d cycles:%lf perf:%lf \n", fun_names[fun_index], n, dim, B0, cycles, perf);


//        free(mm_results);
//        mm_results = XmallocMatrixDouble(n, n);
//        (*fun_array[fun_index])(n, dim, Bp, Bd, input_points, mm_results);
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++) {
//                printf("%lf ", mm_results[i * n + j]);
//            }
//            printf("\n");
//        }

        free(mm_results);
        free(cyclesPtr);
//        fprintf(results_file, "%s, %d, %d, %lf, %lf\n", fun_names[fun_index], num_pts, k, cycles, perf);
    }

    printf("-------------\n");
    free(input_points);
    free(mm_true);
//    fclose(results_file);

}
