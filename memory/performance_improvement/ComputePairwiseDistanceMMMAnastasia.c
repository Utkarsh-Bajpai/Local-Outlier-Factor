//
// Created by Sycheva  Anastasia on 18.05.20.
//
#include <math.h>
#include <assert.h>

#include "../../include/utils.h"
#include "../include/ComputePairwiseDistanceMMMAnastasia.h"
#include "../../include/performance_measurement.h"


//double mmm_baseline(int num_pts, int dim, const double* input_points_ptr, double* result) {
//    /**
//     * Compute pairwise distances and store them in a lower triangular matrix
//     */
//    int i, j;
//    for (i = 0; i < num_pts; i++) {
//        for (j = 0; j < i; j++) { // OLD:  j < i
//            double sum = 0;
//            for (int k = 0; k < dim; k++) {
//                sum += input_points_ptr[i * dim + k] * input_points_ptr[j * dim + k];
//            }
//            result[i * num_pts + j] = sum;
//        }
//    }
//    return 2.0 * num_pts * (num_pts - 1) / 2;
//}


void mini_mmm_lower_triangular(int num_pts, int dim, int Bp, int Bd, const double* input_points_ptr, double* result) {
    /**
     * @param Bp: block size for points
     * @param Bd: block size for dimensions
     *
     * @note: parts of the code where I collect remaining elements of different loops can be
     *        likewise processed in blocks (indicated by NO BLOCKING - CHANGE !) however it
     *        is not done yet since the goal was just to get the correct results.
     */

    int i_out, j_out, i_in, j_in, k_out, k_in;

    for (i_out = 0; i_out < num_pts - Bp; i_out += Bp) {

        for (j_out = 0; j_out < i_out - Bp; j_out += Bp) {      //CHANGE
            for (k_out = 0; k_out < dim - Bd; k_out += Bd) {

                // MINI MMM -------------------------------------------------------
                for (i_in = i_out; i_in < i_out + Bp; i_in++) {
                    for (j_in = j_out; j_in < j_out + Bp; j_in++) {
                        for (k_in = k_out; k_in < k_out + Bd; k_in++) {

                            result[i_in * num_pts + j_in] +=
                                    input_points_ptr[i_in * dim + k_in] * input_points_ptr[j_in * dim + k_in];
                        }
                    }
                }
            }

            for (; k_out < dim; k_out++) {

                // NO BLOCKING - CHANGE !
                for (i_in = i_out; i_in < i_out + Bp; i_in++) {
                    for (j_in = j_out; j_in < j_out + Bp; j_in++) {
                        result[i_in * num_pts + j_in] +=
                                input_points_ptr[i_in * dim + k_out] * input_points_ptr[j_in * dim + k_out];
                    }
                }
            }
        } // j_out
        int j_start = j_out;
        // NO BLOCKING - CHANGE !
        for (i_in = i_out; i_in < i_out + Bp; i_in++) {
            for (int j_out = j_start; j_out < i_in; j_out++) {    // CHANGE HERE !
                for (int k = 0; k < dim; k++) {
                    result[i_in * num_pts + j_out] +=
                            input_points_ptr[i_in * dim + k] * input_points_ptr[j_out * dim + k];
                }
            }
        }

    } // i_out

    // Collect remaining I
    for (; i_out < num_pts; i_out++) {
        for (int j = 0; j < i_out; j++) {       // CHANGED HERE
            for (int k = 0; k < dim; k++) {
                result[i_out * num_pts + j] += input_points_ptr[i_out * dim + k] * input_points_ptr[j * dim + k];
            }
        }
    }

} // mini_mmm_lower_triangular


//_____________________________ MICRO __________________________________________________________________


void micro_mmm_lower_triangular(int num_pts, int dim, int Bp, int Bd, int Bpr, int Bdr, const double* input_points_ptr,
                                double* result, int verbose) {
    /**
     * @param Bp:  cache block size for number of points
     * @param Bd:  cache block size for number of dimensions
     * @param Bpr: register block size for number of points
     * @param Bdr: register block size for number of dimensions
     *
     * @note: parts of the code where I collect remaining elements of different loops can be
     *        likewise processed in blocks (indicated by NO BLOCKING - CHANGE !) however it
     *        is not done yet since the goal was just to get the correct results.
     *
     * @important: still fails some tests - see below test_micro_mmm_triangular_driver()
     */

    // assert( Bpr < Bp  && Bdr < Bd );

    int i_out, j_out, k_out;
    int i_in, j_in, k_in;
    int i_r, j_r, k_r;  // iterate over registers (hopefully)

    for (i_out = 0; i_out < num_pts - Bp; i_out += Bp) {
        for (j_out = 0; j_out < i_out - Bp; j_out += Bp) {      //CHANGE
            for (k_out = 0; k_out < dim - Bd; k_out += Bd) {

                if (verbose) printf("Block Small\n");

                // MINI MMM -------------------------------------------------------
                for (i_in = i_out; i_in + Bpr < i_out + Bp; i_in += Bpr) {
                    for (j_in = j_out; j_in + Bpr < j_out + Bp; j_in += Bpr) {
                        for (k_in = k_out; k_in + Bdr < k_out + Bd; k_in += Bdr) {

                            // MICRO MMM -------------------------------------------------------
                            for (i_r = i_in; i_r < i_in + Bpr; i_r++) {
                                for (j_r = j_in; j_r < j_in + Bpr; j_r++) {
                                    for (k_r = k_in; k_r < k_in + Bdr; k_r++) {

                                        result[i_r * num_pts + j_r] +=
                                                input_points_ptr[i_r * dim + k_r] * input_points_ptr[j_r * dim + k_r];

                                    }
                                }
                            }
                        }

                        // Remining k_in
                        for (; k_in < k_out + Bd; k_in++) {
                            for (i_r = i_in; i_r < i_in + Bpr; i_r++) {
                                for (j_r = j_in; j_r < j_in + Bpr; j_r++) {

                                    result[i_r * num_pts + j_r] +=
                                            input_points_ptr[i_r * dim + k_in] * input_points_ptr[j_r * dim + k_in];

                                } // j_r
                            } // i_r
                        } // k_in
                    } // j_in

                    // Remining j_in
                    int j_start_in = j_in;
                    for (i_r = i_in; i_r < i_in + Bpr; i_r++) {
                        for (int j = j_start_in; j < i_r; j++) {
                            for (int k = k_out; k < k_out + Bd; k++) {
                                result[i_r * num_pts + j] +=
                                        input_points_ptr[i_r * dim + k] * input_points_ptr[j * dim + k];
                            } // k
                        } // j
                    } // i_r

                } // i_in

                // Remaining i_in

                for (; i_in < i_out + Bp; i_in++) {
                    for (j_in = j_out; j_in < j_out + Bp; j_in++) {
                        for (k_in = k_out; k_in < k_out + Bd; k_in++) {
                            result[i_in * num_pts + j_in] +=
                                    input_points_ptr[i_in * dim + k_in] * input_points_ptr[j_in * dim + k_in];
                        } // k_in
                    } // j_in
                } // i_in

            }

            // SAME LIKE MINI MMM

            for (; k_out < dim; k_out++) {

                // NO BLOCKING - CHANGE !
                for (i_in = i_out; i_in < i_out + Bp; i_in++) {
                    for (j_in = j_out; j_in < j_out + Bp; j_in++) {
                        result[i_in * num_pts + j_in] +=
                                input_points_ptr[i_in * dim + k_out] * input_points_ptr[j_in * dim + k_out];
                    }
                }

            }
        } // j_out
        int j_start = j_out;
        // NO BLOCKING - CHANGE !
        if (verbose) printf("Block J\n");
        for (i_in = i_out; i_in < i_out + Bp; i_in++) {
            for (int j_out = j_start; j_out < i_in; j_out++) {    // CHANGE HERE !
                if (verbose) printf("i = %d j = %d \n", i_in, j_out);
                for (int k = 0; k < dim; k++) {
                    result[i_in * num_pts + j_out] +=
                            input_points_ptr[i_in * dim + k] * input_points_ptr[j_out * dim + k];
                }
            }
        }

    } // i_out

    if (verbose) printf("Block I\n");
    // Collect remaining I
    for (; i_out < num_pts; i_out++) {
        if (verbose) printf("Remain i = %d \n", i_out);
        for (int j = 0; j < i_out; j++) {       // CHANGED HERE
            for (int k = 0; k < dim; k++) {
                result[i_out * num_pts + j] += input_points_ptr[i_out * dim + k] * input_points_ptr[j * dim + k];
            }
        }
    }

} // mmm_tmp


//_____________________________ POSTPROCESSING _________________________________________________________

int test_mini_mmm_lower_triangular_single(int num_pts, int dim, int Bp, int Bd, my_mini_mmm mini_fct, my_mmm mmm_fct) {
    /**
     *
     * @param mini_fct: function for computing cache optimized mmm.
     *                  expected one of mini_mmm_lower_triangular or mini_mmm_full
     * @param mmm_fct: function for computing baseline results.
     *                 expected one of mmm_baseline or baseline_full
     */

    double* gt_results = XmallocMatrixDouble(num_pts, num_pts);
    double* results = XmallocMatrixDouble(num_pts, num_pts);
    double* input_points = XmallocMatrixDoubleRandom(num_pts, num_pts);

    int count_different = 0; // count number of different entries

    mmm_fct(num_pts, dim, input_points, gt_results);
    mini_fct(num_pts, dim, Bp, Bd, input_points, results);


    printf("Compare baseline with mmm1\n");
    printf("------------\n");
    int i, j;
    for (i = 0; i < num_pts; i++) {
        for (j = 0; j < num_pts; j++) {

            if (fabs(gt_results[i * num_pts + j] - results[i * num_pts + j]) > 0.001) {
                count_different += 1;
                printf("Different results at %d %d : %4.3lf-%4.3lf\n", i, j, gt_results[i * num_pts + j],
                       results[i * num_pts + j]);
            }
        }
        //printf("\n");
    } // for
    int num_pts_sq = num_pts * num_pts;
    printf("Number of different elements %d (out of %d) \n", count_different, num_pts_sq);
    return count_different;

}

int test_micro_mmm_lower_triangular_single(int num_pts, int dim, int Bp, int Bd, int Bpr, int Bdr, int verbose) {

    double* gt_results = XmallocMatrixDouble(num_pts, num_pts);
    double* results = XmallocMatrixDouble(num_pts, num_pts);
    double* input_points = XmallocMatrixDoubleRandom(num_pts, num_pts);

    int count_different = 0; // count number of different entries

    mmm_baseline(num_pts, dim, input_points, gt_results);
    micro_mmm_lower_triangular(num_pts, dim, Bp, Bd, Bpr, Bdr, input_points, results, 0);

    printf("Compare baseline with mmm1\n");
    printf("------------\n");
    int i, j;
    for (i = 0; i < num_pts; i++) {
        for (j = 0; j < num_pts; j++) {

            if (fabs(gt_results[i * num_pts + j] - results[i * num_pts + j]) > 0.001) {
                count_different += 1;
                if (verbose) {
                    printf("Different results at %d %d : %4.3lf-%4.3lf\n", i, j, gt_results[i * num_pts + j],
                           results[i * num_pts + j]);
                }
            }
        }
        //printf("\n");
    } // for
    int num_pts_sq = num_pts * num_pts;
    printf("Number of different elements %d (out of %d) \n", count_different, num_pts_sq);
    return count_different;

}

//______________________________________________________________________________________________________________________

void test_mini_mmm_lower_triangular_driver(my_mini_mmm mini_fct, my_mmm mmm_fct) {
    /**
     *
     * @param mini_fct: function for computing cache optimized mmm.
     *                  expected one of mini_mmm_lower_triangular or mini_mmm_full
     * @param mmm_fct: function for computing baseline results.
     *                 expected one of mmm_baseline or baseline_full
     */

    // Create Test Values:
    struct MMM_INPUT test_values[5];

    // Block size multiples
    test_values[0].num_pts = 12;
    test_values[0].dim = 4;
    test_values[0].Bp = 2;
    test_values[0].Bd = 2;
    // Not block size multiples
    test_values[1].num_pts = 17;
    test_values[1].dim = 4;
    test_values[1].Bp = 4;
    test_values[1].Bd = 3;
    // Many points
    test_values[2].num_pts = 100;
    test_values[2].dim = 5;
    test_values[2].Bp = 3;
    test_values[2].Bd = 3;
    // Block size dim too large points
    test_values[3].num_pts = 201;
    test_values[3].dim = 3;
    test_values[3].Bp = 4;
    test_values[3].Bd = 4;
    // Block size num_pts too large points
    test_values[4].num_pts = 10;
    test_values[4].dim = 4;
    test_values[4].Bp = 20;
    test_values[4].Bd = 2;


    for (int i = 0; i < 5; ++i) {
        printf("\n\nTest case: %d\n", i);
        printf("num_pts : %d, dim : %d, Bp : %d, Bd : %d\n", test_values[i].num_pts, test_values[i].dim,
               test_values[i].Bp, test_values[i].Bd);
        int val = test_mini_mmm_lower_triangular_single(test_values[i].num_pts, test_values[i].dim, test_values[i].Bp,
                                                        test_values[i].Bd, mini_fct, mmm_fct);
        if (val != 0) {
            printf("DIFFERENT MATRICES !!!!\n\n");
        }
    }
}


void test_micro_mmm_triangular_driver() {

    // Create Test Values:
    struct MMM_MICRO_INPUT test_micro[6];

    // NB: % - divisible, !% - not divisible
    // Standard: number of points % cache block size % register block size
    test_micro[0].num_pts = 20;
    test_micro[0].dim = 4;
    test_micro[0].Bp = 4;
    test_micro[0].Bd = 4;
    test_micro[0].Bpr = 1;
    test_micro[0].Bdr = 1;

    // number of points !% cache block size % register block size
    test_micro[1].num_pts = 101;
    test_micro[1].dim = 4;
    test_micro[1].Bp = 4;
    test_micro[1].Bd = 4;
    test_micro[1].Bpr = 2;
    test_micro[1].Bdr = 2;

    // number of dimensions !% cache block size % register block size   FAILS
    test_micro[2].num_pts = 137;
    test_micro[2].dim = 5;
    test_micro[2].Bp = 4;
    test_micro[2].Bd = 4;
    test_micro[2].Bpr = 1;
    test_micro[2].Bdr = 1;

    // cache block size !% register block size  FAILS
    test_micro[3].num_pts = 100;
    test_micro[3].dim = 4;
    test_micro[3].Bp = 4;
    test_micro[3].Bd = 4;
    test_micro[3].Bpr = 3;
    test_micro[3].Bdr = 1;

    //cache block size = register block size
    test_micro[4].num_pts = 100;
    test_micro[4].dim = 9;
    test_micro[4].Bp = 4;
    test_micro[4].Bd = 4;
    test_micro[4].Bpr = 4;
    test_micro[4].Bdr = 1;

    // number of dimensions != 4
    test_micro[5].num_pts = 10;
    test_micro[5].dim = 6;
    test_micro[5].Bp = 4;
    test_micro[5].Bd = 4;
    test_micro[5].Bpr = 1;
    test_micro[5].Bdr = 1;

    // CONCLUSION: seems to be a problem with num of dimensions

    for (int i = 0; i < 6; ++i) {

        // Can be put in the function, but I do not want to waste the runtime on it.
        assert(test_micro[i].Bp >= test_micro[i].Bpr);
        assert(test_micro[i].Bd >= test_micro[i].Bdr);

        printf("\n\nTest case: %d\n", i);
        printf("num_pts : %d, dim : %d, Bp : %d, Bd : %d, Bpr: %d, Bdr: %d\n", test_micro[i].num_pts, test_micro[i].dim,
               test_micro[i].Bp, test_micro[i].Bd, test_micro[i].Bpr, test_micro[i].Bdr);
        int val = test_micro_mmm_lower_triangular_single(test_micro[i].num_pts, test_micro[i].dim, test_micro[i].Bp,
                                                         test_micro[i].Bd, test_micro[i].Bpr, test_micro[i].Bdr, 0);
        if (val != 0) {
            printf("DIFFERENT MATRICES %d!!!!\n\n", i);
        }
    }
}

//_____________________________ POSTPROCESSING _________________________________________________________

void cache_block_selection(int num_pts, int dim, int num_reps) {
    /**
     * The text output is used to produce box plots
     */

    // Define manually potential block sizes (not optimal)
    int arrayBp[] = {10, 50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000};
    int arrayBd[] = {1, 2, 4, 8, 16};

    // NO BLOCKING
    double res = performance_measure_fast_for_mini_mmm(num_pts, dim, num_reps, num_pts, dim, mini_mmm_lower_triangular);
    printf("'Baseline' %lf\n", res);
    // create a grid
    for (int i = 0; i < 11; i++) {
        if (arrayBp[i] <= num_pts) {
            for (int j = 0; j < 5; j++) {
                if (arrayBd[j] <= dim) {

                    double res = performance_measure_fast_for_mini_mmm(arrayBp[i], arrayBd[j], num_reps, num_pts, dim,
                                                                       mini_mmm_lower_triangular);
                    printf("%d, %d, %d, %d, %lf\n", num_pts, dim, arrayBp[i], arrayBd[j], res);

                }
            }
        } // if
    } // for i

} // cache_block_selection

