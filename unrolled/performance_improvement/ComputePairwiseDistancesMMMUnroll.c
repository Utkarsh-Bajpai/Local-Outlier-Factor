//
// Created by pasca on 15.05.2020.
//

#include <stdlib.h>
#include "../include/metrics.h"
#include "math.h"
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "../../include/tests.h"
#include "../../include/lof_baseline.h"
#include "../include/MMM.h"
#include "../../include/file_utils.h"


typedef long  (* my_fun)(int, int, int, int, const double*, double*);

#define CYCLES_REQ 1e8

/**
 * New computation logic baseline
 **/

double
ComputePairwiseDistancesMMM_baseline(int num_pts, int dim, int B0, int B1, int BK, const double* input_points_ptr,
                                     double* distances_indexed_ptr) {

    double* squared_input = XmallocVectorDouble(num_pts);

    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i < num_pts; i++) {
        double sum = 0;
        for (j = 0; j < dim; j++) {
            sum += input_points_ptr[i * dim + j] * input_points_ptr[i * dim + j];
        }
        squared_input[i] = sum;
    }

    for (i = 0; i < num_pts; i++) {
        for (j = 0; j < i; j++) {
            double sum = 0;
            for (int k = 0; k < dim; k++) {
                sum += input_points_ptr[i * dim + k] * input_points_ptr[j * dim + k];
            }
            mm_res[i * num_pts + j] = sum;
        }
    }


    for (i = 0; i < num_pts; i++) {
        for (j = 0; j < i; j++) {
            distances_indexed_ptr[i * num_pts + j] = sqrt(
                    squared_input[i] - 2 * mm_res[i * num_pts + j] + squared_input[j]);
        }
    }

    free(squared_input);
    free(mm_res);

    return 2 * num_pts * dim + 2.0 * num_pts * (num_pts - 1) * dim / 2 + num_pts * (num_pts - 1) * 4.0 / 2;
}

/**
 * Unroll 4 x 4
 */
void ComputePairwiseDistances_2(int num_pts, int dim, const double* input_points_ptr, double* distances_indexed_ptr) {

    double* squared_input = XmallocVectorDouble(num_pts);

    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i + 3 < num_pts; i += 4) {
        double sum_0_0 = 0;
        double sum_1_0 = 0;
        double sum_2_0 = 0;
        double sum_3_0 = 0;

        double sum_0_1 = 0;
        double sum_1_1 = 0;
        double sum_2_1 = 0;
        double sum_3_1 = 0;

        double sum_0_2 = 0;
        double sum_1_2 = 0;
        double sum_2_2 = 0;
        double sum_3_2 = 0;

        double sum_0_3 = 0;
        double sum_1_3 = 0;
        double sum_2_3 = 0;
        double sum_3_3 = 0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_1_0 += input_points_ptr[base_0 + j + 1] * input_points_ptr[base_0 + j + 1];
            sum_2_0 += input_points_ptr[base_0 + j + 2] * input_points_ptr[base_0 + j + 2];
            sum_3_0 += input_points_ptr[base_0 + j + 3] * input_points_ptr[base_0 + j + 3];

            sum_0_1 += input_points_ptr[base_1 + j] * input_points_ptr[base_1 + j];
            sum_1_1 += input_points_ptr[base_1 + j + 1] * input_points_ptr[base_1 + j + 1];
            sum_2_1 += input_points_ptr[base_1 + j + 2] * input_points_ptr[base_1 + j + 2];
            sum_3_1 += input_points_ptr[base_1 + j + 3] * input_points_ptr[base_1 + j + 3];

            sum_0_2 += input_points_ptr[base_2 + j] * input_points_ptr[base_2 + j];
            sum_1_2 += input_points_ptr[base_2 + j + 1] * input_points_ptr[base_2 + j + 1];
            sum_2_2 += input_points_ptr[base_2 + j + 2] * input_points_ptr[base_2 + j + 2];
            sum_3_2 += input_points_ptr[base_2 + j + 3] * input_points_ptr[base_2 + j + 3];

            sum_0_3 += input_points_ptr[base_3 + j] * input_points_ptr[base_3 + j];
            sum_1_3 += input_points_ptr[base_3 + j + 1] * input_points_ptr[base_3 + j + 1];
            sum_2_3 += input_points_ptr[base_3 + j + 2] * input_points_ptr[base_3 + j + 2];
            sum_3_3 += input_points_ptr[base_3 + j + 3] * input_points_ptr[base_3 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_0_1 += input_points_ptr[base_1 + j] * input_points_ptr[base_1 + j];
            sum_0_2 += input_points_ptr[base_2 + j] * input_points_ptr[base_2 + j];
            sum_0_3 += input_points_ptr[base_3 + j] * input_points_ptr[base_3 + j];
        }
        squared_input[i] = sum_0_0 + sum_1_0 + sum_2_0 + sum_3_0;
        squared_input[i + 1] = sum_0_1 + sum_1_1 + sum_2_1 + sum_3_1;
        squared_input[i + 2] = sum_0_2 + sum_1_2 + sum_2_2 + sum_3_2;
        squared_input[i + 3] = sum_0_3 + sum_1_3 + sum_2_3 + sum_3_3;
    }

    for (; i < num_pts; i++) {
        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;
        int base_0 = i * dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_1 += input_points_ptr[base_0 + j + 1] * input_points_ptr[base_0 + j + 1];
            sum_2 += input_points_ptr[base_0 + j + 2] * input_points_ptr[base_0 + j + 2];
            sum_3 += input_points_ptr[base_0 + j + 3] * input_points_ptr[base_0 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
        }

        squared_input[i] = sum_0 + sum_1 + sum_2 + sum_3;
    }

    mmm_baseline(num_pts, dim, input_points_ptr, mm_res);

    for (i = 0; i < num_pts; i++) {
        for (j = 0; j < i; j++) {
            distances_indexed_ptr[i * num_pts + j] = sqrt(
                    squared_input[i] - 2 * mm_res[i * num_pts + j] + squared_input[j]);
        }
    }

    free(squared_input);
    free(mm_res);
}

/**
 * Unroll outer 8
 */
void ComputePairwiseDistances_3(int num_pts, int dim, const double* input_points_ptr, double* distances_indexed_ptr) {

    double* squared_input = XmallocVectorDouble(num_pts);

    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i + 7 < num_pts; i += 8) {
        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;
        double sum_4 = 0;
        double sum_5 = 0;
        double sum_6 = 0;
        double sum_7 = 0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;
        int base_4 = base_3 + dim;
        int base_5 = base_4 + dim;
        int base_6 = base_5 + dim;
        int base_7 = base_6 + dim;

        for (j = 0; j < dim; j++) {
            sum_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_1 += input_points_ptr[base_1 + j] * input_points_ptr[base_1 + j];
            sum_2 += input_points_ptr[base_2 + j] * input_points_ptr[base_2 + j];
            sum_3 += input_points_ptr[base_3 + j] * input_points_ptr[base_3 + j];

            sum_4 += input_points_ptr[base_4 + j] * input_points_ptr[base_4 + j];
            sum_5 += input_points_ptr[base_5 + j] * input_points_ptr[base_5 + j];
            sum_6 += input_points_ptr[base_6 + j] * input_points_ptr[base_6 + j];
            sum_7 += input_points_ptr[base_7 + j] * input_points_ptr[base_7 + j];
        }
        squared_input[i] = sum_0;
        squared_input[i + 1] = sum_1;
        squared_input[i + 2] = sum_2;
        squared_input[i + 3] = sum_3;

        squared_input[i + 4] = sum_4;
        squared_input[i + 5] = sum_5;
        squared_input[i + 6] = sum_6;
        squared_input[i + 7] = sum_7;
    }

    for (; i < num_pts; i++) {
        double sum = 0;
        for (j = 0; j < dim; j++) {
            sum += input_points_ptr[i * dim + j] * input_points_ptr[i * dim + j];
        }
        squared_input[i] = sum;
    }

    mmm_baseline(num_pts, dim, input_points_ptr, mm_res);

    for (i = 0; i < num_pts; i++) {
        for (j = 0; j < i; j++) {
            distances_indexed_ptr[i * num_pts + j] = sqrt(
                    squared_input[i] - 2 * mm_res[i * num_pts + j] + squared_input[j]);
        }
    }

    free(squared_input);
    free(mm_res);
}

/**
 * Unroll outer 8 and sqrt inner 4
 */
void ComputePairwiseDistances_4(int num_pts, int dim, const double* input_points_ptr, double* distances_indexed_ptr) {
    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i + 7 < num_pts; i += 8) {
        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;
        double sum_4 = 0;
        double sum_5 = 0;
        double sum_6 = 0;
        double sum_7 = 0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;
        int base_4 = base_3 + dim;
        int base_5 = base_4 + dim;
        int base_6 = base_5 + dim;
        int base_7 = base_6 + dim;

        for (j = 0; j < dim; j++) {
            sum_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_1 += input_points_ptr[base_1 + j] * input_points_ptr[base_1 + j];
            sum_2 += input_points_ptr[base_2 + j] * input_points_ptr[base_2 + j];
            sum_3 += input_points_ptr[base_3 + j] * input_points_ptr[base_3 + j];

            sum_4 += input_points_ptr[base_4 + j] * input_points_ptr[base_4 + j];
            sum_5 += input_points_ptr[base_5 + j] * input_points_ptr[base_5 + j];
            sum_6 += input_points_ptr[base_6 + j] * input_points_ptr[base_6 + j];
            sum_7 += input_points_ptr[base_7 + j] * input_points_ptr[base_7 + j];
        }
        squared_input[i] = sum_0;
        squared_input[i + 1] = sum_1;
        squared_input[i + 2] = sum_2;
        squared_input[i + 3] = sum_3;

        squared_input[i + 4] = sum_4;
        squared_input[i + 5] = sum_5;
        squared_input[i + 6] = sum_6;
        squared_input[i + 7] = sum_7;
    }

    for (; i < num_pts; i++) {
        double sum = 0;
        for (j = 0; j < dim; j++) {
            sum += input_points_ptr[i * dim + j] * input_points_ptr[i * dim + j];
        }
        squared_input[i] = sum;
    }


    mmm_baseline(num_pts, dim, input_points_ptr, mm_res);

    for (i = 0; i < num_pts - 1; i++) {
        for (j = i + 1; j + 3 < num_pts; j += 4) {
            double s0 = squared_input[i] + squared_input[j];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - 2 * mm_res[j * num_pts + i]);

            double s1 = squared_input[i] + squared_input[j + 1];
            distances_indexed_ptr[(j + 1) * num_pts + i] = sqrt(s1 - 2 * mm_res[(j + 1) * num_pts + i]);

            double s2 = squared_input[i] + squared_input[j + 2];
            distances_indexed_ptr[(j + 2) * num_pts + i] = sqrt(s2 - 2 * mm_res[(j + 2) * num_pts + i]);

            double s3 = squared_input[i] + squared_input[j + 3];
            distances_indexed_ptr[(j + 3) * num_pts + i] = sqrt(s3 - 2 * mm_res[(j + 3) * num_pts + i]);
        }
        for (; j < num_pts; j++) {
            double s0 = squared_input[i] + squared_input[j];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - 2 * mm_res[j * num_pts + i]);
        }
    }

    free(squared_input);
    free(mm_res);

}

/**
 * Unroll outer 8 and sqrt outer 4
 */
double
ComputePairwiseDistancesMMMUnroll_fastest(int num_pts, int dim, int B0, int B1, int BK, const double* input_points_ptr,
                                          double* distances_indexed_ptr) {

    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i + 7 < num_pts; i += 8) {
        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;
        double sum_4 = 0;
        double sum_5 = 0;
        double sum_6 = 0;
        double sum_7 = 0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;
        int base_4 = base_3 + dim;
        int base_5 = base_4 + dim;
        int base_6 = base_5 + dim;
        int base_7 = base_6 + dim;

        for (j = 0; j < dim; j++) {
            sum_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_1 += input_points_ptr[base_1 + j] * input_points_ptr[base_1 + j];
            sum_2 += input_points_ptr[base_2 + j] * input_points_ptr[base_2 + j];
            sum_3 += input_points_ptr[base_3 + j] * input_points_ptr[base_3 + j];

            sum_4 += input_points_ptr[base_4 + j] * input_points_ptr[base_4 + j];
            sum_5 += input_points_ptr[base_5 + j] * input_points_ptr[base_5 + j];
            sum_6 += input_points_ptr[base_6 + j] * input_points_ptr[base_6 + j];
            sum_7 += input_points_ptr[base_7 + j] * input_points_ptr[base_7 + j];
        }
        squared_input[i] = sum_0;
        squared_input[i + 1] = sum_1;
        squared_input[i + 2] = sum_2;
        squared_input[i + 3] = sum_3;

        squared_input[i + 4] = sum_4;
        squared_input[i + 5] = sum_5;
        squared_input[i + 6] = sum_6;
        squared_input[i + 7] = sum_7;
    }

    for (; i < num_pts; i++) {
        double sum = 0;
        for (j = 0; j < dim; j++) {
            sum += input_points_ptr[i * dim + j] * input_points_ptr[i * dim + j];
        }
        squared_input[i] = sum;
    }

    mmm_unroll_fastest(num_pts, dim, B0, B1, BK, input_points_ptr, mm_res);

    for (i = 0; i + 3 < num_pts - 1; i += 4) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - prod);

            double s1 = squared_input[i + 1] + bla;
            double prod1 = 2 * mm_res[j * num_pts + i + 1];
            if (i + 1 < j) {
                distances_indexed_ptr[j * num_pts + i + 1] = sqrt(s1 - prod1);
            }

            double s2 = squared_input[i + 2] + bla;
            double prod2 = 2 * mm_res[j * num_pts + i + 2];
            if (i + 2 < j) {
                distances_indexed_ptr[j * num_pts + i + 2] = sqrt(s2 - prod2);
            }

            double s3 = squared_input[i + 3] + bla;
            double prod3 = 2 * mm_res[j * num_pts + i + 3];
            if (i + 3 < j) {
                distances_indexed_ptr[j * num_pts + i + 3] = sqrt(s3 - prod3);
            }
        }
    }
    for (; i < num_pts; i++) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - prod);
        }
    }

    free(squared_input);
    free(mm_res);

    return 2 * num_pts * dim + 2.0 * num_pts * (num_pts - 1) * dim / 2 + num_pts * (num_pts - 1) * 4.0 / 2;

}

/**
 * Baseline squaring and dist outer 4
 */
void ComputePairwiseDistances_6(int num_pts, int dim, int B0, int B1, const double* input_points_ptr,
                                double* distances_indexed_ptr) {

    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i < num_pts; i++) {
        double sum = 0;
        for (j = 0; j < dim; j++) {
            sum += input_points_ptr[i * dim + j] * input_points_ptr[i * dim + j];
        }
        squared_input[i] = sum;
    }

    mmm_baseline(num_pts, dim, input_points_ptr, mm_res);

    for (i = 0; i + 3 < num_pts - 1; i += 4) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - prod);

            double s1 = squared_input[i + 1] + bla;
            double prod1 = 2 * mm_res[j * num_pts + i + 1];
            if (i + 1 < j) {
                distances_indexed_ptr[j * num_pts + i + 1] = sqrt(s1 - prod1);
            }

            double s2 = squared_input[i + 2] + bla;
            double prod2 = 2 * mm_res[j * num_pts + i + 2];
            if (i + 2 < j) {
                distances_indexed_ptr[j * num_pts + i + 2] = sqrt(s2 - prod2);
            }

            double s3 = squared_input[i + 3] + bla;
            double prod3 = 2 * mm_res[j * num_pts + i + 3];
            if (i + 3 < j) {
                distances_indexed_ptr[j * num_pts + i + 3] = sqrt(s3 - prod3);
            }
        }
    }
    for (; i < num_pts; i++) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - prod);
        }
    }

    free(squared_input);
    free(mm_res);
}

/**
 * Squaring 4x4 and dist outer by 4
 */

void ComputePairwiseDistances_7(int num_pts, int dim, int B0, int B1, const double* input_points_ptr,
                                double* distances_indexed_ptr) {

    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i + 3 < num_pts; i += 4) {
        double sum_0_0 = 0;
        double sum_1_0 = 0;
        double sum_2_0 = 0;
        double sum_3_0 = 0;

        double sum_0_1 = 0;
        double sum_1_1 = 0;
        double sum_2_1 = 0;
        double sum_3_1 = 0;

        double sum_0_2 = 0;
        double sum_1_2 = 0;
        double sum_2_2 = 0;
        double sum_3_2 = 0;

        double sum_0_3 = 0;
        double sum_1_3 = 0;
        double sum_2_3 = 0;
        double sum_3_3 = 0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_1_0 += input_points_ptr[base_0 + j + 1] * input_points_ptr[base_0 + j + 1];
            sum_2_0 += input_points_ptr[base_0 + j + 2] * input_points_ptr[base_0 + j + 2];
            sum_3_0 += input_points_ptr[base_0 + j + 3] * input_points_ptr[base_0 + j + 3];

            sum_0_1 += input_points_ptr[base_1 + j] * input_points_ptr[base_1 + j];
            sum_1_1 += input_points_ptr[base_1 + j + 1] * input_points_ptr[base_1 + j + 1];
            sum_2_1 += input_points_ptr[base_1 + j + 2] * input_points_ptr[base_1 + j + 2];
            sum_3_1 += input_points_ptr[base_1 + j + 3] * input_points_ptr[base_1 + j + 3];

            sum_0_2 += input_points_ptr[base_2 + j] * input_points_ptr[base_2 + j];
            sum_1_2 += input_points_ptr[base_2 + j + 1] * input_points_ptr[base_2 + j + 1];
            sum_2_2 += input_points_ptr[base_2 + j + 2] * input_points_ptr[base_2 + j + 2];
            sum_3_2 += input_points_ptr[base_2 + j + 3] * input_points_ptr[base_2 + j + 3];

            sum_0_3 += input_points_ptr[base_3 + j] * input_points_ptr[base_3 + j];
            sum_1_3 += input_points_ptr[base_3 + j + 1] * input_points_ptr[base_3 + j + 1];
            sum_2_3 += input_points_ptr[base_3 + j + 2] * input_points_ptr[base_3 + j + 2];
            sum_3_3 += input_points_ptr[base_3 + j + 3] * input_points_ptr[base_3 + j + 3];
        }
        for (; j < dim; j++) {

            sum_0_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_0_1 += input_points_ptr[base_1 + j] * input_points_ptr[base_1 + j];
            sum_0_2 += input_points_ptr[base_2 + j] * input_points_ptr[base_2 + j];
            sum_0_3 += input_points_ptr[base_3 + j] * input_points_ptr[base_3 + j];
        }
        squared_input[i] = sum_0_0 + sum_1_0 + sum_2_0 + sum_3_0;
        squared_input[i + 1] = sum_0_1 + sum_1_1 + sum_2_1 + sum_3_1;
        squared_input[i + 2] = sum_0_2 + sum_1_2 + sum_2_2 + sum_3_2;
        squared_input[i + 3] = sum_0_3 + sum_1_3 + sum_2_3 + sum_3_3;
    }

    for (; i < num_pts; i++) {
        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;
        int base_0 = i * dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
            sum_1 += input_points_ptr[base_0 + j + 1] * input_points_ptr[base_0 + j + 1];
            sum_2 += input_points_ptr[base_0 + j + 2] * input_points_ptr[base_0 + j + 2];
            sum_3 += input_points_ptr[base_0 + j + 3] * input_points_ptr[base_0 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0 += input_points_ptr[base_0 + j] * input_points_ptr[base_0 + j];
        }
    }

    mmm_baseline(num_pts, dim, input_points_ptr, mm_res);

    for (i = 0; i + 3 < num_pts - 1; i += 4) {

        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];

            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - prod);

            double s1 = squared_input[i + 1] + bla;
            double prod1 = 2 * mm_res[j * num_pts + i + 1];
            if (i + 1 < j) {
                distances_indexed_ptr[j * num_pts + i + 1] = sqrt(s1 - prod1);
            }

            double s2 = squared_input[i + 2] + bla;
            double prod2 = 2 * mm_res[j * num_pts + i + 2];
            if (i + 2 < j) {
                distances_indexed_ptr[j * num_pts + i + 2] = sqrt(s2 - prod2);
            }

            double s3 = squared_input[i + 3] + bla;
            double prod3 = 2 * mm_res[j * num_pts + i + 3];
            if (i + 3 < j) {
                distances_indexed_ptr[j * num_pts + i + 3] = sqrt(s3 - prod3);
            }
        }
    }

    for (; i < num_pts; i++) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - prod);
        }
    }

    free(squared_input);
    free(mm_res);
}

/**
 * Baseline squaring and sqrt outer 4 IF strong
 */
void ComputePairwiseDistances_8(int num_pts, int dim, int B0, int B1, const double* input_points_ptr,
                                double* distances_indexed_ptr) {

    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i < num_pts; i++) {
        double sum = 0;
        for (j = 0; j < dim; j++) {
            sum += input_points_ptr[i * dim + j] * input_points_ptr[i * dim + j];
        }
        squared_input[i] = sum;
    }

    mmm_baseline(num_pts, dim, input_points_ptr, mm_res);

    for (i = 0; i + 3 < num_pts - 1; i += 4) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - prod);

            if (i + 1 < j) {
                double s1 = squared_input[i + 1] + bla;
                double prod1 = 2 * mm_res[j * num_pts + i + 1];
                distances_indexed_ptr[j * num_pts + i + 1] = sqrt(s1 - prod1);
            }

            if (i + 2 < j) {
                double s2 = squared_input[i + 2] + bla;
                double prod2 = 2 * mm_res[j * num_pts + i + 2];
                distances_indexed_ptr[j * num_pts + i + 2] = sqrt(s2 - prod2);
            }

            if (i + 3 < j) {
                double s3 = squared_input[i + 3] + bla;
                double prod3 = 2 * mm_res[j * num_pts + i + 3];
                distances_indexed_ptr[j * num_pts + i + 3] = sqrt(s3 - prod3);
            }
        }
    }
    for (; i < num_pts; i++) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            distances_indexed_ptr[j * num_pts + i] = sqrt(s0 - prod);
        }
    }

    free(squared_input);
    free(mm_res);
}

long ComputePairwiseDistancesWrapper(int n, int dim, int B0, int B1, const double* input_points,
                                     double* pairwise_distances) {
    ComputePairwiseDistances(dim, n, input_points, UnrolledEuclideanDistance, pairwise_distances);

    return n * (n - 1) * (4 * dim + 1) / 2;
}

long ComputePairwiseDistancesWrapper1(int n, int dim, int B0, int B1, const double* input_points,
                                      double* pairwise_distances) {
    ComputePairwiseDistancesMMM_baseline(n, dim, B0, B1, 8, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2;
}

long ComputePairwiseDistancesWrapper2(int n, int dim, int B0, int B1, const double* input_points,
                                      double* pairwise_distances) {
    ComputePairwiseDistances_2(n, dim, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2;
}

long ComputePairwiseDistancesWrapper3(int n, int dim, int B0, int B1, const double* input_points,
                                      double* pairwise_distances) {
    ComputePairwiseDistances_3(n, dim, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2;
}

long ComputePairwiseDistancesWrapper4(int n, int dim, int B0, int B1, const double* input_points,
                                      double* pairwise_distances) {
    ComputePairwiseDistances_4(n, dim, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2;
}

long ComputePairwiseDistancesWrapper5(int n, int dim, int B0, int B1, const double* input_points,
                                      double* pairwise_distances) {
    ComputePairwiseDistancesMMMUnroll_fastest(n, dim, B0, B1, 8, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2;
}

long ComputePairwiseDistancesWrapper6(int n, int dim, int B0, int B1, const double* input_points,
                                      double* pairwise_distances) {
    ComputePairwiseDistances_6(n, dim, B0, B1, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2;
}

long ComputePairwiseDistancesWrapper7(int n, int dim, int B0, int B1, const double* input_points,
                                      double* pairwise_distances) {
    ComputePairwiseDistances_7(n, dim, B0, B1, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2;
}

long ComputePairwiseDistancesWrapper8(int n, int dim, int B0, int B1, const double* input_points,
                                      double* pairwise_distances) {
    ComputePairwiseDistances_8(n, dim, B0, B1, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2;
}

#define  NUM_FUNCTIONS 9

int pairwise_distances_driver(int n, int k, int dim, int B0, int B1, int nr_reps) {

    double* input_points = XmallocMatrixDoubleRandom(n, dim);
    double* pairwise_true = XmallocMatrixDouble(n, n);

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &ComputePairwiseDistancesWrapper;
    fun_array[1] = &ComputePairwiseDistancesWrapper1;
    fun_array[2] = &ComputePairwiseDistancesWrapper2;
    fun_array[3] = &ComputePairwiseDistancesWrapper3;
    fun_array[4] = &ComputePairwiseDistancesWrapper4;
    fun_array[5] = &ComputePairwiseDistancesWrapper5;
    fun_array[6] = &ComputePairwiseDistancesWrapper6;
    fun_array[7] = &ComputePairwiseDistancesWrapper7;
    fun_array[8] = &ComputePairwiseDistancesWrapper8;


    char* fun_names[NUM_FUNCTIONS] = {"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"};


    myInt64 start, end;
    double cycles1;
    double flops;

    FILE* results_file = open_with_error_check("../pairwise_distances_matrix.txt", "a");

//    test_matrices(n, dim, input_points);

    ComputePairwiseDistancesWrapper(n, dim, B0, B1, input_points, pairwise_true);

    for (int fun_index = 8; fun_index < NUM_FUNCTIONS; fun_index++) {
        double* pairwise_results = XmallocMatrixDouble(n, n);

        // VERIFICATION : ------------------------------------------------------------------------
        (*fun_array[fun_index])(n, dim, B0, B1, input_points, pairwise_results);

        int ver = test_double_arrays(n * n, 1e-3, pairwise_results, pairwise_true);
        if (ver != 1) {
            printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
//            print_matrices(n, pairwise_true, pairwise_results);
            exit(-1);
        }
        double multiplier = 1;
        double numRuns = 10;

        // Warm-up phase: we determine a number of executions that allows
        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[fun_index])(n, dim, B0, B1, input_points, pairwise_results);
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
                flops = (*fun_array[fun_index])(n, dim, B0, B1, input_points, pairwise_results);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;

            totalCycles += cycles1;
        }

        qsort(cyclesPtr, nr_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[(int) nr_reps / 2 + 1];


        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        printf("%s n:%d d:%d cycles:%lf perf:%lf \n", fun_names[fun_index], n, dim, cycles, perf);

        free(cyclesPtr);
        free(pairwise_results);
        fprintf(results_file, "%s, %d, %d, %lf, %lf\n", fun_names[fun_index], n, dim, cycles, perf);
    }

    printf("-------------\n");
    free(input_points);
    free(pairwise_true);
//    fclose(results_file);
    return 0;
}

