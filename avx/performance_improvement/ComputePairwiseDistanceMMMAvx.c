//
// Created by pasca on 19.05.2020.
//

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "../../include/tests.h"
#include "../../include/file_utils.h"
#include "../../unrolled/include/MMM.h"
#include "../include/MMMAvx.h"
#include "../../unrolled/include/ComputePairwiseDistancesMMMUnroll.h"

typedef double  (* my_fun)(int, int, int, int, int, const double*, double*);

#define CYCLES_REQ 1e8


/**
 * AVX-ing the vector squaring
 */
void AVXComputePairwiseDistances_1(int num_pts, int dim, int B0, int B1, const double* input_points,
                                double* pairwise_distances) {

    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    __m256d sq_norm_0_3, input_0_0_3, input_1_0_3, input_2_0_3, input_3_0_3;
    __m256d squared_0_0_3, squared_1_0_3, squared_2_0_3, squared_3_0_3;

    int j, i;

    for (i = 0; i + 3 < num_pts; i += 4) {

        double sum_0_0 = 0;
        double sum_0_1 = 0;
        double sum_0_2 = 0;
        double sum_0_3 = 0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;

        squared_0_0_3 = _mm256_setzero_pd();
        squared_1_0_3 = _mm256_setzero_pd();
        squared_2_0_3 = _mm256_setzero_pd();
        squared_3_0_3 = _mm256_setzero_pd();

        for (j = 0; j + 3 < dim; j += 4) {
            input_0_0_3 = _mm256_loadu_pd(input_points + base_0 + j);
            input_1_0_3 = _mm256_loadu_pd(input_points + base_1 + j);
            input_2_0_3 = _mm256_loadu_pd(input_points + base_2 + j);
            input_3_0_3 = _mm256_loadu_pd(input_points + base_3 + j);

            squared_0_0_3 = _mm256_fmadd_pd(input_0_0_3, input_0_0_3, squared_0_0_3);
            squared_1_0_3 = _mm256_fmadd_pd(input_1_0_3, input_1_0_3, squared_1_0_3);
            squared_2_0_3 = _mm256_fmadd_pd(input_2_0_3, input_2_0_3, squared_2_0_3);
            squared_3_0_3 = _mm256_fmadd_pd(input_3_0_3, input_3_0_3, squared_3_0_3);
        }

        squared_0_0_3 = _mm256_hadd_pd(squared_0_0_3, squared_1_0_3);
        squared_2_0_3 = _mm256_hadd_pd(squared_2_0_3, squared_3_0_3);
        //sq0,sq1,sq2,sq3 -> sq2,sq3,sq0,sq1
        //need to perform reduction on all these guys
        __m256d perm_sq_0 = _mm256_permute2f128_pd(squared_0_0_3, squared_0_0_3, 0b00000001);
        __m256d perm_sq_2 = _mm256_permute2f128_pd(squared_2_0_3, squared_2_0_3, 0b00000001);

        __m256d half_1 = _mm256_add_pd(squared_0_0_3, perm_sq_0);
        __m256d half_2 = _mm256_add_pd(squared_2_0_3, perm_sq_2);

        //combine the results in the proper order -> 00001100
        sq_norm_0_3 = _mm256_blend_pd(half_1, half_2, 12);

        _mm256_storeu_pd(squared_input + i, sq_norm_0_3);

        for (; j < dim; j++) {
            sum_0_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_0_1 += input_points[base_1 + j] * input_points[base_1 + j];
            sum_0_2 += input_points[base_2 + j] * input_points[base_2 + j];
            sum_0_3 += input_points[base_3 + j] * input_points[base_3 + j];
        }
        squared_input[i] += sum_0_0;
        squared_input[i + 1] += sum_0_1;
        squared_input[i + 2] += sum_0_2;
        squared_input[i + 3] += sum_0_3;

    }

    for (; i < num_pts; i++) {
        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;
        int base_0 = i * dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_1 += input_points[base_0 + j + 1] * input_points[base_0 + j + 1];
            sum_2 += input_points[base_0 + j + 2] * input_points[base_0 + j + 2];
            sum_3 += input_points[base_0 + j + 3] * input_points[base_0 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0 += input_points[base_0 + j] * input_points[base_0 + j];
        }
    }

    mmm_baseline(num_pts, dim, input_points, mm_res);

    for (i = 0; i + 3 < num_pts - 1; i += 4) {

        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];

            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            pairwise_distances[j * num_pts + i] = sqrt(s0 - prod);

            double s1 = squared_input[i + 1] + bla;
            double prod1 = 2 * mm_res[j * num_pts + i + 1];
            if (i + 1 < j) {
                pairwise_distances[j * num_pts + i + 1] = sqrt(s1 - prod1);
            }

            double s2 = squared_input[i + 2] + bla;
            double prod2 = 2 * mm_res[j * num_pts + i + 2];
            if (i + 2 < j) {
                pairwise_distances[j * num_pts + i + 2] = sqrt(s2 - prod2);
            }

            double s3 = squared_input[i + 3] + bla;
            double prod3 = 2 * mm_res[j * num_pts + i + 3];
            if (i + 3 < j) {
                pairwise_distances[j * num_pts + i + 3] = sqrt(s3 - prod3);
            }
        }
    }

    for (; i < num_pts; i++) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            pairwise_distances[j * num_pts + i] = sqrt(s0 - prod);
        }
    }

    free(squared_input);
    free(mm_res);
}

/**
 * AVX-ing distances by 8
 */
void AVXComputePairwiseDistances_2(int num_pts, int dim, int B0, int B1, const double* input_points,
                                double* pairwise_distances) {

    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i + 3 < num_pts; i += 4) {
        double sum_0_0 = 0, sum_1_0 = 0, sum_2_0 = 0, sum_3_0 = 0;
        double sum_0_1 = 0, sum_1_1 = 0, sum_2_1 = 0, sum_3_1 = 0;
        double sum_0_2 = 0, sum_1_2 = 0, sum_2_2 = 0, sum_3_2 = 0;
        double sum_0_3 = 0, sum_1_3 = 0, sum_2_3 = 0, sum_3_3 = 0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_1_0 += input_points[base_0 + j + 1] * input_points[base_0 + j + 1];
            sum_2_0 += input_points[base_0 + j + 2] * input_points[base_0 + j + 2];
            sum_3_0 += input_points[base_0 + j + 3] * input_points[base_0 + j + 3];

            sum_0_1 += input_points[base_1 + j] * input_points[base_1 + j];
            sum_1_1 += input_points[base_1 + j + 1] * input_points[base_1 + j + 1];
            sum_2_1 += input_points[base_1 + j + 2] * input_points[base_1 + j + 2];
            sum_3_1 += input_points[base_1 + j + 3] * input_points[base_1 + j + 3];

            sum_0_2 += input_points[base_2 + j] * input_points[base_2 + j];
            sum_1_2 += input_points[base_2 + j + 1] * input_points[base_2 + j + 1];
            sum_2_2 += input_points[base_2 + j + 2] * input_points[base_2 + j + 2];
            sum_3_2 += input_points[base_2 + j + 3] * input_points[base_2 + j + 3];

            sum_0_3 += input_points[base_3 + j] * input_points[base_3 + j];
            sum_1_3 += input_points[base_3 + j + 1] * input_points[base_3 + j + 1];
            sum_2_3 += input_points[base_3 + j + 2] * input_points[base_3 + j + 2];
            sum_3_3 += input_points[base_3 + j + 3] * input_points[base_3 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_0_1 += input_points[base_1 + j] * input_points[base_1 + j];
            sum_0_2 += input_points[base_2 + j] * input_points[base_2 + j];
            sum_0_3 += input_points[base_3 + j] * input_points[base_3 + j];
        }
        squared_input[i] = sum_0_0 + sum_1_0 + sum_2_0 + sum_3_0;
        squared_input[i + 1] = sum_0_1 + sum_1_1 + sum_2_1 + sum_3_1;
        squared_input[i + 2] = sum_0_2 + sum_1_2 + sum_2_2 + sum_3_2;
        squared_input[i + 3] = sum_0_3 + sum_1_3 + sum_2_3 + sum_3_3;
    }
    for (; i < num_pts; i++) {
        double sum_0 = 0, sum_1 = 0, sum_2 = 0, sum_3 = 0;
        int base_0 = i * dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_1 += input_points[base_0 + j + 1] * input_points[base_0 + j + 1];
            sum_2 += input_points[base_0 + j + 2] * input_points[base_0 + j + 2];
            sum_3 += input_points[base_0 + j + 3] * input_points[base_0 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0 += input_points[base_0 + j] * input_points[base_0 + j];
        }
    }

    mmm_baseline(num_pts, dim, input_points, mm_res);
    __m256d minus_twos = _mm256_set_pd(-2.0, -2.0, -2.0, -2.0);

    for (i = 0; i + 7 < num_pts - 1; i += 8) {
        __m256d sq_i_0_3 = _mm256_loadu_pd(squared_input + i);
        __m256d sq_i_4_7 = _mm256_loadu_pd(squared_input + i + 4);

        for (j = i + 1; j < num_pts; j++) {
            int base = j * num_pts;

            __m256d sq_j = _mm256_set1_pd(squared_input[j]);

            __m256d mm_j_i_0_3 = _mm256_loadu_pd(mm_res + base + i);
            __m256d mm_j_i_4_7 = _mm256_loadu_pd(mm_res + base + i + 4);

            __m256d results_i_0_3_j = _mm256_add_pd(sq_j, sq_i_0_3);
            results_i_0_3_j = _mm256_fmadd_pd(minus_twos, mm_j_i_0_3, results_i_0_3_j);
            results_i_0_3_j = _mm256_sqrt_pd(results_i_0_3_j);
            _mm256_storeu_pd(pairwise_distances + base + i, results_i_0_3_j);

            __m256d results_i_4_7_j = _mm256_add_pd(sq_j, sq_i_4_7);
            results_i_4_7_j = _mm256_fmadd_pd(minus_twos, mm_j_i_4_7, results_i_4_7_j);
            results_i_4_7_j = _mm256_sqrt_pd(results_i_4_7_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 4, results_i_4_7_j);
        }
    }
    for (; i + 3 < num_pts - 1; i += 4) {
        __m256d sq_i_0_3 = _mm256_loadu_pd(squared_input + i);

        for (j = i + 1; j < num_pts; j++) {
            int base = j * num_pts;

            __m256d sq_j = _mm256_set1_pd(squared_input[j]);

            __m256d mm_j_i_0_3 = _mm256_loadu_pd(mm_res + base + i);

            __m256d results_i_0_3_j = _mm256_add_pd(sq_j, sq_i_0_3);
            results_i_0_3_j = _mm256_fmadd_pd(minus_twos, mm_j_i_0_3, results_i_0_3_j);
            results_i_0_3_j = _mm256_sqrt_pd(results_i_0_3_j);
            _mm256_storeu_pd(pairwise_distances + base + i, results_i_0_3_j);
        }
    }
    for (; i < num_pts; i++) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            pairwise_distances[j * num_pts + i] = sqrt(s0 - prod);
        }
    }

    free(squared_input);
    free(mm_res);
}


/**
 * AVX-ing distances by 16
 */
void AVXComputePairwiseDistances_3(int num_pts, int dim, int B0, int B1, const double* input_points,
                                double* pairwise_distances) {

    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i + 3 < num_pts; i += 4) {
        double sum_0_0 = 0, sum_1_0 = 0, sum_2_0 = 0, sum_3_0 = 0;
        double sum_0_1 = 0, sum_1_1 = 0, sum_2_1 = 0, sum_3_1 = 0;
        double sum_0_2 = 0, sum_1_2 = 0, sum_2_2 = 0, sum_3_2 = 0;
        double sum_0_3 = 0, sum_1_3 = 0, sum_2_3 = 0, sum_3_3 = 0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_1_0 += input_points[base_0 + j + 1] * input_points[base_0 + j + 1];
            sum_2_0 += input_points[base_0 + j + 2] * input_points[base_0 + j + 2];
            sum_3_0 += input_points[base_0 + j + 3] * input_points[base_0 + j + 3];

            sum_0_1 += input_points[base_1 + j] * input_points[base_1 + j];
            sum_1_1 += input_points[base_1 + j + 1] * input_points[base_1 + j + 1];
            sum_2_1 += input_points[base_1 + j + 2] * input_points[base_1 + j + 2];
            sum_3_1 += input_points[base_1 + j + 3] * input_points[base_1 + j + 3];

            sum_0_2 += input_points[base_2 + j] * input_points[base_2 + j];
            sum_1_2 += input_points[base_2 + j + 1] * input_points[base_2 + j + 1];
            sum_2_2 += input_points[base_2 + j + 2] * input_points[base_2 + j + 2];
            sum_3_2 += input_points[base_2 + j + 3] * input_points[base_2 + j + 3];

            sum_0_3 += input_points[base_3 + j] * input_points[base_3 + j];
            sum_1_3 += input_points[base_3 + j + 1] * input_points[base_3 + j + 1];
            sum_2_3 += input_points[base_3 + j + 2] * input_points[base_3 + j + 2];
            sum_3_3 += input_points[base_3 + j + 3] * input_points[base_3 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_0_1 += input_points[base_1 + j] * input_points[base_1 + j];
            sum_0_2 += input_points[base_2 + j] * input_points[base_2 + j];
            sum_0_3 += input_points[base_3 + j] * input_points[base_3 + j];
        }
        squared_input[i] = sum_0_0 + sum_1_0 + sum_2_0 + sum_3_0;
        squared_input[i + 1] = sum_0_1 + sum_1_1 + sum_2_1 + sum_3_1;
        squared_input[i + 2] = sum_0_2 + sum_1_2 + sum_2_2 + sum_3_2;
        squared_input[i + 3] = sum_0_3 + sum_1_3 + sum_2_3 + sum_3_3;
    }
    for (; i < num_pts; i++) {
        double sum_0 = 0, sum_1 = 0, sum_2 = 0, sum_3 = 0;
        int base_0 = i * dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_1 += input_points[base_0 + j + 1] * input_points[base_0 + j + 1];
            sum_2 += input_points[base_0 + j + 2] * input_points[base_0 + j + 2];
            sum_3 += input_points[base_0 + j + 3] * input_points[base_0 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0 += input_points[base_0 + j] * input_points[base_0 + j];
        }
    }

    mmm_baseline(num_pts, dim, input_points, mm_res);
    __m256d minus_twos = _mm256_set_pd(-2.0, -2.0, -2.0, -2.0);

    for (i = 0; i + 15 < num_pts - 1; i += 16) {
        __m256d sq_i_0_3 = _mm256_loadu_pd(squared_input + i);
        __m256d sq_i_4_7 = _mm256_loadu_pd(squared_input + i + 4);
        __m256d sq_i_8_11 = _mm256_loadu_pd(squared_input + i + 8);
        __m256d sq_i_12_15 = _mm256_loadu_pd(squared_input + i + 12);

        for (j = i + 1; j < num_pts; j++) {
            int base = j * num_pts;

            __m256d sq_j = _mm256_set1_pd(squared_input[j]);

            __m256d mm_j_i_0_3 = _mm256_loadu_pd(mm_res + base + i);
            __m256d mm_j_i_4_7 = _mm256_loadu_pd(mm_res + base + i + 4);
            __m256d mm_j_i_8_11 = _mm256_loadu_pd(mm_res + base + i + 8);
            __m256d mm_j_i_12_15 = _mm256_loadu_pd(mm_res + base + i + 12);

            __m256d results_i_0_3_j = _mm256_add_pd(sq_j, sq_i_0_3);
            results_i_0_3_j = _mm256_fmadd_pd(minus_twos, mm_j_i_0_3, results_i_0_3_j);
            results_i_0_3_j = _mm256_sqrt_pd(results_i_0_3_j);
            _mm256_storeu_pd(pairwise_distances + base + i, results_i_0_3_j);

            __m256d results_i_4_7_j = _mm256_add_pd(sq_j, sq_i_4_7);
            results_i_4_7_j = _mm256_fmadd_pd(minus_twos, mm_j_i_4_7, results_i_4_7_j);
            results_i_4_7_j = _mm256_sqrt_pd(results_i_4_7_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 4, results_i_4_7_j);

            __m256d results_i_8_11_j = _mm256_add_pd(sq_j, sq_i_8_11);
            results_i_8_11_j = _mm256_fmadd_pd(minus_twos, mm_j_i_8_11, results_i_8_11_j);
            results_i_8_11_j = _mm256_sqrt_pd(results_i_8_11_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 8, results_i_8_11_j);

            __m256d results_i_12_15_j = _mm256_add_pd(sq_j, sq_i_12_15);
            results_i_12_15_j = _mm256_fmadd_pd(minus_twos, mm_j_i_12_15, results_i_12_15_j);
            results_i_12_15_j = _mm256_sqrt_pd(results_i_12_15_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 12, results_i_12_15_j);
        }
    }

    for (; i + 7 < num_pts - 1; i += 8) {
        __m256d sq_i_0_3 = _mm256_loadu_pd(squared_input + i);
        __m256d sq_i_4_7 = _mm256_loadu_pd(squared_input + i + 4);

        for (j = i + 1; j < num_pts; j++) {
            int base = j * num_pts;

            __m256d sq_j = _mm256_set1_pd(squared_input[j]);

            __m256d mm_j_i_0_3 = _mm256_loadu_pd(mm_res + base + i);
            __m256d mm_j_i_4_7 = _mm256_loadu_pd(mm_res + base + i + 4);

            __m256d results_i_0_3_j = _mm256_add_pd(sq_j, sq_i_0_3);
            results_i_0_3_j = _mm256_fmadd_pd(minus_twos, mm_j_i_0_3, results_i_0_3_j);
            results_i_0_3_j = _mm256_sqrt_pd(results_i_0_3_j);
            _mm256_storeu_pd(pairwise_distances + base + i, results_i_0_3_j);

            __m256d results_i_4_7_j = _mm256_add_pd(sq_j, sq_i_4_7);
            results_i_4_7_j = _mm256_fmadd_pd(minus_twos, mm_j_i_4_7, results_i_4_7_j);
            results_i_4_7_j = _mm256_sqrt_pd(results_i_4_7_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 4, results_i_4_7_j);
        }
    }
    for (; i + 3 < num_pts - 1; i += 4) {
        __m256d sq_i_0_3 = _mm256_loadu_pd(squared_input + i);

        for (j = i + 1; j < num_pts; j++) {
            int base = j * num_pts;

            __m256d sq_j = _mm256_set1_pd(squared_input[j]);

            __m256d mm_j_i_0_3 = _mm256_loadu_pd(mm_res + base + i);

            __m256d results_i_0_3_j = _mm256_add_pd(sq_j, sq_i_0_3);
            results_i_0_3_j = _mm256_fmadd_pd(minus_twos, mm_j_i_0_3, results_i_0_3_j);
            results_i_0_3_j = _mm256_sqrt_pd(results_i_0_3_j);
            _mm256_storeu_pd(pairwise_distances + base + i, results_i_0_3_j);
        }
    }
    for (; i < num_pts; i++) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            pairwise_distances[j * num_pts + i] = sqrt(s0 - prod);
        }
    }

    free(squared_input);
    free(mm_res);
}

/**
 * Distance AVX by 16 and fastest MMM
 */
double ComputePairwiseDistancesMMMAvx_Fastest(int num_pts, int dim, int B0, int B1, int BK, const double* input_points,
                                              double* pairwise_distances) {


    double* squared_input = XmallocVectorDouble(num_pts);
    double* mm_res = XmallocMatrixDouble(num_pts, num_pts);

    int j, i;

    for (i = 0; i + 3 < num_pts; i += 4) {
        double sum_0_0 = 0.0, sum_1_0 = 0.0, sum_2_0 = 0.0, sum_3_0 = 0.0;
        double sum_0_1 = 0.0, sum_1_1 = 0.0, sum_2_1 = 0.0, sum_3_1 = 0.0;
        double sum_0_2 = 0.0, sum_1_2 = 0.0, sum_2_2 = 0.0, sum_3_2 = 0.0;
        double sum_0_3 = 0.0, sum_1_3 = 0.0, sum_2_3 = 0.0, sum_3_3 = 0.0;

        int base_0 = i * dim;
        int base_1 = base_0 + dim;
        int base_2 = base_1 + dim;
        int base_3 = base_2 + dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_1_0 += input_points[base_0 + j + 1] * input_points[base_0 + j + 1];
            sum_2_0 += input_points[base_0 + j + 2] * input_points[base_0 + j + 2];
            sum_3_0 += input_points[base_0 + j + 3] * input_points[base_0 + j + 3];

            sum_0_1 += input_points[base_1 + j] * input_points[base_1 + j];
            sum_1_1 += input_points[base_1 + j + 1] * input_points[base_1 + j + 1];
            sum_2_1 += input_points[base_1 + j + 2] * input_points[base_1 + j + 2];
            sum_3_1 += input_points[base_1 + j + 3] * input_points[base_1 + j + 3];

            sum_0_2 += input_points[base_2 + j] * input_points[base_2 + j];
            sum_1_2 += input_points[base_2 + j + 1] * input_points[base_2 + j + 1];
            sum_2_2 += input_points[base_2 + j + 2] * input_points[base_2 + j + 2];
            sum_3_2 += input_points[base_2 + j + 3] * input_points[base_2 + j + 3];

            sum_0_3 += input_points[base_3 + j] * input_points[base_3 + j];
            sum_1_3 += input_points[base_3 + j + 1] * input_points[base_3 + j + 1];
            sum_2_3 += input_points[base_3 + j + 2] * input_points[base_3 + j + 2];
            sum_3_3 += input_points[base_3 + j + 3] * input_points[base_3 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_0_1 += input_points[base_1 + j] * input_points[base_1 + j];
            sum_0_2 += input_points[base_2 + j] * input_points[base_2 + j];
            sum_0_3 += input_points[base_3 + j] * input_points[base_3 + j];
        }
        squared_input[i] = sum_0_0 + sum_1_0 + sum_2_0 + sum_3_0;
        squared_input[i + 1] = sum_0_1 + sum_1_1 + sum_2_1 + sum_3_1;
        squared_input[i + 2] = sum_0_2 + sum_1_2 + sum_2_2 + sum_3_2;
        squared_input[i + 3] = sum_0_3 + sum_1_3 + sum_2_3 + sum_3_3;
    }
    for (; i < num_pts; i++) {
        double sum_0 = 0.0, sum_1 = 0.0, sum_2 = 0.0, sum_3 = 0.0;
        int base_0 = i * dim;

        for (j = 0; j + 3 < dim; j += 4) {
            sum_0 += input_points[base_0 + j] * input_points[base_0 + j];
            sum_1 += input_points[base_0 + j + 1] * input_points[base_0 + j + 1];
            sum_2 += input_points[base_0 + j + 2] * input_points[base_0 + j + 2];
            sum_3 += input_points[base_0 + j + 3] * input_points[base_0 + j + 3];
        }
        for (; j < dim; j++) {
            sum_0 += input_points[base_0 + j] * input_points[base_0 + j];
        }
    }

    mmm_avx_fastest(num_pts, dim, B0, B1, BK, input_points, mm_res);

    __m256d minus_twos = _mm256_set_pd(-2.0, -2.0, -2.0, -2.0);

    for (i = 0; i + 15 < num_pts - 1; i += 16) {
        __m256d sq_i_0_3 = _mm256_loadu_pd(squared_input + i);
        __m256d sq_i_4_7 = _mm256_loadu_pd(squared_input + i + 4);
        __m256d sq_i_8_11 = _mm256_loadu_pd(squared_input + i + 8);
        __m256d sq_i_12_15 = _mm256_loadu_pd(squared_input + i + 12);

        for (j = i + 1; j < num_pts; j++) {
            int base = j * num_pts;

            __m256d sq_j = _mm256_set1_pd(squared_input[j]);

            __m256d mm_j_i_0_3 = _mm256_loadu_pd(mm_res + base + i);
            __m256d mm_j_i_4_7 = _mm256_loadu_pd(mm_res + base + i + 4);
            __m256d mm_j_i_8_11 = _mm256_loadu_pd(mm_res + base + i + 8);
            __m256d mm_j_i_12_15 = _mm256_loadu_pd(mm_res + base + i + 12);

            __m256d results_i_0_3_j = _mm256_add_pd(sq_j, sq_i_0_3);
            results_i_0_3_j = _mm256_fmadd_pd(minus_twos, mm_j_i_0_3, results_i_0_3_j);
            results_i_0_3_j = _mm256_sqrt_pd(results_i_0_3_j);
            _mm256_storeu_pd(pairwise_distances + base + i, results_i_0_3_j);

            __m256d results_i_4_7_j = _mm256_add_pd(sq_j, sq_i_4_7);
            results_i_4_7_j = _mm256_fmadd_pd(minus_twos, mm_j_i_4_7, results_i_4_7_j);
            results_i_4_7_j = _mm256_sqrt_pd(results_i_4_7_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 4, results_i_4_7_j);

            __m256d results_i_8_11_j = _mm256_add_pd(sq_j, sq_i_8_11);
            results_i_8_11_j = _mm256_fmadd_pd(minus_twos, mm_j_i_8_11, results_i_8_11_j);
            results_i_8_11_j = _mm256_sqrt_pd(results_i_8_11_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 8, results_i_8_11_j);

            __m256d results_i_12_15_j = _mm256_add_pd(sq_j, sq_i_12_15);
            results_i_12_15_j = _mm256_fmadd_pd(minus_twos, mm_j_i_12_15, results_i_12_15_j);
            results_i_12_15_j = _mm256_sqrt_pd(results_i_12_15_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 12, results_i_12_15_j);
        }
    }

    for (; i + 7 < num_pts - 1; i += 8) {
        __m256d sq_i_0_3 = _mm256_loadu_pd(squared_input + i);
        __m256d sq_i_4_7 = _mm256_loadu_pd(squared_input + i + 4);

        for (j = i + 1; j < num_pts; j++) {
            int base = j * num_pts;

            __m256d sq_j = _mm256_set1_pd(squared_input[j]);

            __m256d mm_j_i_0_3 = _mm256_loadu_pd(mm_res + base + i);
            __m256d mm_j_i_4_7 = _mm256_loadu_pd(mm_res + base + i + 4);

            __m256d results_i_0_3_j = _mm256_add_pd(sq_j, sq_i_0_3);
            results_i_0_3_j = _mm256_fmadd_pd(minus_twos, mm_j_i_0_3, results_i_0_3_j);
            results_i_0_3_j = _mm256_sqrt_pd(results_i_0_3_j);
            _mm256_storeu_pd(pairwise_distances + base + i, results_i_0_3_j);

            __m256d results_i_4_7_j = _mm256_add_pd(sq_j, sq_i_4_7);
            results_i_4_7_j = _mm256_fmadd_pd(minus_twos, mm_j_i_4_7, results_i_4_7_j);
            results_i_4_7_j = _mm256_sqrt_pd(results_i_4_7_j);
            _mm256_storeu_pd(pairwise_distances + base + i + 4, results_i_4_7_j);
        }
    }
    for (; i + 3 < num_pts - 1; i += 4) {
        __m256d sq_i_0_3 = _mm256_loadu_pd(squared_input + i);

        for (j = i + 1; j < num_pts; j++) {
            int base = j * num_pts;

            __m256d sq_j = _mm256_set1_pd(squared_input[j]);

            __m256d mm_j_i_0_3 = _mm256_loadu_pd(mm_res + base + i);

            __m256d results_i_0_3_j = _mm256_add_pd(sq_j, sq_i_0_3);
            results_i_0_3_j = _mm256_fmadd_pd(minus_twos, mm_j_i_0_3, results_i_0_3_j);
            results_i_0_3_j = _mm256_sqrt_pd(results_i_0_3_j);
            _mm256_storeu_pd(pairwise_distances + base + i, results_i_0_3_j);
        }
    }
    for (; i < num_pts; i++) {
        for (j = i + 1; j < num_pts; j++) {
            double bla = squared_input[j];
            double s0 = squared_input[i] + bla;
            double prod = 2 * mm_res[j * num_pts + i];
            pairwise_distances[j * num_pts + i] = sqrt(s0 - prod);
        }
    }

    free(squared_input);
    free(mm_res);

    return 2.0 * num_pts * dim + 2.0 * num_pts * (num_pts - 1.0) * dim / 2.0 + num_pts * (num_pts - 1.0) * 4.0 / 2.0;

}


double AVXComputePairwiseDistancesWrapper(int n, int dim, int B0, int B1, int BK, const double* input_points,
                                       double* pairwise_distances) {
    ComputePairwiseDistancesMMMUnroll_Fastest(n, dim, B0, B1, BK, input_points, pairwise_distances);

    return n * (n - 1) * 2 * dim + 2.0 * n * (n - 1) * dim / 2 + n * (n - 1) * 4 / 2.0;
}

double AVXComputePairwiseDistancesWrapper1(int n, int dim, int B0, int B1, int BK, const double* input_points,
                                        double* pairwise_distances) {
    AVXComputePairwiseDistances_1(n, dim, B0, B1, input_points, pairwise_distances);

    return n * (n - 1) * 2.0 * dim + 2.0 * n * (n - 1) * dim / 2.0 + n * (n - 1) * 4 / 2.0;
}

double AVXComputePairwiseDistancesWrapper2(int n, int dim, int B0, int B1, int BK, const double* input_points,
                                        double* pairwise_distances) {
    AVXComputePairwiseDistances_2(n, dim, B0, B1, input_points, pairwise_distances);

    return n * (n - 1) * 2.0 * dim + 2.0 * n * (n - 1) * dim / 2.0 + n * (n - 1) * 4 / 2.0;
}

double AVXComputePairwiseDistancesWrapper3(int n, int dim, int B0, int B1, int BK, const double* input_points,
                                        double* pairwise_distances) {
    AVXComputePairwiseDistances_3(n, dim, B0, B1, input_points, pairwise_distances);

    return n * (n - 1) * 2.0 * dim + 2.0 * n * (n - 1) * dim / 2.0 + n * (n - 1) * 4 / 2.0;
}

double AVXComputePairwiseDistancesWrapper4(int n, int dim, int B0, int B1, int BK, const double* input_points,
                                        double* pairwise_distances) {
    ComputePairwiseDistancesMMMAvx_Fastest(n, dim, B0, B1, BK, input_points, pairwise_distances);

    return n * (n - 1) * 2.0 * dim + 2.0 * n * (n - 1) * dim / 2.0 + n * (n - 1) * 4 / 2.0;
}


#define  NUM_FUNCTIONS 5

int avx_pairwise_distances_driver(int n, int k, int dim, int B0, int B1, int BK, int nr_reps) {

    double* input_points = XmallocMatrixDoubleRandom(n, dim);
    double* pairwise_true = XmallocMatrixDouble(n, n);

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &AVXComputePairwiseDistancesWrapper;
    fun_array[1] = &AVXComputePairwiseDistancesWrapper1;
    fun_array[2] = &AVXComputePairwiseDistancesWrapper2;
    fun_array[3] = &AVXComputePairwiseDistancesWrapper3;
    fun_array[4] = &AVXComputePairwiseDistancesWrapper4;

    char* fun_names[NUM_FUNCTIONS] = {"Unrolled baseline", "AVX sq norm", "AVX distance 8", "AVX distance 16",
                                      "MM AVX distance 16", "MM AVX distance 16x2", "V7", "V8", "V8", "V9"};

    myInt64 start, end;
    double cycles1;
    double flops;

//    FILE* results_file = open_with_error_check("../pairwise_distances_matri_avx.txt", "a");

    AVXComputePairwiseDistancesWrapper(n, dim, B0, B1, BK, input_points, pairwise_true);

    for (int fun_index = 0; fun_index < NUM_FUNCTIONS; fun_index++) {
        double* pairwise_results = XmallocMatrixDouble(n, n);

        // VERIFICATION : ------------------------------------------------------------------------
        (*fun_array[fun_index])(n, dim, B0, B1, BK, input_points, pairwise_results);

        int ver = test_double_matrices(n, 1e-3, pairwise_results, pairwise_true);
        if (ver != 1) {
            print_matrices(n, pairwise_true, pairwise_results);
            printf("\nRESULTS ARE DIFFERENT FROM BASELINE! at index %d\n", fun_index);
            exit(-1);
        }
        double multiplier = 1;
        double numRuns = 10;

        // Warm-up phase: we determine a number of executions that allows
        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[fun_index])(n, dim, B0, B1, BK, input_points, pairwise_results);
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
                flops = (*fun_array[fun_index])(n, dim, B0, B1, BK, input_points, pairwise_results);
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
//        fprintf(results_file, "%s, %d, %d, %lf, %lf\n", fun_names[fun_index], n, dim, cycles, perf);
    }

    printf("-------------\n");
    free(input_points);
    free(pairwise_true);
//    fclose(results_file);
    return 0;
}

