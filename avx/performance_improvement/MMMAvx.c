//
// Created by pasca on 19.05.2020.
//


#include <immintrin.h>
#include "../../include/utils.h"
#include "stdlib.h"
#include "../../include/tsc_x86.h"
#include "../../include/tests.h"
#include "math.h"
#include "../../unrolled/include/MMM.h"
#include "../../include/file_utils.h"

typedef double (* my_fun)(int, int, int, int, int, const double*, double*);

/**
 * FMA AVX 2_4_4
 */
double avx_micro_2_4_4(int n, int dim, int B0, int B1, int BK, const double* input, double* result) {
    int i, j, i1, j1, k, k1, i2, j2, k2, i3, j3;
    double sum;
    int BK2 = 4;
    int B2i = 2;
    int B2j = 4;


    for (i = 0; i < n - B0; i += B0) {
        for (j = 0; j < i - B0; j += B0) {
            for (k = 0; k < dim - BK; k += BK) {

                for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
                    for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                        for (i2 = i1; i2 + B2i <= i1 + B1; i2 += B2i) {
                            for (j2 = j1; j2 + B2j <= j1 + B1; j2 += B2j) {
                                __m256d p_00 = _mm256_setzero_pd();
                                __m256d p_01 = _mm256_setzero_pd();
                                __m256d p_10 = _mm256_setzero_pd();
                                __m256d p_11 = _mm256_setzero_pd();
                                __m256d p_02 = _mm256_setzero_pd();
                                __m256d p_03 = _mm256_setzero_pd();
                                __m256d p_12 = _mm256_setzero_pd();
                                __m256d p_13 = _mm256_setzero_pd();

                                __m256d final_i_0_j = _mm256_loadu_pd(result + i2 * n + j2);
                                __m256d final_i_1_j = _mm256_loadu_pd(result + (i2 + 1) * n + j2);

                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    __m256d input_j_0_0_3 = _mm256_loadu_pd(input + j2 * dim + k1);
                                    __m256d input_j_1_0_3 = _mm256_loadu_pd(input + (j2 + 1) * dim + k1);
                                    __m256d input_i_0_0_3 = _mm256_loadu_pd(input + i2 * dim + k1);
                                    __m256d input_i_1_0_3 = _mm256_loadu_pd(input + (i2 + 1) * dim + k1);
                                    __m256d input_j_2_0_3 = _mm256_loadu_pd(input + (j2 + 2) * dim + k1);
                                    __m256d input_j_3_0_3 = _mm256_loadu_pd(input + (j2 + 3) * dim + k1);


                                    p_00 = _mm256_fmadd_pd(input_i_0_0_3, input_j_0_0_3, p_00);
                                    p_01 = _mm256_fmadd_pd(input_i_0_0_3, input_j_1_0_3, p_01);
                                    p_10 = _mm256_fmadd_pd(input_i_1_0_3, input_j_0_0_3, p_10);
                                    p_11 = _mm256_fmadd_pd(input_i_1_0_3, input_j_1_0_3, p_11);
                                    p_02 = _mm256_fmadd_pd(input_i_0_0_3, input_j_2_0_3, p_02);
                                    p_03 = _mm256_fmadd_pd(input_i_0_0_3, input_j_3_0_3, p_03);
                                    p_12 = _mm256_fmadd_pd(input_i_1_0_3, input_j_2_0_3, p_12);
                                    p_13 = _mm256_fmadd_pd(input_i_1_0_3, input_j_3_0_3, p_13);
                                }
                                __m256d hadd_00_01 = _mm256_hadd_pd(p_00, p_01);
                                __m256d hadd_10_11 = _mm256_hadd_pd(p_10, p_11);
                                __m256d hadd_02_03 = _mm256_hadd_pd(p_02, p_03);
                                __m256d hadd_12_13 = _mm256_hadd_pd(p_12, p_13);

                                __m256d perm_hadd_00_01 = _mm256_permute2f128_pd(hadd_00_01, hadd_00_01, 0b00000001);
                                __m256d perm_hadd_10_11 = _mm256_permute2f128_pd(hadd_10_11, hadd_10_11, 0b00000001);
                                __m256d perm_hadd_02_03 = _mm256_permute2f128_pd(hadd_02_03, hadd_02_03, 0b00000001);
                                __m256d perm_hadd_12_13 = _mm256_permute2f128_pd(hadd_12_13, hadd_12_13, 0b00000001);

                                __m256d half_i_0_1 = _mm256_add_pd(hadd_00_01, perm_hadd_00_01);
                                __m256d half_i_0_2 = _mm256_add_pd(hadd_02_03, perm_hadd_02_03);
                                __m256d half_i_1_1 = _mm256_add_pd(hadd_10_11, perm_hadd_10_11);
                                __m256d half_i_1_2 = _mm256_add_pd(hadd_12_13, perm_hadd_12_13);

                                half_i_0_1 = _mm256_blend_pd(half_i_0_1, half_i_0_2, 0b00001100);
                                half_i_1_1 = _mm256_blend_pd(half_i_1_1, half_i_1_2, 0b00001100);

                                final_i_0_j = _mm256_add_pd(half_i_0_1, final_i_0_j);
                                final_i_1_j = _mm256_add_pd(half_i_1_1, final_i_1_j);

                                _mm256_storeu_pd(result + i2 * n + j2, final_i_0_j);
                                _mm256_storeu_pd(result + (i2 + 1) * n + j2, final_i_1_j);
                            }
                            for (j3 = j2; j3 < j1 + B1; j3++) {
                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    result[i2 * n + j3] += input[i2 * dim + k1] * input[j3 * dim + k1];
                                    result[i2 * n + j3] += input[i2 * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                    result[i2 * n + j3] += input[i2 * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                    result[i2 * n + j3] += input[i2 * dim + k1 + 3] * input[j3 * dim + k1 + 3];

                                    result[(i2 + 1) * n + j3] += input[(i2 + 1) * dim + k1] * input[j3 * dim + k1];
                                    result[(i2 + 1) * n + j3] +=
                                            input[(i2 + 1) * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                    result[(i2 + 1) * n + j3] +=
                                            input[(i2 + 1) * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                    result[(i2 + 1) * n + j3] +=
                                            input[(i2 + 1) * dim + k1 + 3] * input[j3 * dim + k1 + 3];
                                }
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


/**
 * FMA AVX 1_4_4
 */
double avx_micro_1_4_4(int n, int dim, int B0, int B1, int BK, const double* input, double* result) {
    int i, j, i1, j1, k, k1, i2, j2, k2, i3, j3;
    double sum;
    int BK2 = 4;
    int B2j = 4;

    for (i = 0; i < n - B0; i += B0) {
        for (j = 0; j < i - B0; j += B0) {
            for (k = 0; k < dim - BK; k += BK) {

                for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
                    for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                        for (i2 = i1; i2 < i1 + B1; i2 += 1) {
                            for (j2 = j1; j2 + B2j <= j1 + B1; j2 += B2j) {

                                __m256d p_00 = _mm256_setzero_pd();
                                __m256d p_01 = _mm256_setzero_pd();
                                __m256d p_02 = _mm256_setzero_pd();
                                __m256d p_03 = _mm256_setzero_pd();
                                __m256d final_i_0_j = _mm256_loadu_pd(result + i2 * n + j2);

                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    __m256d input_i_0_0_3 = _mm256_loadu_pd(input + i2 * dim + k1);

                                    __m256d input_j_0_0_3 = _mm256_loadu_pd(input + j2 * dim + k1);
                                    __m256d input_j_1_0_3 = _mm256_loadu_pd(input + (j2 + 1) * dim + k1);
                                    __m256d input_j_2_0_3 = _mm256_loadu_pd(input + (j2 + 2) * dim + k1);
                                    __m256d input_j_3_0_3 = _mm256_loadu_pd(input + (j2 + 3) * dim + k1);

                                    p_00 = _mm256_fmadd_pd(input_i_0_0_3, input_j_0_0_3, p_00);
                                    p_01 = _mm256_fmadd_pd(input_i_0_0_3, input_j_1_0_3, p_01);
                                    p_02 = _mm256_fmadd_pd(input_i_0_0_3, input_j_2_0_3, p_02);
                                    p_03 = _mm256_fmadd_pd(input_i_0_0_3, input_j_3_0_3, p_03);
                                }
                                __m256d hadd_00_01 = _mm256_hadd_pd(p_00, p_01);
                                __m256d hadd_02_03 = _mm256_hadd_pd(p_02, p_03);

                                __m256d perm_hadd_00_01 = _mm256_permute2f128_pd(hadd_00_01, hadd_00_01,
                                                                                 0b00000001);
                                __m256d perm_hadd_02_03 = _mm256_permute2f128_pd(hadd_02_03, hadd_02_03,
                                                                                 0b00000001);

                                __m256d half_i_0_1 = _mm256_add_pd(hadd_00_01, perm_hadd_00_01);
                                __m256d half_i_0_2 = _mm256_add_pd(hadd_02_03, perm_hadd_02_03);

                                half_i_0_1 = _mm256_blend_pd(half_i_0_1, half_i_0_2, 0b00001100);
                                final_i_0_j = _mm256_add_pd(half_i_0_1, final_i_0_j);

                                _mm256_storeu_pd(result + i2 * n + j2, final_i_0_j);
                            }
                            for (j3 = j2; j3 < j1 + B1; j3++) {
                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    result[i2 * n + j3] += input[i2 * dim + k1] * input[j3 * dim + k1];
                                    result[i2 * n + j3] += input[i2 * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                    result[i2 * n + j3] += input[i2 * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                    result[i2 * n + j3] += input[i2 * dim + k1 + 3] * input[j3 * dim + k1 + 3];
                                }
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

/**
 * FMA AVX 4_4_4
 */
double avx_micro_4_4_4(int n, int dim, int B0, int B1, int BK, const double* input, double* result) {
    int i, j, i1, j1, k, k1, i2, j2, k2, i3, j3;
    double sum;
    int BK2 = 4;
    int B2i = 4;
    int B2j = 4;

    for (i = 0; i < n - B0; i += B0) {
        for (j = 0; j < i - B0; j += B0) {
            for (k = 0; k < dim - BK; k += BK) {

                for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
                    for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                        //if B2 is even, all possible remainders are 2 or 0
                        for (i2 = i1; i2 + B2i <= i1 + B1; i2 += B2i) {
                            for (j2 = j1; j2 + B2j <= j1 + B1; j2 += B2j) {
                                __m256d p_00 = _mm256_setzero_pd();
                                __m256d p_01 = _mm256_setzero_pd();
                                __m256d p_10 = _mm256_setzero_pd();
                                __m256d p_11 = _mm256_setzero_pd();
                                __m256d p_02 = _mm256_setzero_pd();
                                __m256d p_03 = _mm256_setzero_pd();
                                __m256d p_12 = _mm256_setzero_pd();
                                __m256d p_13 = _mm256_setzero_pd();
                                __m256d p_20 = _mm256_setzero_pd();
                                __m256d p_21 = _mm256_setzero_pd();
                                __m256d p_30 = _mm256_setzero_pd();
                                __m256d p_31 = _mm256_setzero_pd();
                                __m256d p_22 = _mm256_setzero_pd();
                                __m256d p_23 = _mm256_setzero_pd();
                                __m256d p_32 = _mm256_setzero_pd();
                                __m256d p_33 = _mm256_setzero_pd();


                                __m256d final_i_0_j = _mm256_loadu_pd(result + i2 * n + j2);
                                __m256d final_i_1_j = _mm256_loadu_pd(result + (i2 + 1) * n + j2);
                                __m256d final_i_2_j = _mm256_loadu_pd(result + (i2 + 2) * n + j2);
                                __m256d final_i_3_j = _mm256_loadu_pd(result + (i2 + 3) * n + j2);

                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    __m256d input_i_0_0_3 = _mm256_loadu_pd(input + i2 * dim + k1);
                                    __m256d input_i_1_0_3 = _mm256_loadu_pd(input + (i2 + 1) * dim + k1);
                                    __m256d input_i_2_0_3 = _mm256_loadu_pd(input + (i2 + 2) * dim + k1);
                                    __m256d input_i_3_0_3 = _mm256_loadu_pd(input + (i2 + 3) * dim + k1);

                                    __m256d input_j_0_0_3 = _mm256_loadu_pd(input + j2 * dim + k1);
                                    __m256d input_j_1_0_3 = _mm256_loadu_pd(input + (j2 + 1) * dim + k1);
                                    __m256d input_j_2_0_3 = _mm256_loadu_pd(input + (j2 + 2) * dim + k1);
                                    __m256d input_j_3_0_3 = _mm256_loadu_pd(input + (j2 + 3) * dim + k1);

                                    p_00 = _mm256_fmadd_pd(input_i_0_0_3, input_j_0_0_3, p_00);
                                    p_01 = _mm256_fmadd_pd(input_i_0_0_3, input_j_1_0_3, p_01);
                                    p_10 = _mm256_fmadd_pd(input_i_1_0_3, input_j_0_0_3, p_10);
                                    p_11 = _mm256_fmadd_pd(input_i_1_0_3, input_j_1_0_3, p_11);
                                    p_02 = _mm256_fmadd_pd(input_i_0_0_3, input_j_2_0_3, p_02);
                                    p_03 = _mm256_fmadd_pd(input_i_0_0_3, input_j_3_0_3, p_03);
                                    p_12 = _mm256_fmadd_pd(input_i_1_0_3, input_j_2_0_3, p_12);
                                    p_13 = _mm256_fmadd_pd(input_i_1_0_3, input_j_3_0_3, p_13);

                                    p_20 = _mm256_fmadd_pd(input_i_2_0_3, input_j_0_0_3, p_20);
                                    p_21 = _mm256_fmadd_pd(input_i_2_0_3, input_j_1_0_3, p_21);
                                    p_30 = _mm256_fmadd_pd(input_i_3_0_3, input_j_0_0_3, p_30);
                                    p_31 = _mm256_fmadd_pd(input_i_3_0_3, input_j_1_0_3, p_31);
                                    p_22 = _mm256_fmadd_pd(input_i_2_0_3, input_j_2_0_3, p_22);
                                    p_23 = _mm256_fmadd_pd(input_i_2_0_3, input_j_3_0_3, p_23);
                                    p_32 = _mm256_fmadd_pd(input_i_3_0_3, input_j_2_0_3, p_32);
                                    p_33 = _mm256_fmadd_pd(input_i_3_0_3, input_j_3_0_3, p_33);

                                }
                                __m256d hadd_00_01 = _mm256_hadd_pd(p_00, p_01);
                                __m256d hadd_10_11 = _mm256_hadd_pd(p_10, p_11);
                                __m256d hadd_02_03 = _mm256_hadd_pd(p_02, p_03);
                                __m256d hadd_12_13 = _mm256_hadd_pd(p_12, p_13);
                                __m256d hadd_20_21 = _mm256_hadd_pd(p_20, p_21);
                                __m256d hadd_30_31 = _mm256_hadd_pd(p_30, p_31);
                                __m256d hadd_22_23 = _mm256_hadd_pd(p_22, p_23);
                                __m256d hadd_32_33 = _mm256_hadd_pd(p_32, p_33);

                                __m256d perm_hadd_00_01 = _mm256_permute2f128_pd(hadd_00_01, hadd_00_01, 0b00000001);
                                __m256d perm_hadd_10_11 = _mm256_permute2f128_pd(hadd_10_11, hadd_10_11, 0b00000001);
                                __m256d perm_hadd_02_03 = _mm256_permute2f128_pd(hadd_02_03, hadd_02_03, 0b00000001);
                                __m256d perm_hadd_12_13 = _mm256_permute2f128_pd(hadd_12_13, hadd_12_13, 0b00000001);
                                __m256d perm_hadd_20_21 = _mm256_permute2f128_pd(hadd_20_21, hadd_20_21, 0b00000001);
                                __m256d perm_hadd_30_31 = _mm256_permute2f128_pd(hadd_30_31, hadd_30_31, 0b00000001);
                                __m256d perm_hadd_22_23 = _mm256_permute2f128_pd(hadd_22_23, hadd_22_23, 0b00000001);
                                __m256d perm_hadd_32_33 = _mm256_permute2f128_pd(hadd_32_33, hadd_32_33, 0b00000001);

                                __m256d half_i_0_1 = _mm256_add_pd(hadd_00_01, perm_hadd_00_01);
                                __m256d half_i_0_2 = _mm256_add_pd(hadd_02_03, perm_hadd_02_03);
                                __m256d half_i_1_1 = _mm256_add_pd(hadd_10_11, perm_hadd_10_11);
                                __m256d half_i_1_2 = _mm256_add_pd(hadd_12_13, perm_hadd_12_13);
                                __m256d half_i_2_1 = _mm256_add_pd(hadd_20_21, perm_hadd_20_21);
                                __m256d half_i_2_2 = _mm256_add_pd(hadd_22_23, perm_hadd_22_23);
                                __m256d half_i_3_1 = _mm256_add_pd(hadd_30_31, perm_hadd_30_31);
                                __m256d half_i_3_2 = _mm256_add_pd(hadd_32_33, perm_hadd_32_33);


                                half_i_0_1 = _mm256_blend_pd(half_i_0_1, half_i_0_2, 0b00001100);
                                half_i_1_1 = _mm256_blend_pd(half_i_1_1, half_i_1_2, 0b00001100);
                                half_i_2_1 = _mm256_blend_pd(half_i_2_1, half_i_2_2, 0b00001100);
                                half_i_3_1 = _mm256_blend_pd(half_i_3_1, half_i_3_2, 0b00001100);

                                final_i_0_j = _mm256_add_pd(half_i_0_1, final_i_0_j);
                                final_i_1_j = _mm256_add_pd(half_i_1_1, final_i_1_j);
                                final_i_2_j = _mm256_add_pd(half_i_2_1, final_i_2_j);
                                final_i_3_j = _mm256_add_pd(half_i_3_1, final_i_3_j);

                                _mm256_storeu_pd(result + i2 * n + j2, final_i_0_j);
                                _mm256_storeu_pd(result + (i2 + 1) * n + j2, final_i_1_j);
                                _mm256_storeu_pd(result + (i2 + 2) * n + j2, final_i_2_j);
                                _mm256_storeu_pd(result + (i2 + 3) * n + j2, final_i_3_j);
                            }
                            for (; j2 < j1 + B1; j2++) {
                                double s_00 = 0, s_10 = 0, s_20 = 0, s_30 = 0;
                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    s_00 += input[i2 * dim + k1] * input[j2 * dim + k1];
                                    s_00 += input[i2 * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_00 += input[i2 * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_00 += input[i2 * dim + k1 + 3] * input[j2 * dim + k1 + 3];

                                    s_10 += input[(i2 + 1) * dim + k1] * input[j2 * dim + k1];
                                    s_10 += input[(i2 + 1) * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_10 += input[(i2 + 1) * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_10 += input[(i2 + 1) * dim + k1 + 3] * input[j2 * dim + k1 + 3];

                                    s_20 += input[(i2 + 2) * dim + k1] * input[j2 * dim + k1];
                                    s_20 += input[(i2 + 2) * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_20 += input[(i2 + 2) * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_20 += input[(i2 + 2) * dim + k1 + 3] * input[j2 * dim + k1 + 3];

                                    s_30 += input[(i2 + 3) * dim + k1] * input[j2 * dim + k1];
                                    s_30 += input[(i2 + 3) * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_30 += input[(i2 + 3) * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_30 += input[(i2 + 3) * dim + k1 + 3] * input[j2 * dim + k1 + 3];
                                }
                                result[i2 * n + j2] += s_00;
                                result[(i2 + 1) * n + j2] += s_10;
                                result[(i2 + 2) * n + j2] += s_20;
                                result[(i2 + 3) * n + j2] += s_30;
                            }
                        }
                        for (; i2 < i1 + B1; i2++) {
                            for (j2 = j1; j2 < j1 + B1; j2++) {
                                double s_00 = 0;
                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    s_00 += input[i2 * dim + k1] * input[j2 * dim + k1];
                                    s_00 += input[i2 * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_00 += input[i2 * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_00 += input[i2 * dim + k1 + 3] * input[j2 * dim + k1 + 3];
                                }
                                result[i2 * n + j2] += s_00;
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

/**
 * FMA AVX 2_4_4 and blocking for remainder
 */
double mmm_avx_fastest(int n, int dim, int B0, int B1, int BK, const double* input, double* result) {
    int i, j, i1, j1, k, k1, i2, j2, k2, i3, j3;
    double sum;
    int BK2 = 4;
    int B2i = 2;
    int B2j = 4;

    for (i = 0; i < n - B0; i += B0) {
        for (j = 0; j < i - B0; j += B0) {
            for (k = 0; k <= dim - BK; k += BK) {

                for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
                    for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                        for (i2 = i1; i2 + B2i <= i1 + B1; i2 += B2i) {
                            for (j2 = j1; j2 + B2j <= j1 + B1; j2 += B2j) {
                                __m256d p_00 = _mm256_setzero_pd();
                                __m256d p_01 = _mm256_setzero_pd();
                                __m256d p_10 = _mm256_setzero_pd();
                                __m256d p_11 = _mm256_setzero_pd();
                                __m256d p_02 = _mm256_setzero_pd();
                                __m256d p_03 = _mm256_setzero_pd();
                                __m256d p_12 = _mm256_setzero_pd();
                                __m256d p_13 = _mm256_setzero_pd();

                                __m256d final_i_0_j = _mm256_loadu_pd(result + i2 * n + j2);
                                __m256d final_i_1_j = _mm256_loadu_pd(result + (i2 + 1) * n + j2);

                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    __m256d input_j_0_0_3 = _mm256_loadu_pd(input + j2 * dim + k1);
                                    __m256d input_j_1_0_3 = _mm256_loadu_pd(input + (j2 + 1) * dim + k1);
                                    __m256d input_i_0_0_3 = _mm256_loadu_pd(input + i2 * dim + k1);
                                    __m256d input_i_1_0_3 = _mm256_loadu_pd(input + (i2 + 1) * dim + k1);
                                    __m256d input_j_2_0_3 = _mm256_loadu_pd(input + (j2 + 2) * dim + k1);
                                    __m256d input_j_3_0_3 = _mm256_loadu_pd(input + (j2 + 3) * dim + k1);

                                    p_00 = _mm256_fmadd_pd(input_i_0_0_3, input_j_0_0_3, p_00);
                                    p_01 = _mm256_fmadd_pd(input_i_0_0_3, input_j_1_0_3, p_01);
                                    p_10 = _mm256_fmadd_pd(input_i_1_0_3, input_j_0_0_3, p_10);
                                    p_11 = _mm256_fmadd_pd(input_i_1_0_3, input_j_1_0_3, p_11);
                                    p_02 = _mm256_fmadd_pd(input_i_0_0_3, input_j_2_0_3, p_02);
                                    p_03 = _mm256_fmadd_pd(input_i_0_0_3, input_j_3_0_3, p_03);
                                    p_12 = _mm256_fmadd_pd(input_i_1_0_3, input_j_2_0_3, p_12);
                                    p_13 = _mm256_fmadd_pd(input_i_1_0_3, input_j_3_0_3, p_13);
                                }
                                __m256d hadd_00_01 = _mm256_hadd_pd(p_00, p_01);
                                __m256d hadd_10_11 = _mm256_hadd_pd(p_10, p_11);
                                __m256d hadd_02_03 = _mm256_hadd_pd(p_02, p_03);
                                __m256d hadd_12_13 = _mm256_hadd_pd(p_12, p_13);

                                __m256d perm_hadd_00_01 = _mm256_permute2f128_pd(hadd_00_01, hadd_00_01, 0b00000001);
                                __m256d perm_hadd_10_11 = _mm256_permute2f128_pd(hadd_10_11, hadd_10_11, 0b00000001);
                                __m256d perm_hadd_02_03 = _mm256_permute2f128_pd(hadd_02_03, hadd_02_03, 0b00000001);
                                __m256d perm_hadd_12_13 = _mm256_permute2f128_pd(hadd_12_13, hadd_12_13, 0b00000001);

                                __m256d half_i_0_1 = _mm256_add_pd(hadd_00_01, perm_hadd_00_01);
                                __m256d half_i_0_2 = _mm256_add_pd(hadd_02_03, perm_hadd_02_03);
                                __m256d half_i_1_1 = _mm256_add_pd(hadd_10_11, perm_hadd_10_11);
                                __m256d half_i_1_2 = _mm256_add_pd(hadd_12_13, perm_hadd_12_13);

                                half_i_0_1 = _mm256_blend_pd(half_i_0_1, half_i_0_2, 0b00001100);
                                half_i_1_1 = _mm256_blend_pd(half_i_1_1, half_i_1_2, 0b00001100);

                                final_i_0_j = _mm256_add_pd(half_i_0_1, final_i_0_j);
                                final_i_1_j = _mm256_add_pd(half_i_1_1, final_i_1_j);

                                _mm256_storeu_pd(result + i2 * n + j2, final_i_0_j);
                                _mm256_storeu_pd(result + (i2 + 1) * n + j2, final_i_1_j);
                            }
                            for (j3 = j2; j3 < j1 + B1; j3++) {
                                double s_00 = 0, s_10 = 0;
                                for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                    s_00 += input[i1 * dim + k1] * input[j3 * dim + k1];
                                    s_00 += input[i1 * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                    s_00 += input[i1 * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                    s_00 += input[i1 * dim + k1 + 3] * input[j3 * dim + k1 + 3];

                                    s_10 += input[(i1 + 1) * dim + k1] * input[j3 * dim + k1];
                                    s_10 += input[(i1 + 1) * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                    s_10 += input[(i1 + 1) * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                    s_10 += input[(i1 + 1) * dim + k1 + 3] * input[j3 * dim + k1 + 3];
                                }
                                result[i1 * n + j3] += s_00;
                                result[(i1 + 1) * n + j3] += s_10;
                            }
                        }
                    }
                }
            }
            int k_remainder = dim - k;
            //collecting K with smaller blocks
            for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
                for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                    for (i2 = i1; i2 + B2i <= i1 + B1; i2 += B2i) {
                        if (k_remainder >= BK2) {
                            for (j2 = j1; j2 + B2j <= j1 + B1; j2 += B2j) {
                                __m256d p_00 = _mm256_setzero_pd();
                                __m256d p_01 = _mm256_setzero_pd();
                                __m256d p_10 = _mm256_setzero_pd();
                                __m256d p_11 = _mm256_setzero_pd();
                                __m256d p_02 = _mm256_setzero_pd();
                                __m256d p_03 = _mm256_setzero_pd();
                                __m256d p_12 = _mm256_setzero_pd();
                                __m256d p_13 = _mm256_setzero_pd();

                                __m256d final_i_0_j = _mm256_loadu_pd(result + i2 * n + j2);
                                __m256d final_i_1_j = _mm256_loadu_pd(result + (i2 + 1) * n + j2);

                                for (k1 = k; k1 + BK2 <= k + k_remainder; k1 += BK2) {
                                    __m256d input_j_0_0_3 = _mm256_loadu_pd(input + j2 * dim + k1);
                                    __m256d input_j_1_0_3 = _mm256_loadu_pd(input + (j2 + 1) * dim + k1);
                                    __m256d input_i_0_0_3 = _mm256_loadu_pd(input + i2 * dim + k1);
                                    __m256d input_i_1_0_3 = _mm256_loadu_pd(input + (i2 + 1) * dim + k1);
                                    __m256d input_j_2_0_3 = _mm256_loadu_pd(input + (j2 + 2) * dim + k1);
                                    __m256d input_j_3_0_3 = _mm256_loadu_pd(input + (j2 + 3) * dim + k1);

                                    p_00 = _mm256_fmadd_pd(input_i_0_0_3, input_j_0_0_3, p_00);
                                    p_01 = _mm256_fmadd_pd(input_i_0_0_3, input_j_1_0_3, p_01);
                                    p_10 = _mm256_fmadd_pd(input_i_1_0_3, input_j_0_0_3, p_10);
                                    p_11 = _mm256_fmadd_pd(input_i_1_0_3, input_j_1_0_3, p_11);
                                    p_02 = _mm256_fmadd_pd(input_i_0_0_3, input_j_2_0_3, p_02);
                                    p_03 = _mm256_fmadd_pd(input_i_0_0_3, input_j_3_0_3, p_03);
                                    p_12 = _mm256_fmadd_pd(input_i_1_0_3, input_j_2_0_3, p_12);
                                    p_13 = _mm256_fmadd_pd(input_i_1_0_3, input_j_3_0_3, p_13);
                                }
                                __m256d hadd_00_01 = _mm256_hadd_pd(p_00, p_01);
                                __m256d hadd_10_11 = _mm256_hadd_pd(p_10, p_11);
                                __m256d hadd_02_03 = _mm256_hadd_pd(p_02, p_03);
                                __m256d hadd_12_13 = _mm256_hadd_pd(p_12, p_13);

                                __m256d perm_hadd_00_01 = _mm256_permute2f128_pd(hadd_00_01, hadd_00_01, 0b00000001);
                                __m256d perm_hadd_10_11 = _mm256_permute2f128_pd(hadd_10_11, hadd_10_11, 0b00000001);
                                __m256d perm_hadd_02_03 = _mm256_permute2f128_pd(hadd_02_03, hadd_02_03, 0b00000001);
                                __m256d perm_hadd_12_13 = _mm256_permute2f128_pd(hadd_12_13, hadd_12_13, 0b00000001);

                                __m256d half_i_0_1 = _mm256_add_pd(hadd_00_01, perm_hadd_00_01);
                                __m256d half_i_0_2 = _mm256_add_pd(hadd_02_03, perm_hadd_02_03);
                                __m256d half_i_1_1 = _mm256_add_pd(hadd_10_11, perm_hadd_10_11);
                                __m256d half_i_1_2 = _mm256_add_pd(hadd_12_13, perm_hadd_12_13);

                                half_i_0_1 = _mm256_blend_pd(half_i_0_1, half_i_0_2, 0b00001100);
                                half_i_1_1 = _mm256_blend_pd(half_i_1_1, half_i_1_2, 0b00001100);

                                final_i_0_j = _mm256_add_pd(half_i_0_1, final_i_0_j);
                                final_i_1_j = _mm256_add_pd(half_i_1_1, final_i_1_j);

                                _mm256_storeu_pd(result + i2 * n + j2, final_i_0_j);
                                _mm256_storeu_pd(result + (i2 + 1) * n + j2, final_i_1_j);

                                for (j3 = j2; j3 < j2 + B2j; j3++) {
                                    double s_0 = 0, s_1 = 0;
                                    for (k2 = k1; k2 < dim; k2++) {
                                        s_0 += input[i2 * dim + k2] * input[j3 * dim + k2];
                                        s_1 += input[(i2 + 1) * dim + k2] * input[j3 * dim + k2];
                                    }
                                    result[i2 * n + j3] += s_0;
                                    result[(i2 + 1) * n + j3] += s_1;
                                }
                            }
                            for (; j2 < j1 + B1; j2++) {
                                double s_0 = 0, s_1 = 0;
                                for (k1 = k; k1 + BK2 <= dim; k1 += BK2) {
                                    s_0 += input[i2 * dim + k1] * input[j2 * dim + k1];
                                    s_0 += input[i2 * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_0 += input[i2 * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_0 += input[i2 * dim + k1 + 3] * input[j2 * dim + k1 + 3];

                                    s_1 += input[(i2 + 1) * dim + k1] * input[j2 * dim + k1];
                                    s_1 += input[(i2 + 1) * dim + k1 + 1] * input[j2 * dim + k1 + 1];
                                    s_1 += input[(i2 + 1) * dim + k1 + 2] * input[j2 * dim + k1 + 2];
                                    s_1 += input[(i2 + 1) * dim + k1 + 3] * input[j2 * dim + k1 + 3];
                                }
                                for (; k1 < dim; k1++) {
                                    s_0 += input[i2 * dim + k1] * input[j2 * dim + k1];
                                    s_1 += input[(i2 + 1) * dim + k1] * input[j2 * dim + k1];
                                }
                                result[i2 * n + j2] += s_0;
                                result[(i2 + 1) * n + j2] += s_1;
                            }

                        } else {
                            for (j2 = j1; j2 + B2i <= j1 + B1; j2 += B2i) {
                                double s_00 = 0, s_10 = 0, s_01 = 0, s_11 = 0;
                                for (k1 = k; k1 < dim; k1++) {
                                    s_00 += input[i2 * dim + k1] * input[j2 * dim + k1];
                                    s_01 += input[i2 * dim + k1] * input[(j2 + 1) * dim + k1];
                                    s_10 += input[(i2 + 1) * dim + k1] * input[j2 * dim + k1];
                                    s_11 += input[(i2 + 1) * dim + k1] * input[(j2 + 1) * dim + k1];
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
        }
        // j_out
        //finishing J
        for (i1 = i; i1 + B1 <= i + B0; i1 += B1) {
            for (j1 = j; j1 + B1 < i1; j1 += B1) {
                for (k = 0; k + BK < dim - BK; k += BK) {

                    for (i2 = i1; i2 + 2 <= i1 + B1; i2 += 2) {
                        for (j2 = j1; j2 + 2 <= j1 + B1; j2 += 2) {
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
                //collecting k
                for (i2 = i1; i2 + 2 <= i1 + B1; i2 += 2) {
                    for (j2 = j1; j2 + 2 <= j1 + B1; j2 += 2) {
                        double sum_00 = 0, sum_10 = 0, sum_01 = 0, sum_11 = 0;
                        for (k1 = k; k1 < dim; k1++) {
                            sum_00 += input[i2 * dim + k1] * input[j2 * dim + k1];
                            sum_10 += input[(i2 + 1) * dim + k1] * input[j2 * dim + k1];
                            sum_01 += input[i2 * dim + k1] * input[(j2 + 1) * dim + k1];
                            sum_11 += input[(i2 + 1) * dim + k1] * input[(j2 + 1) * dim + k1];
                        }
                        result[i2 * n + j2] += sum_00;
                        result[i2 * n + j2 + 1] += sum_01;
                        result[(i2 + 1) * n + j2] += sum_10;
                        result[(i2 + 1) * n + j2 + 1] += sum_11;
                    }
                }
            }
            //collecting j1 which was unrolled by B1
            for (i2 = i1; i2 + 2 <= i1 + B1; i2 += 2) {
                for (j2 = j1; j2 <= i2; j2++) {
                    double sum_0 = 0, sum_1 = 0;
                    for (k = 0; k + BK2 <= dim; k += BK2) {
                        sum_0 += input[i2 * dim + k] * input[j2 * dim + k];
                        sum_1 += input[(i2 + 1) * dim + k] * input[j2 * dim + k];

                        sum_0 += input[i2 * dim + k + 1] * input[j2 * dim + k + 1];
                        sum_1 += input[(i2 + 1) * dim + k + 1] * input[j2 * dim + k + 1];

                        sum_0 += input[i2 * dim + k + 2] * input[j2 * dim + k + 2];
                        sum_1 += input[(i2 + 1) * dim + k + 2] * input[j2 * dim + k + 2];

                        sum_0 += input[i2 * dim + k + 3] * input[j2 * dim + k + 3];
                        sum_1 += input[(i2 + 1) * dim + k + 3] * input[j2 * dim + k + 3];
                    }

                    for (k1 = k; k1 < dim; k1++) {
                        sum_0 += input[i2 * dim + k1] * input[j2 * dim + k1];
                        sum_1 += input[(i2 + 1) * dim + k1] * input[j2 * dim + k1];
                    }
                    result[i2 * n + j2] += sum_0;
                    result[(i2 + 1) * n + j2] += sum_1;
                }
            }
        }
    }

    // i_out
    for (; i < n - B1; i += B1) {
        for (j = 0; j <= i - B0; j += B0) {
            for (k = 0; k <= dim - BK; k += BK) {

                for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                    for (i1 = i; i1 + B2i <= i + B1; i1 += B2i) {
                        for (j2 = j1; j2 + B2j <= j1 + B1; j2 += B2j) {
                            __m256d p_00 = _mm256_setzero_pd();
                            __m256d p_01 = _mm256_setzero_pd();
                            __m256d p_10 = _mm256_setzero_pd();
                            __m256d p_11 = _mm256_setzero_pd();
                            __m256d p_02 = _mm256_setzero_pd();
                            __m256d p_03 = _mm256_setzero_pd();
                            __m256d p_12 = _mm256_setzero_pd();
                            __m256d p_13 = _mm256_setzero_pd();

                            __m256d final_i_0_j = _mm256_loadu_pd(result + i1 * n + j2);
                            __m256d final_i_1_j = _mm256_loadu_pd(result + (i1 + 1) * n + j2);
                            for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                __m256d input_j_0_0_3 = _mm256_loadu_pd(input + j2 * dim + k1);
                                __m256d input_j_1_0_3 = _mm256_loadu_pd(input + (j2 + 1) * dim + k1);
                                __m256d input_i_0_0_3 = _mm256_loadu_pd(input + i1 * dim + k1);
                                __m256d input_i_1_0_3 = _mm256_loadu_pd(input + (i1 + 1) * dim + k1);
                                __m256d input_j_2_0_3 = _mm256_loadu_pd(input + (j2 + 2) * dim + k1);
                                __m256d input_j_3_0_3 = _mm256_loadu_pd(input + (j2 + 3) * dim + k1);


                                p_00 = _mm256_fmadd_pd(input_i_0_0_3, input_j_0_0_3, p_00);
                                p_01 = _mm256_fmadd_pd(input_i_0_0_3, input_j_1_0_3, p_01);
                                p_10 = _mm256_fmadd_pd(input_i_1_0_3, input_j_0_0_3, p_10);
                                p_11 = _mm256_fmadd_pd(input_i_1_0_3, input_j_1_0_3, p_11);
                                p_02 = _mm256_fmadd_pd(input_i_0_0_3, input_j_2_0_3, p_02);
                                p_03 = _mm256_fmadd_pd(input_i_0_0_3, input_j_3_0_3, p_03);
                                p_12 = _mm256_fmadd_pd(input_i_1_0_3, input_j_2_0_3, p_12);
                                p_13 = _mm256_fmadd_pd(input_i_1_0_3, input_j_3_0_3, p_13);
                            }
                            __m256d hadd_00_01 = _mm256_hadd_pd(p_00, p_01);
                            __m256d hadd_10_11 = _mm256_hadd_pd(p_10, p_11);
                            __m256d hadd_02_03 = _mm256_hadd_pd(p_02, p_03);
                            __m256d hadd_12_13 = _mm256_hadd_pd(p_12, p_13);

                            __m256d perm_hadd_00_01 = _mm256_permute2f128_pd(hadd_00_01, hadd_00_01, 0b00000001);
                            __m256d perm_hadd_10_11 = _mm256_permute2f128_pd(hadd_10_11, hadd_10_11, 0b00000001);
                            __m256d perm_hadd_02_03 = _mm256_permute2f128_pd(hadd_02_03, hadd_02_03, 0b00000001);
                            __m256d perm_hadd_12_13 = _mm256_permute2f128_pd(hadd_12_13, hadd_12_13, 0b00000001);

                            __m256d half_i_0_1 = _mm256_add_pd(hadd_00_01, perm_hadd_00_01);
                            __m256d half_i_0_2 = _mm256_add_pd(hadd_02_03, perm_hadd_02_03);
                            __m256d half_i_1_1 = _mm256_add_pd(hadd_10_11, perm_hadd_10_11);
                            __m256d half_i_1_2 = _mm256_add_pd(hadd_12_13, perm_hadd_12_13);

                            half_i_0_1 = _mm256_blend_pd(half_i_0_1, half_i_0_2, 0b00001100);
                            half_i_1_1 = _mm256_blend_pd(half_i_1_1, half_i_1_2, 0b00001100);

                            final_i_0_j = _mm256_add_pd(half_i_0_1, final_i_0_j);
                            final_i_1_j = _mm256_add_pd(half_i_1_1, final_i_1_j);

                            _mm256_storeu_pd(result + i1 * n + j2, final_i_0_j);
                            _mm256_storeu_pd(result + (i1 + 1) * n + j2, final_i_1_j);
                        }
                        for (j3 = j2; j3 < j1 + B1; j3++) {
                            double s_00 = 0, s_10 = 0;
                            for (k1 = k; k1 + BK2 <= k + BK; k1 += BK2) {
                                s_00 += input[i1 * dim + k1] * input[j3 * dim + k1];
                                s_00 += input[i1 * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                s_00 += input[i1 * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                s_00 += input[i1 * dim + k1 + 3] * input[j3 * dim + k1 + 3];

                                s_10 += input[(i1 + 1) * dim + k1] * input[j3 * dim + k1];
                                s_10 += input[(i1 + 1) * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                s_10 += input[(i1 + 1) * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                s_10 += input[(i1 + 1) * dim + k1 + 3] * input[j3 * dim + k1 + 3];
                            }
                            result[i1 * n + j3] += s_00;
                            result[(i1 + 1) * n + j3] += s_10;
                        }
                    }
                }
            }
            //collecting k from BK2 incrementing i.e. if dim < 8
            int k_remainder = dim - k;
            for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {
                for (i1 = i; i1 + B2i <= i + B1; i1 += B2i) {

                    if (k_remainder >= BK2) {
                        for (j2 = j1; j2 + B2j <= j1 + B1; j2 += B2j) {
                            __m256d p_00 = _mm256_setzero_pd();
                            __m256d p_01 = _mm256_setzero_pd();
                            __m256d p_10 = _mm256_setzero_pd();
                            __m256d p_11 = _mm256_setzero_pd();
                            __m256d p_02 = _mm256_setzero_pd();
                            __m256d p_03 = _mm256_setzero_pd();
                            __m256d p_12 = _mm256_setzero_pd();
                            __m256d p_13 = _mm256_setzero_pd();

                            __m256d final_i_0_j = _mm256_loadu_pd(result + i1 * n + j2);
                            __m256d final_i_1_j = _mm256_loadu_pd(result + (i1 + 1) * n + j2);

                            for (k1 = k; k1 + BK2 <= k + k_remainder; k1 += BK2) {
                                __m256d input_j_0_0_3 = _mm256_loadu_pd(input + j2 * dim + k1);
                                __m256d input_j_1_0_3 = _mm256_loadu_pd(input + (j2 + 1) * dim + k1);
                                __m256d input_i_0_0_3 = _mm256_loadu_pd(input + i1 * dim + k1);
                                __m256d input_i_1_0_3 = _mm256_loadu_pd(input + (i1 + 1) * dim + k1);
                                __m256d input_j_2_0_3 = _mm256_loadu_pd(input + (j2 + 2) * dim + k1);
                                __m256d input_j_3_0_3 = _mm256_loadu_pd(input + (j2 + 3) * dim + k1);

                                p_00 = _mm256_fmadd_pd(input_i_0_0_3, input_j_0_0_3, p_00);
                                p_01 = _mm256_fmadd_pd(input_i_0_0_3, input_j_1_0_3, p_01);
                                p_10 = _mm256_fmadd_pd(input_i_1_0_3, input_j_0_0_3, p_10);
                                p_11 = _mm256_fmadd_pd(input_i_1_0_3, input_j_1_0_3, p_11);
                                p_02 = _mm256_fmadd_pd(input_i_0_0_3, input_j_2_0_3, p_02);
                                p_03 = _mm256_fmadd_pd(input_i_0_0_3, input_j_3_0_3, p_03);
                                p_12 = _mm256_fmadd_pd(input_i_1_0_3, input_j_2_0_3, p_12);
                                p_13 = _mm256_fmadd_pd(input_i_1_0_3, input_j_3_0_3, p_13);
                            }
                            __m256d hadd_00_01 = _mm256_hadd_pd(p_00, p_01);
                            __m256d hadd_10_11 = _mm256_hadd_pd(p_10, p_11);
                            __m256d hadd_02_03 = _mm256_hadd_pd(p_02, p_03);
                            __m256d hadd_12_13 = _mm256_hadd_pd(p_12, p_13);

                            __m256d perm_hadd_00_01 = _mm256_permute2f128_pd(hadd_00_01, hadd_00_01, 0b00000001);
                            __m256d perm_hadd_10_11 = _mm256_permute2f128_pd(hadd_10_11, hadd_10_11, 0b00000001);
                            __m256d perm_hadd_02_03 = _mm256_permute2f128_pd(hadd_02_03, hadd_02_03, 0b00000001);
                            __m256d perm_hadd_12_13 = _mm256_permute2f128_pd(hadd_12_13, hadd_12_13, 0b00000001);

                            __m256d half_i_0_1 = _mm256_add_pd(hadd_00_01, perm_hadd_00_01);
                            __m256d half_i_0_2 = _mm256_add_pd(hadd_02_03, perm_hadd_02_03);
                            __m256d half_i_1_1 = _mm256_add_pd(hadd_10_11, perm_hadd_10_11);
                            __m256d half_i_1_2 = _mm256_add_pd(hadd_12_13, perm_hadd_12_13);

                            half_i_0_1 = _mm256_blend_pd(half_i_0_1, half_i_0_2, 0b00001100);
                            half_i_1_1 = _mm256_blend_pd(half_i_1_1, half_i_1_2, 0b00001100);

                            final_i_0_j = _mm256_add_pd(half_i_0_1, final_i_0_j);
                            final_i_1_j = _mm256_add_pd(half_i_1_1, final_i_1_j);

                            _mm256_storeu_pd(result + i1 * n + j2, final_i_0_j);
                            _mm256_storeu_pd(result + (i1 + 1) * n + j2, final_i_1_j);
                            for (; k1 < k + k_remainder; k1++) {
                                for (i2 = i1; i2 < i1 + B2i; i2++) {
                                    for (j3 = j2; j3 < j2 + B2j; j3++) {
                                        result[i2 * n + j3] += input[i2 * dim + k1] * input[j3 * dim + k1];
                                    }
                                }
                            }
                        }
                        for (j3 = j2; j3 < j1 + B1; j3++) {
                            for (k1 = k; k1 + BK2 <= k + k_remainder; k1 += BK2) {
                                result[i1 * n + j3] += input[i1 * dim + k1] * input[j3 * dim + k1];
                                result[i1 * n + j3] += input[i1 * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                result[i1 * n + j3] += input[i1 * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                result[i1 * n + j3] += input[i1 * dim + k1 + 3] * input[j3 * dim + k1 + 3];

                                result[(i1 + 1) * n + j3] += input[(i1 + 1) * dim + k1] * input[j3 * dim + k1];
                                result[(i1 + 1) * n + j3] +=
                                        input[(i1 + 1) * dim + k1 + 1] * input[j3 * dim + k1 + 1];
                                result[(i1 + 1) * n + j3] +=
                                        input[(i1 + 1) * dim + k1 + 2] * input[j3 * dim + k1 + 2];
                                result[(i1 + 1) * n + j3] +=
                                        input[(i1 + 1) * dim + k1 + 3] * input[j3 * dim + k1 + 3];
                            }
                            for (; k1 < k_remainder; k1++) {
                                result[i1 * n + j3] += input[i1 * dim + k1] * input[j3 * dim + k1];
                            }
                        }
                    } else {
                        for (j2 = j1; j2 + B2i <= j1 + B1; j2 += B2i) {
                            double sum_00 = 0, sum_01 = 0, sum_10 = 0, sum_11 = 0;
                            for (k1 = k; k1 < dim; k1++) {
                                sum_00 += input[i1 * dim + k1] * input[j2 * dim + k1];
                                sum_10 += input[(i1 + 1) * dim + k1] * input[j2 * dim + k1];
                                sum_01 += input[i1 * dim + k1] * input[(j2 + 1) * dim + k1];
                                sum_11 += input[(i1 + 1) * dim + k1] * input[(j2 + 1) * dim + k1];
                            }
                            result[i1 * n + j2] += sum_00;
                            result[i1 * n + j2 + 1] += sum_01;
                            result[(i1 + 1) * n + j2] += sum_10;
                            result[(i1 + 1) * n + j2 + 1] += sum_11;
                        }
                    }
                }
            }
        }

        //Need to block here on j1 since it goes from 200 to 220 for B0 =100 and B1= 20
        for (i1 = i; i1 + B2i <= i + B1; i1 += B2i) {
            for (j1 = j; j1 + B2i <= i1 + B2i; j1 += B2i) {
                double s_00 = 0, s_01 = 0, s_10 = 0, s_11 = 0;
                for (k1 = 0; k1 + BK2 <= dim; k1 += BK2) {
                    s_00 += input[i1 * dim + k1] * input[j1 * dim + k1];
                    s_01 += input[i1 * dim + k1] * input[(j1 + 1) * dim + k1];
                    s_10 += input[(i1 + 1) * dim + k1] * input[j1 * dim + k1];
                    s_11 += input[(i1 + 1) * dim + k1] * input[(j1 + 1) * dim + k1];

                    s_00 += input[i1 * dim + k1 + 1] * input[j1 * dim + k1 + 1];
                    s_01 += input[i1 * dim + k1 + 1] * input[(j1 + 1) * dim + k1 + 1];
                    s_10 += input[(i1 + 1) * dim + k1 + 1] * input[j1 * dim + k1 + 1];
                    s_11 += input[(i1 + 1) * dim + k1 + 1] * input[(j1 + 1) * dim + k1 + 1];

                    s_00 += input[i1 * dim + k1 + 2] * input[j1 * dim + k1 + 2];
                    s_01 += input[i1 * dim + k1 + 2] * input[(j1 + 1) * dim + k1 + 2];
                    s_10 += input[(i1 + 1) * dim + k1 + 2] * input[j1 * dim + k1 + 2];
                    s_11 += input[(i1 + 1) * dim + k1 + 2] * input[(j1 + 1) * dim + k1 + 2];

                    s_00 += input[i1 * dim + k1 + 3] * input[j1 * dim + k1 + 3];
                    s_01 += input[i1 * dim + k1 + 3] * input[(j1 + 1) * dim + k1 + 3];
                    s_10 += input[(i1 + 1) * dim + k1 + 3] * input[j1 * dim + k1 + 3];
                    s_11 += input[(i1 + 1) * dim + k1 + 3] * input[(j1 + 1) * dim + k1 + 3];
                }
                for (; k1 < dim; k1++) {
                    s_00 += input[i1 * dim + k1] * input[j1 * dim + k1];
                    s_01 += input[i1 * dim + k1] * input[(j1 + 1) * dim + k1];
                    s_10 += input[(i1 + 1) * dim + k1] * input[j1 * dim + k1];
                    s_11 += input[(i1 + 1) * dim + k1] * input[(j1 + 1) * dim + k1];
                }
                result[i1 * n + j1] += s_00;
                result[i1 * n + j1 + 1] += s_01;
                result[(i1 + 1) * n + j1] += s_10;
                result[(i1 + 1) * n + j1 + 1] += s_11;
            }
        }
    }

    for (i1 = i; i1 < n; i1++) {
        for (j = 0; j <= i1 - B0; j += B0) {
            for (j1 = j; j1 + B1 <= j + B0; j1 += B1) {

                for (j2 = j1; j2 + B2j <= j1 + B1; j2 += B2j) {
                    __m256d p_00 = _mm256_setzero_pd();
                    __m256d p_01 = _mm256_setzero_pd();
                    __m256d p_02 = _mm256_setzero_pd();
                    __m256d p_03 = _mm256_setzero_pd();
                    __m256d final_i_0_j = _mm256_loadu_pd(result + i1 * n + j2);
                    double s_0 = 0, s_1 = 0, s_2 = 0, s_3 = 0, s_4 = 0;
                    for (k = 0; k + BK2 <= dim; k += BK2) {
                        __m256d input_i_0_0_3 = _mm256_loadu_pd(input + i1 * dim + k);
                        __m256d input_j_0_0_3 = _mm256_loadu_pd(input + j2 * dim + k);
                        __m256d input_j_1_0_3 = _mm256_loadu_pd(input + (j2 + 1) * dim + k);
                        __m256d input_j_2_0_3 = _mm256_loadu_pd(input + (j2 + 2) * dim + k);
                        __m256d input_j_3_0_3 = _mm256_loadu_pd(input + (j2 + 3) * dim + k);

                        p_00 = _mm256_fmadd_pd(input_i_0_0_3, input_j_0_0_3, p_00);
                        p_01 = _mm256_fmadd_pd(input_i_0_0_3, input_j_1_0_3, p_01);
                        p_02 = _mm256_fmadd_pd(input_i_0_0_3, input_j_2_0_3, p_02);
                        p_03 = _mm256_fmadd_pd(input_i_0_0_3, input_j_3_0_3, p_03);
                    }
                    __m256d hadd_00_01 = _mm256_hadd_pd(p_00, p_01);
                    __m256d hadd_02_03 = _mm256_hadd_pd(p_02, p_03);

                    __m256d perm_hadd_00_01 = _mm256_permute2f128_pd(hadd_00_01, hadd_00_01,
                                                                     0b00000001);
                    __m256d perm_hadd_02_03 = _mm256_permute2f128_pd(hadd_02_03, hadd_02_03,
                                                                     0b00000001);

                    __m256d half_i_0_1 = _mm256_add_pd(hadd_00_01, perm_hadd_00_01);
                    __m256d half_i_0_2 = _mm256_add_pd(hadd_02_03, perm_hadd_02_03);

                    half_i_0_1 = _mm256_blend_pd(half_i_0_1, half_i_0_2, 0b00001100);
                    final_i_0_j = _mm256_add_pd(half_i_0_1, final_i_0_j);

                    _mm256_storeu_pd(result + i1 * n + j2, final_i_0_j);

                    for (k1 = k; k1 < dim; k1++) {
                        s_0 += input[i1 * dim + k1] * input[j2 * dim + k1];
                        s_1 += input[i1 * dim + k1] * input[(j2 + 1) * dim + k1];
                        s_2 += input[i1 * dim + k1] * input[(j2 + 2) * dim + k1];
                        s_3 += input[i1 * dim + k1] * input[(j2 + 3) * dim + k1];
                    }
                    result[i1 * n + j2] += s_0;
                    result[i1 * n + j2 + 1] += s_1;
                    result[i1 * n + j2 + 2] += s_2;
                    result[i1 * n + j2 + 3] += s_3;
                }
                for (; j2 < j1 + B1; j2++) {
                    sum = 0;
                    for (k = 0; k + BK2 <= dim; k += BK2) {
                        sum += input[i1 * dim + k] * input[j2 * dim + k];
                        sum += input[i1 * dim + k + 1] * input[j2 * dim + k + 1];
                        sum += input[i1 * dim + k + 2] * input[j2 * dim + k + 2];
                        sum += input[i1 * dim + k + 3] * input[j2 * dim + k + 3];
                    }
                    for (; k < dim; k++) {
                        sum += input[i1 * dim + k] * input[j2 * dim + k];
                    }
                    result[i1 * n + j2] += sum;
                }
            }
        }
        for (; j <= i1 - B1; j += B1) {
            for (j1 = j; j1 < j + B1; j1++) {
                sum = 0;
                for (k = 0; k + BK2 <= dim; k += BK2) {
                    sum += input[i1 * dim + k] * input[j1 * dim + k];
                    sum += input[i1 * dim + k + 1] * input[j1 * dim + k + 1];
                    sum += input[i1 * dim + k + 2] * input[j1 * dim + k + 2];
                    sum += input[i1 * dim + k + 3] * input[j1 * dim + k + 3];
                }
                for (; k < dim; k++) {
                    sum += input[i1 * dim + k] * input[j1 * dim + k];
                }
                result[i1 * n + j1] += sum;
            }
        }
        for (; j < i1; j++) {
            sum = 0;
            for (k = 0; k + BK2 <= dim; k += BK2) {
                sum += input[i1 * dim + k] * input[j * dim + k];
                sum += input[i1 * dim + k + 1] * input[j * dim + k + 1];
                sum += input[i1 * dim + k + 2] * input[j * dim + k + 2];
                sum += input[i1 * dim + k + 3] * input[j * dim + k + 3];
            }
            for (; k < dim; k++) {
                sum += input[i1 * dim + k] * input[j * dim + k];
            }
            result[i1 * n + j] += sum;
        }
    }
    return 2.0 * n * (n - 1) * dim / 2;
}


double baseline_wrapper(int n, int dim, int B0, int B1, int BK, const double* input, double* result) {
    mmm_baseline(n, dim, input, result);

    return 2.0 * n * (n - 1) * dim / 2;
}


#define  NUM_FUNCTIONS 6
#define  CYCLES_REQ 1e8

void avx_mmm_driver(int n, int dim, int B0, int B1, int BK, int nr_reps, int fun_index) {
    double* input_points = XmallocMatrixDoubleRandom(n, dim);
    double* mm_true = XmallocMatrixDouble(n, n);

    if (B0 % B1 != 0) {
        printf("B1 must be divisor of B0");
        exit(-1);
    }

    if (BK % 4 != 0) {
        printf("Bk must be multiple of 4");
        exit(-1);
    }

    int BKadj = dim / 4;
    BKadj = BKadj == 0 ? dim : BKadj * 4;

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &baseline_wrapper;
    fun_array[1] = &mmm_unroll_fastest;
    fun_array[2] = &avx_micro_2_4_4;
    fun_array[3] = &avx_micro_1_4_4;
    fun_array[4] = &avx_micro_4_4_4;
    fun_array[5] = &mmm_avx_fastest;


    char* fun_names[NUM_FUNCTIONS] = {"mmm_baseline", "micro_2_2_4", "avx_micro_2_4_4",
                                      "avx_micro_1_4_4", "avx_micro_4_4_4", "mmm_avx_fastest"};

    myInt64 start, end;
    double cycles1;
    double flops;

    FILE* results_file = open_with_error_check("../MMM_AVX_N.txt", "a");
    mmm_baseline(n, dim, input_points, mm_true);

    fprintf(results_file, "%s\n", fun_names[fun_index]);
    fprintf(results_file, "%d\n%d, ", dim, n);

    double* mm_results = XmallocMatrixDouble(n, n);

    // VERIFICATION : ------------------------------------------------------------------------
    (*fun_array[fun_index])(n, dim, B0, B1, BKadj, input_points, mm_results);
    int ver = test_double_matrices(n, 1e-3, mm_results, mm_true);
//        int ver = test_double_arrays(n * n, 1e-3, mm_results, mm_true);
    if (ver != 1) {
//        printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
//        print_matrices(n, mm_true, mm_results);
//        exit(-1);
    }

    double multiplier = 1;
    double numRuns = 10;

    // Warm-up phase: we determine a number of executions that allows
    do {
        numRuns = numRuns * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < numRuns; i++) {
            (*fun_array[fun_index])(n, dim, B0, B1, BKadj, input_points, mm_results);
        }
        end = stop_tsc(start);

        cycles1 = (double) end;
        multiplier = (CYCLES_REQ) / (cycles1);

    } while (multiplier > 2);

    double* cyclesPtr = XmallocVectorDouble(nr_reps);

    for (size_t j = 0; j < nr_reps; j++) {
        CleanTheCache(200);
        start = start_tsc();
        for (size_t i = 0; i < 2; ++i) {
            flops = (*fun_array[fun_index])(n, dim, B0, B1, BKadj, input_points, mm_results);
        }
        end = stop_tsc(start);

        cycles1 = ((double) end) / 2;
        cyclesPtr[j] = cycles1;
    }

    qsort(cyclesPtr, nr_reps, sizeof(double), compare_double);
    double cycles = cyclesPtr[(int) nr_reps / 2 + 1];
    double perf = round((1000.0 * flops) / cycles) / 1000.0;

    printf("%s n:%d d:%d B0:%d B1:%d cycles:%lf perf:%lf \n", fun_names[fun_index], n, dim, B0, B1, cycles, perf);

    for (int i = 0; i < nr_reps; i++) {
        double perf = round((1000.0 * flops) / cyclesPtr[i]) / 1000.0;
        fprintf(results_file, "%lf,", perf);
    }
    fprintf(results_file, "\n");

    free(mm_results);
    free(cyclesPtr);
    printf("-------------\n");
    free(input_points);
    free(mm_true);
    fclose(results_file);

}

void avx_mmm_block_profiler(int n, int dim, int B0, int B1, int BK, int nr_reps) {
    double* input_points = XmallocMatrixDoubleRandom(n, dim);
    double* mm_results = XmallocMatrixDouble(n, n);

    if (B0 % B1 != 0) {
        printf("B1 must be divisor of B0");
        exit(-1);
    }

    if (BK % 4 != 0) {
        printf("Bk must be multiple of 4");
        exit(-1);
    }

    myInt64 start, end;
    double cycles1;
    double flops;
    int numRuns = 2;

    double* cyclesPtr = XmallocVectorDouble(nr_reps);

    for (size_t j = 0; j < nr_reps; j++) {
        CleanTheCache(200);
        start = start_tsc();
        for (size_t i = 0; i < numRuns; ++i) {
            flops = mmm_avx_fastest(n, dim, B0, B1, BK, input_points, mm_results);
        }
        end = stop_tsc(start);

        cycles1 = ((double) end) / numRuns;
        cyclesPtr[j] = cycles1;
    }

    qsort(cyclesPtr, nr_reps, sizeof(double), compare_double);

    double cycles = cyclesPtr[(int) nr_reps / 2 + 1];
    double perf = round((1000.0 * flops) / cycles) / 1000.0;
    printf("%d, %d, %d-%d, %d, %lf\n", n, dim, B0, B1, BK, perf);

    free(cyclesPtr);
    free(input_points);
    free(mm_results);
}