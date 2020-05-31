//
// Created by fvasluia on 5/19/20.
//
#include <immintrin.h>
#ifndef FASTLOF_AVX_UTILS_H
#define FASTLOF_AVX_UTILS_H

double sum_double_avx(__m256d v);
double sum_double_avx_2(__m256d v);
void check_intrinsics();
__m256i cvt_to_256i( __m128i inp );
void __MM256_TRANSPOSE(__m256i row0, __m256i row1, __m256i row2, __m256i row3);

#endif //FASTLOF_AVX_UTILS_H
