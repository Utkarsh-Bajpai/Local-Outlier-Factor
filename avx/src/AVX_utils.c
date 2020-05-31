//
// Created by Sycheva  Anastasia on 19.05.20.
//

#include "../include/AVX_utils.h"
#include "../../include/utils.h"
#include <string.h>

double sum_double_avx(__m256d v) { // claimed to be faster
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low  = _mm_add_pd(low, high);

    __m128d high64 = _mm_unpackhi_pd(low, low);
    return  _mm_cvtsd_f64(_mm_add_sd(low, high64));
}

double sum_double_avx_2(__m256d v){

    __m256d tmp1 = _mm256_hadd_pd(v, v);
    //__m256d tmp2 = _mm256_hadd_pd(tmp1, tmp1);
    __m256d tmp2 = _mm256_permute2f128_pd(tmp1, tmp1, 33);

    return _mm256_cvtsd_f64(_mm256_add_pd(tmp2, tmp1));
}
// ----------------------------------------------------------------->

void check_intrinsics(){
    /**
     * Used this function to clarify the effect of different intrinsics
     */

    __m128i var_128i, var_128i_1, var_128i_2;
    __m256i var_256i;
    __m256d var_256d;

    int check_int[4];
    double check_double[4];
    int* neighb_mat = XmallocMatrixIntRandom(100, 10, 100);
    double* matrix_test = XmallocMatrixDoubleRandom(100, 100);

    // SET: CORRECT
    var_128i = _mm_set_epi32( neighb_mat[6], neighb_mat[17], neighb_mat[2], neighb_mat[11]);
    memcpy(check_int, &var_128i, sizeof(check_int));
    printf("Out :\t\t%d %d %d %d\n", check_int[0], check_int[1], check_int[2], check_int[3]);
    printf("Should :\t%d %d %d %d\n", neighb_mat[11], neighb_mat[2], neighb_mat[17], neighb_mat[6]);

    // GATHER: CORRECT
    var_256d = _mm256_i32gather_pd(matrix_test, var_128i, sizeof(double));
    memcpy(check_double, &var_256d, sizeof(check_double));
    printf("Out :\t\t%f %f %f %f\n", check_double[0], check_double[1], check_double[2], check_double[3]);
    printf("Should :\t%f %f %f %f\n", matrix_test[check_int[0]], matrix_test[check_int[1]], matrix_test[check_int[2]], matrix_test[check_int[3]]);

    // CHECK DIFFERENT OPERATIONS ON _m128i
    var_128i_1 = _mm_set_epi32( 10, 101, 14, 3);
    var_128i_2 = _mm_set_epi32( 15, 2, 20, 5 );

    printf("\n\nInput 1 :\t\t%d %d %d %d\n", 3, 14, 101, 10);
    printf("Input 2 :\t\t%d %d %d %d\n\n", 5, 20, 2, 15 );

    // MULTIPLICATION
    var_128i = _mm_mullo_epi32(var_128i_1, var_128i_2); //_mm_mul_epi32(var_128i_1, var_128i_2); // => DOES NOT WORK
    memcpy(check_int, &var_128i, sizeof(check_int));
    printf("Out mul :\t\t%d %d %d %d\n", check_int[0], check_int[1], check_int[2], check_int[3]);

    // ADDITION
    var_128i = _mm_add_epi32(var_128i_1, var_128i_2);
    memcpy(check_int, &var_128i, sizeof(check_int));
    printf("Out add :\t\t%d %d %d %d\n", check_int[0], check_int[1], check_int[2], check_int[3]);

    // MIN
    var_128i = _mm_min_epi32(var_128i_1, var_128i_2);
    memcpy(check_int, &var_128i, sizeof(check_int));
    printf("Out min :\t\t%d %d %d %d\n", check_int[0], check_int[1], check_int[2], check_int[3]);

    // SUM INTRINSICS:
    var_256d = _mm256_i32gather_pd(matrix_test, var_128i, sizeof(double));
    memcpy(check_double, &var_256d, sizeof(check_double));
    double res = sum_double_avx_2(var_256d);
    double expected = check_double[0] + check_double[1] + check_double[2] + check_double[3];
    printf("Out :\t\t%f %f %f %f: sum = %lf vs expected %lf\n", check_double[0], check_double[1], check_double[2], check_double[3], res, expected);


    // SUM ELEMENTS
    printf("TRY SUM\n");
    __m256d to_sum = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    __m256d tmp1 = _mm256_hadd_pd(to_sum, to_sum);
    memcpy(check_double, &tmp1, sizeof(check_double));
    printf("Out :\t\t%f %f %f %f\n", check_double[0], check_double[1], check_double[2], check_double[3]);
    __m256d tmp2 = _mm256_permute2f128_pd(tmp1, tmp1, 33);
    // 00110011
    memcpy(check_double, &tmp2, sizeof(check_double));
    printf("Out :\t\t%f %f %f %f\n", check_double[0], check_double[1], check_double[2], check_double[3]);

}

// -------------------------------------------------------------------------

__m256i cvt_to_256i( __m128i inp ){
    /**
     * "convert" from __m128i to __m256i without using AVX-512
     * Load 8 ints -
     * PROBLEM: need to collect indexes from neighborhood (ints): -> do min, add (_m256i - AVX 512) -> access the elements (__256i)
     * QUESTION: upstream double ? 4 doubles from arbitrary location, _128i
     */
    int tmp_int[4];
    _mm_store_si128( tmp_int, inp );
    return _mm256_set_epi64x( tmp_int[3], tmp_int[2], tmp_int[1], tmp_int[0] );
}

void __MM256_TRANSPOSE(__m256i row0, __m256i row1, __m256i row2, __m256i row3){
    /**
     * Transpose a matrix consisting of 4 __m256i rows
     * NORMALLY IT IS SUBOPTIMAL ...
     */
    __m256i tmp0, tmp1, tmp2, tmp3;
    tmp0 = _mm256_unpacklo_epi64( row0, row1 );   // ( a1, b1, a3, b3 )
    tmp1 = _mm256_unpackhi_epi64( row0, row1 );   // ( a2, b2, a4, b4 )
    tmp2 = _mm256_unpacklo_epi64( row2, row3 );   // ( c1, d1, c3, d3 )
    tmp3 = _mm256_unpackhi_epi64( row2, row3 );   // ( c1, d1, c3, d3 )

    row0 = _mm256_permute2f128_si256(tmp2, tmp0, 2);    // (a0, b0, c0, d0)
    row1 = _mm256_permute2f128_si256(tmp3, tmp1, 2);    // (a2, b2, c2, d2)
    row2 = _mm256_permute2f128_si256(tmp2, tmp0, 19);   // (a3, b3, c3, d3)
    row3 = _mm256_permute2f128_si256(tmp3, tmp1, 19);   // (a4, b4, c4, d4)

}
