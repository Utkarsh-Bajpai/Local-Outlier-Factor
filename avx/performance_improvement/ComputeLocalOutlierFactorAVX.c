#include <stdio.h>
#include <immintrin.h>
#include <math.h>
#include "stdlib.h"

#include "../../include/Algorithm.h"
#include "../../include/utils.h"
#include "../../include/tests.h"
#include "../../include/tsc_x86.h"
#include "../../include/lof_baseline.h"
#include "../../include/performance_measurement.h"
#include "../include/AVX_utils.h"


#define NUM_FUNCTIONS_LOF_AVX 2

typedef void (* my_fun)(int k, int num_pts, const double* lrd_score_table_ptr,
                        const double* neighborhood_index_table_ptr,
                        double* lof_score_table_ptr);

#define CYCLES_REQUIRED 1e8
#define NUM_FUNCTIONS 5
// num of new functions + 1 for baseline
// ------------------------------------------------------------------------------------ <AVX improvements>

double ComputeLocalOutlierFactor_1(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr) {
    /**
     * Change operation strength
     */

    unsigned int i, j;
    for (i = 0; i < num_pts; i++) {

        double lrd_neighs_sum = 0;
        for (j = 0; j < k; j++) {

            int neigh_index = neighborhood_index_table_ptr[i * k + j];
            lrd_neighs_sum += lrd_score_table_ptr[neigh_index];

        }
        double denom = lrd_score_table_ptr[i]*k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }
    printf("\n");

    return num_pts * (2 + k);
}   // faster_function_1

void ComputeLocalOutlierFactor_Unrolled(int k, int num_pts, const double* lrd_score_table_ptr,
                                        const int* neighborhood_index_table_ptr,
                                        double* lof_score_table_ptr) {
    /**
     * ComputeLocalOutlierFactor_4 + outer loop unrolling by 8
     */

    double lrd_neighs_sum, lrd_neighs_sum_0, lrd_neighs_sum_1, lrd_neighs_sum_2, lrd_neighs_sum_3, lrd_neighs_sum_4, lrd_neighs_sum_5, lrd_neighs_sum_6, lrd_neighs_sum_7;
    double denom, denom_0, denom_1, denom_2, denom_3, denom_4, denom_5, denom_6, denom_7;

    int idx_base, idx_base_0, idx_base_1, idx_base_2, idx_base_3, idx_base_4, idx_base_5, idx_base_6, idx_base_7;
    int idx, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;

    unsigned int i, j;
    i = 0; j = 0;

    for ( ; i + 8 < num_pts; i += 8) {

        lrd_neighs_sum_0 = 0;
        lrd_neighs_sum_1 = 0;
        lrd_neighs_sum_2 = 0;
        lrd_neighs_sum_3 = 0;
        lrd_neighs_sum_4 = 0;
        lrd_neighs_sum_5 = 0;
        lrd_neighs_sum_6 = 0;
        lrd_neighs_sum_7 = 0;

        idx_base_0 = i * k;
        idx_base_1 = (i + 1) * k;
        idx_base_2 = (i + 2) * k;
        idx_base_3 = (i + 3) * k;
        idx_base_4 = (i + 4) * k;
        idx_base_5 = (i + 5) * k;
        idx_base_6 = (i + 6) * k;
        idx_base_7 = (i + 7) * k;

        j = 0;
        for ( ; j + 4 < k; j += 4 ) {

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0 + 1]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0 + 2]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0 + 3]];

            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 1]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 2]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 3]];

            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 1]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 2]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 3]];

            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 1]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 2]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 3]];

            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 1]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 2]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 3]];

            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 1]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 2]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 3]];

            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 1]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 2]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 3]];

            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 1]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 2]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 3]];

        }
        if(i==0) {
            printf("%lf, %lf, %lf, %lf,  ", lrd_neighs_sum_0, lrd_neighs_sum_1, lrd_neighs_sum_2, lrd_neighs_sum_3);
            printf("%lf, %lf, %lf, %lf\n", lrd_neighs_sum_4, lrd_neighs_sum_5, lrd_neighs_sum_6, lrd_neighs_sum_7);
        }


        // collect remaining stuff
        for( ; j < k; ++j){

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7]];

        }

        denom_0 = lrd_score_table_ptr[i]*k;
        denom_1 = lrd_score_table_ptr[i+1]*k;
        denom_2 = lrd_score_table_ptr[i+2]*k;
        denom_3 = lrd_score_table_ptr[i+3]*k;
        denom_4 = lrd_score_table_ptr[i+4]*k;
        denom_5 = lrd_score_table_ptr[i+5]*k;
        denom_6 = lrd_score_table_ptr[i+6]*k;
        denom_7 = lrd_score_table_ptr[i+7]*k;

        lof_score_table_ptr[i] = lrd_neighs_sum_0 / denom_0;
        lof_score_table_ptr[i+1] = lrd_neighs_sum_1 / denom_1;
        lof_score_table_ptr[i+2] = lrd_neighs_sum_2 / denom_2;
        lof_score_table_ptr[i+3] = lrd_neighs_sum_3 / denom_3;
        lof_score_table_ptr[i+4] = lrd_neighs_sum_4 / denom_4;
        lof_score_table_ptr[i+5] = lrd_neighs_sum_5 / denom_5;
        lof_score_table_ptr[i+6] = lrd_neighs_sum_6 / denom_6;
        lof_score_table_ptr[i+7] = lrd_neighs_sum_7 / denom_7;

    }

    // collect remaining i: ---------------------------------------------

    for ( ; i < num_pts; i++) {

        lrd_neighs_sum = 0;
        idx_base = i*k;
        j = 0;
        for ( ; j + 4 < k; j += 4) {

            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 1]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 2]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 3]];

        }

        // collect remaining stuff
        for( ; j < k; ++j){
            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[ neighborhood_index_table_ptr[idx]];
        }

        denom = lrd_score_table_ptr[i]*k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }

}






void ComputeLocalOutlierFactor_AVX_Old(int k, int num_pts, const double* lrd_score_table_ptr,
                                       const int* neighborhood_index_table_ptr,
                                       double* lof_score_table_ptr) {
    /**
     * ComputeLocalOutlierFactor_4 + outer loop unrolling by 8
     */

    double lrd_neighs_sum, lrd_neighs_sum_0, lrd_neighs_sum_1, lrd_neighs_sum_2, lrd_neighs_sum_3, lrd_neighs_sum_4, lrd_neighs_sum_5, lrd_neighs_sum_6, lrd_neighs_sum_7;
    double denom, denom_0, denom_1, denom_2, denom_3, denom_4, denom_5, denom_6, denom_7;

    int idx_base, idx_base_0, idx_base_1, idx_base_2, idx_base_3, idx_base_4, idx_base_5, idx_base_6, idx_base_7;
    int idx, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;

    __m256d vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec, temp1, temp2;

    __m256d result0_3, result4_7;

    __m256d res0, res1, res2, res3, sum0, sum1, sum2, sum3;

    __m256d mul_k = _mm256_set1_pd(k);

    __m256d denom0_3, denom4_7;

    __m128i idx_128_0, idx_128_1, idx_128_2, idx_128_3, idx_128_4, idx_128_5, idx_128_6, idx_128_7;

    //unsigned int i, j;
    int i = 0, j = 0;

    double print_vec_1[4], print_vec_2[4];

    for ( ; i + 8 < num_pts; i += 8) {

        printf("\nHello!\n");

        result0_3 = _mm256_set1_pd(0.0);
        result4_7 = _mm256_set1_pd(0.0);

        idx_base_0 = i * k;
        idx_base_1 = (i + 1) * k;
        idx_base_2 = (i + 2) * k;
        idx_base_3 = (i + 3) * k;
        idx_base_4 = (i + 4) * k;
        idx_base_5 = (i + 5) * k;
        idx_base_6 = (i + 6) * k;
        idx_base_7 = (i + 7) * k;

        j = 0;
        for ( ; j + 4 < k; j += 4 ) {

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;


            idx_128_0 = _mm_loadu_si128( neighborhood_index_table_ptr + idx0);

            idx_128_1 = _mm_loadu_si128( neighborhood_index_table_ptr + idx1);

            idx_128_2 = _mm_loadu_si128( neighborhood_index_table_ptr + idx2);

            idx_128_3 = _mm_loadu_si128( neighborhood_index_table_ptr + idx3);

            _MM_TRANSPOSE4_PS(idx_128_0, idx_128_1, idx_128_2, idx_128_3);

            vec0 = _mm256_i32gather_pd(lrd_score_table_ptr, idx_128_0, sizeof(double) );

            vec1 = _mm256_i32gather_pd(lrd_score_table_ptr, idx_128_1, sizeof(double) );

            vec2 = _mm256_i32gather_pd(lrd_score_table_ptr, idx_128_2, sizeof(double) );

            vec3 = _mm256_i32gather_pd(lrd_score_table_ptr, idx_128_3, sizeof(double) );

            sum0 = _mm256_hadd_pd(vec0, vec1);
            sum1 = _mm256_hadd_pd(vec2, vec3);

            res0 = _mm256_blend_pd(sum0, sum1, 12);
            res1 = _mm256_permute2f128_pd(sum0, sum1, 33);

            result0_3 = _mm256_add_pd(result0_3, _mm256_add_pd(res0, res1));




            idx_128_4 = _mm_loadu_si128( neighborhood_index_table_ptr + idx4);

            idx_128_5 = _mm_loadu_si128( neighborhood_index_table_ptr + idx5);

            idx_128_6 = _mm_loadu_si128( neighborhood_index_table_ptr + idx6);

            idx_128_7 = _mm_loadu_si128( neighborhood_index_table_ptr + idx7);

            _MM_TRANSPOSE4_PS(idx_128_4, idx_128_5, idx_128_6, idx_128_7);

            vec4 = _mm256_i32gather_pd(lrd_score_table_ptr, idx_128_4, sizeof(double) );

            vec5 = _mm256_i32gather_pd(lrd_score_table_ptr, idx_128_5, sizeof(double) );

            vec6 = _mm256_i32gather_pd(lrd_score_table_ptr, idx_128_6, sizeof(double) );

            vec7 = _mm256_i32gather_pd(lrd_score_table_ptr, idx_128_7, sizeof(double) );

            sum2 = _mm256_hadd_pd(vec4, vec5);
            sum3 = _mm256_hadd_pd(vec6, vec7);

            res2 = _mm256_blend_pd(sum2, sum3, 12);
            res3 = _mm256_permute2f128_pd(sum2, sum3, 33);

            result4_7 = _mm256_add_pd(result4_7, _mm256_add_pd(res2, res3));

        }

        if(i==0) {
            _mm256_store_pd(print_vec_1, result0_3);
            _mm256_store_pd(print_vec_2, result4_7);
            printf("%lf, %lf, %lf, %lf, ", print_vec_1[0], print_vec_1[1], print_vec_1[2], print_vec_1[3] );
            printf("%lf, %lf, %lf, %lf\n", print_vec_2[0], print_vec_2[1], print_vec_2[2], print_vec_2[3] );
        }


        printf("\nHelloa!\n");

        // collect remaining stuff
        for( ; j < k; ++j){

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            /*
            __m256d offset = _mm256_set_pd(neighborhood_index_table_ptr[idx0], neighborhood_index_table_ptr[idx1], neighborhood_index_table_ptr[idx2], neighborhood_index_table_ptr[idx3]);
            temp =  _mm256_i64gather_pd(lrd_score_table_ptr, offset, 8);
            result0_3 = _mm256_add_pd(result0_3, temp);*/

            __m128i offset1 = _mm_set_epi32(neighborhood_index_table_ptr[idx0], neighborhood_index_table_ptr[idx1], neighborhood_index_table_ptr[idx2], neighborhood_index_table_ptr[idx3]);
            temp1 = _mm256_i32gather_pd(lrd_score_table_ptr, offset1, sizeof(double) );
            result0_3 = _mm256_add_pd(result0_3, temp1);

            __m128i offset2 = _mm_set_epi32(neighborhood_index_table_ptr[idx4], neighborhood_index_table_ptr[idx5], neighborhood_index_table_ptr[idx6], neighborhood_index_table_ptr[idx7]);
            temp2 = _mm256_i32gather_pd(lrd_score_table_ptr, offset2, sizeof(double) );
            result4_7 = _mm256_add_pd(result4_7, temp2);

        }
        printf("\nHellob!\n");

        //idx_128_0 = _mm_loadu_si128( i );
        //denom0_3 = _mm256_i32gather_pd( lrd_score_table_ptr, idx_128_0, sizeof(double) );

        denom0_3 = _mm256_loadu_pd( lrd_score_table_ptr+i);

        //idx_128_1 = _mm_loadu_si128( i+4 );
        //denom4_7 = _mm256_i32gather_pd( lrd_score_table_ptr, idx_128_1, sizeof(double) );

        denom4_7 = _mm256_loadu_pd( lrd_score_table_ptr+i+4);

        printf("\nHelloc!\n");

        denom0_3 = _mm256_mul_pd(denom0_3, mul_k);
        denom4_7 = _mm256_mul_pd(denom4_7, mul_k);

        printf("\nHellod!\n");

        result0_3 = _mm256_div_pd(result0_3, denom0_3);
        result4_7 = _mm256_div_pd(result4_7, denom4_7);

        printf("\nHelloe!\n");

        _mm256_storeu_pd(lof_score_table_ptr+i, result0_3);
        _mm256_storeu_pd(lof_score_table_ptr+i+4, result4_7);

        printf("\nHellof!\n");

    }

    printf("\nHellog!\n");

    // collect remaining i: ---------------------------------------------

    for ( ; i < num_pts; i++) {

        lrd_neighs_sum = 0;
        idx_base = i*k;
        j = 0;
        for ( ; j + 4 < k; j += 4) {

            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 1]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 2]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 3]];

        }

        // collect remaining stuff
        for( ; j < k; ++j){

            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[ neighborhood_index_table_ptr[idx]];

        }

        denom = lrd_score_table_ptr[i]*k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }

    /*

    // collect remaining i: ---------------------------------------------

    for ( ; i < num_pts; i++) {

        lrd_neighs_sum = 0;
        idx_base = i*k;
        j = 0;
        for ( ; j + 4 < k; j += 4) {

            idx = idx_base + j;
            vec = _mm256_load_pd(lrd_score_table_ptr + neighborhood_index_table_ptr[idx]);
            lrd_neighs_sum += sum_double_avx(vec);

        }

        printf("\nHelloh!\n");

        // collect remaining stuff
        for( ; j < k; ++j){

            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[ neighborhood_index_table_ptr[idx]];

        }

        printf("\nHelloi!\n");

        denom = lrd_score_table_ptr[i]*k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;

        printf("\nHelloj!\n");
    }*/

}


void ComputeLocalOutlierFactor_2(int k, int num_pts,
                                 const double* lrd_score_table_ptr,
                                 const double* lrd_score_neigh_table_ptr,
                                 double* lof_score_table_ptr){

    //printf("New LOF Function Run\n\n");

    /**
     *@param lrd_score_table_ptr_tmp: an array of size num_pts
     *                                lrd_score_table_ptr[ i ] contains lrd score of point i
     *
     *@param lrd_score_neigh_table_tmp_ptr: an array of size num_pts * k
     *                                 lrd_score_neigh_table_ptr[ i*k + j ], where i < num_pts, j < k
     *                                 stores the lrd score of the jth (of k nearest neighbors) of the point i
     */

    for(int i=0; i<num_pts; ++i){

        double lrd_neighs_sum = 0;
        for(int j=0; j<k; ++j){
            lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j
        //printf("%d \n", lrd_neighs_sum);
        lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i
}   // ComputeLocalOutlierFactor_Pipeline2


//Half AVX
void ComputeLocalOutlierFactor_2_unroll_AVX_fixed(int k, int num_pts,
                                            const double* lrd_score_table_ptr,
                                            const double* lrd_score_neigh_table_ptr,
                                            double* lof_score_table_ptr){

    /**
     *@param lrd_score_table_ptr_tmp: an array of size num_pts
     *                                lrd_score_table_ptr[ i ] contains lrd score of point i
     *
     *@param lrd_score_neigh_table_tmp_ptr: an array of size num_pts * k
     *                                 lrd_score_neigh_table_ptr[ i*k + j ], where i < num_pts, j < k
     *                                 stores the lrd score of the jth (of k nearest neighbors) of the point i
     */

    //printf("New Vectorized LOF fix Function Run\n\n");

    int idx_base_0, idx_base_1, idx_base_2, idx_base_3, idx_base_4, idx_base_5, idx_base_6, idx_base_7;
    int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;

    __m256d vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec, temp1, temp2;
    __m256d result0_3, result4_7;
    __m256d res0, res1, res2, res3, sum0, sum1, sum2, sum3;
    __m256d mul_k = _mm256_set1_pd(k);
    __m256d denom0_3, denom4_7;
    __m128i idx_128_0, idx_128_1, idx_128_2, idx_128_3, idx_128_4, idx_128_5, idx_128_6, idx_128_7;

    int i=0;

    for(; i+7<num_pts; i+=8){

        //result0_3 = _mm256_set1_pd(0.0);
        //result4_7 = _mm256_set1_pd(0.0);

        double a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0;

        idx_base_0 = i * k;
        idx_base_1 = (i + 1) * k;
        idx_base_2 = (i + 2) * k;
        idx_base_3 = (i + 3) * k;
        idx_base_4 = (i + 4) * k;
        idx_base_5 = (i + 5) * k;
        idx_base_6 = (i + 6) * k;
        idx_base_7 = (i + 7) * k;

        double lrd_neighs_sum = 0;
        int j=0;
        for(; j+3<k; j+=4){

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            /*
            idx_128_0 = _mm_set_epi32(idx0, idx0 + 1, idx0 + 2, idx0 + 3);
            idx_128_1 = _mm_set_epi32(idx1, idx1 + 1, idx1 + 2, idx1 + 3);
            idx_128_2 = _mm_set_epi32(idx2, idx2 + 1, idx2 + 2, idx2 + 3);
            idx_128_3 = _mm_set_epi32(idx3, idx3 + 1, idx3 + 2, idx3 + 3);

            vec0 = _mm256_i32gather_pd(lrd_score_neigh_table_ptr, idx_128_0, sizeof(double) );
            vec1 = _mm256_i32gather_pd(lrd_score_neigh_table_ptr, idx_128_1, sizeof(double) );
            vec2 = _mm256_i32gather_pd(lrd_score_neigh_table_ptr, idx_128_2, sizeof(double) );
            vec3 = _mm256_i32gather_pd(lrd_score_neigh_table_ptr, idx_128_3, sizeof(double) );

            /*

            sum0 = _mm256_hadd_pd(vec0, vec1);
            sum1 = _mm256_hadd_pd(vec2, vec3);

            res0 = _mm256_blend_pd(sum0, sum1, 12);
            res1 = _mm256_permute2f128_pd(sum0, sum1, 33);

            result0_3 = _mm256_add_pd(result0_3, _mm256_add_pd(res0, res1));*/

            vec0 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx0);
            vec1 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx1);
            vec2 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx2);
            vec3 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx3);

            a += sum_double_avx(vec0);
            b += sum_double_avx(vec1);
            c += sum_double_avx(vec2);
            d += sum_double_avx(vec3);

            //res0 =  _mm256_set_pd(a, b, c, d);

            //result0_3 = _mm256_add_pd(result0_3, res0);



            /*
            idx_128_4 = _mm_set_epi32(idx4, idx4 + 1, idx4 + 2, idx4 + 3);
            idx_128_5 = _mm_set_epi32(idx5, idx5 + 1, idx5 + 2, idx5 + 3);
            idx_128_6 = _mm_set_epi32(idx6, idx6 + 1, idx6 + 2, idx6 + 3);
            idx_128_7 = _mm_set_epi32(idx7, idx7 + 1, idx7 + 2, idx7 + 3);

            vec4 = _mm256_i32gather_pd(lrd_score_neigh_table_ptr, idx_128_4, sizeof(double) );
            vec5 = _mm256_i32gather_pd(lrd_score_neigh_table_ptr, idx_128_5, sizeof(double) );
            vec6 = _mm256_i32gather_pd(lrd_score_neigh_table_ptr, idx_128_6, sizeof(double) );
            vec7 = _mm256_i32gather_pd(lrd_score_neigh_table_ptr, idx_128_7, sizeof(double) );*/

            vec4 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx4);
            vec5 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx5);
            vec6 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx6);
            vec7 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx7);

            /*

            sum2 = _mm256_hadd_pd(vec4, vec5);
            sum3 = _mm256_hadd_pd(vec6, vec7);

            res2 = _mm256_blend_pd(sum2, sum3, 12);
            res3 = _mm256_permute2f128_pd(sum2, sum3, 33);

            result4_7 = _mm256_add_pd(result4_7, _mm256_add_pd(res2, res3));*/

            e += sum_double_avx(vec4);
            f += sum_double_avx(vec5);
            g += sum_double_avx(vec6);
            h += sum_double_avx(vec7);

            //res1 =  _mm256_set_pd(a, b, c, d);

            //result4_7 = _mm256_add_pd(result4_7, res1);


            //lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j

        // collect remaining stuff
        for( ; j < k; ++j){

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            a += lrd_score_neigh_table_ptr[idx0];
            b += lrd_score_neigh_table_ptr[idx1];
            c += lrd_score_neigh_table_ptr[idx2];
            d += lrd_score_neigh_table_ptr[idx3];
            e += lrd_score_neigh_table_ptr[idx4];
            f += lrd_score_neigh_table_ptr[idx5];
            g += lrd_score_neigh_table_ptr[idx6];
            h += lrd_score_neigh_table_ptr[idx7];

        }


        //printf("%d, %d, %d, %d, %d, %d, %d, %d\n", a, b, c, d, e, f, g, h);

        /*

        result0_3 =  _mm256_set_pd(a, b, c, d);
        result4_7 =  _mm256_set_pd(e, f, g, h);

        denom0_3 = _mm256_loadu_pd( lrd_score_table_ptr+i);
        denom4_7 = _mm256_loadu_pd( lrd_score_table_ptr+i+4);

        denom0_3 = _mm256_mul_pd(denom0_3, mul_k);
        denom4_7 = _mm256_mul_pd(denom4_7, mul_k);

        result0_3 = _mm256_div_pd(result0_3, denom0_3);
        result4_7 = _mm256_div_pd(result4_7, denom4_7);

        _mm256_storeu_pd(lof_score_table_ptr+i, result0_3);
        _mm256_storeu_pd(lof_score_table_ptr+i+4, result4_7);*/



        lof_score_table_ptr[i] = a / (lrd_score_table_ptr[i] * k);
        lof_score_table_ptr[i+1] = b / (lrd_score_table_ptr[i+1] * k);
        lof_score_table_ptr[i+2] = c / (lrd_score_table_ptr[i+2] * k);
        lof_score_table_ptr[i+3] = d / (lrd_score_table_ptr[i+3] * k);
        lof_score_table_ptr[i+4] = e / (lrd_score_table_ptr[i+4] * k);
        lof_score_table_ptr[i+5] = f / (lrd_score_table_ptr[i+5] * k);
        lof_score_table_ptr[i+6] = g / (lrd_score_table_ptr[i+6] * k);
        lof_score_table_ptr[i+7] = h / (lrd_score_table_ptr[i+7] * k);

        /*
        if(i==0) {
            printf("k=%i, j=%i\n", k, j);
            double print_vec_1[4], print_vec_2[4];
            _mm256_store_pd(print_vec_1, result0_3);
            _mm256_store_pd(print_vec_2, result4_7);
            printf("%d, %d, %d, %d, ", print_vec_1[0], print_vec_1[1], print_vec_1[2], print_vec_1[3] );
            printf("%d, %d, %d, %d\n", print_vec_2[0], print_vec_2[1], print_vec_2[2], print_vec_2[3] );
        }

        denom0_3 = _mm256_loadu_pd( lrd_score_table_ptr+i);
        denom4_7 = _mm256_loadu_pd( lrd_score_table_ptr+i+4);

        denom0_3 = _mm256_mul_pd(denom0_3, mul_k);
        denom4_7 = _mm256_mul_pd(denom4_7, mul_k);

        result0_3 = _mm256_div_pd(result0_3, denom0_3);
        result4_7 = _mm256_div_pd(result4_7, denom4_7);

        _mm256_storeu_pd(lof_score_table_ptr+i, result0_3);
        _mm256_storeu_pd(lof_score_table_ptr+i+4, result4_7);*/

        //lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

    //Remaining i
    for(; i<num_pts; ++i){

        double lrd_neighs_sum = 0;
        for(int j=0; j<k; ++j){
            lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j
        lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

}   // ComputeLocalOutlierFactor_Pipeline2



//AVX Slow
void ComputeLocalOutlierFactor_2_AVX_working(int k, int num_pts,
                                             const double* lrd_score_table_ptr,
                                             const double* lrd_score_neigh_table_ptr,
                                             double* lof_score_table_ptr){

    /**
     *@param lrd_score_table_ptr_tmp: an array of size num_pts
     *                                lrd_score_table_ptr[ i ] contains lrd score of point i
     *
     *@param lrd_score_neigh_table_tmp_ptr: an array of size num_pts * k
     *                                 lrd_score_neigh_table_ptr[ i*k + j ], where i < num_pts, j < k
     *                                 stores the lrd score of the jth (of k nearest neighbors) of the point i
     */

    //printf("New Vectorized LOF fix Function Run\n\n");

    int idx_base_0, idx_base_1, idx_base_2, idx_base_3, idx_base_4, idx_base_5, idx_base_6, idx_base_7;
    int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;

    __m256d vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec, temp1, temp2;
    __m256d res0, res1, res2, res3, sum0, sum1, sum2, sum3;
    __m256d mul_k = _mm256_set1_pd(k);
    __m256d denom0_3, denom4_7;
    __m128i idx_128_0, idx_128_1, idx_128_2, idx_128_3, idx_128_4, idx_128_5, idx_128_6, idx_128_7;

    int i=0;

    for(; i+7<num_pts; i+=8){

        double a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0;

        idx_base_0 = i * k;
        idx_base_1 = (i + 1) * k;
        idx_base_2 = (i + 2) * k;
        idx_base_3 = (i + 3) * k;
        idx_base_4 = (i + 4) * k;
        idx_base_5 = (i + 5) * k;
        idx_base_6 = (i + 6) * k;
        idx_base_7 = (i + 7) * k;

        double lrd_neighs_sum = 0;
        int j=0;
        for(; j+3<k; j+=4){

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            vec0 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx0);
            vec1 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx1);
            vec2 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx2);
            vec3 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx3);

            a += sum_double_avx(vec0);
            b += sum_double_avx(vec1);
            c += sum_double_avx(vec2);
            d += sum_double_avx(vec3);

            vec4 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx4);
            vec5 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx5);
            vec6 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx6);
            vec7 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx7);

            e += sum_double_avx(vec4);
            f += sum_double_avx(vec5);
            g += sum_double_avx(vec6);
            h += sum_double_avx(vec7);


            //lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j

        // collect remaining stuff
        for( ; j < k; ++j){

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            a += lrd_score_neigh_table_ptr[idx0];
            b += lrd_score_neigh_table_ptr[idx1];
            c += lrd_score_neigh_table_ptr[idx2];
            d += lrd_score_neigh_table_ptr[idx3];
            e += lrd_score_neigh_table_ptr[idx4];
            f += lrd_score_neigh_table_ptr[idx5];
            g += lrd_score_neigh_table_ptr[idx6];
            h += lrd_score_neigh_table_ptr[idx7];

        }


        //printf("%d, %d, %d, %d, %d, %d, %d, %d\n", a, b, c, d, e, f, g, h);


        __m256d result0_3, result4_7;

        double print_vec_1[4] = {a,b,c,d};
        double print_vec_2[4] = {e,f,g,h};

        //result0_3 =  _mm256_set_pd((const double)a, (const double)b, (const double)c, (const double)d);
        //result4_7 =  _mm256_set_pd((const double)e, (const double)f, (const double)g, (const double)h);

        result0_3 = _mm256_loadu_pd(print_vec_1);
        result4_7 = _mm256_loadu_pd(print_vec_2);

        /*

        if(i==0) {
            printf("k=%i, j=%i\n", k, j);
            double print_vec_1[4], print_vec_2[4];
            _mm256_store_pd(print_vec_1, result0_3);
            _mm256_store_pd(print_vec_2, result4_7);
            printf("Result - %f, %f, %f, %f\n", print_vec_1[0], print_vec_1[1], print_vec_1[2], print_vec_1[3] );
            printf("Actual - %f, %f, %f, %f\n", a, b, c, d );
            printf("Result - %f, %f, %f, %f\n", print_vec_2[0], print_vec_2[1], print_vec_2[2], print_vec_2[3] );
            printf("Actual - %f, %f, %f, %f\n", e, f, g, h );

            printf("\n%d, %d, %d, %d\n", print_vec_1[0], print_vec_1[1], print_vec_1[2], print_vec_1[3] );
            printf("%d, %d, %d, %d\n", a, b, c, d );
            printf("%d, %d, %d, %d\n", print_vec_2[0], print_vec_2[1], print_vec_2[2], print_vec_2[3] );
            printf("%d, %d, %d, %d\n", e, f, g, h );
        }*/

        denom0_3 = _mm256_loadu_pd( lrd_score_table_ptr+i);
        denom4_7 = _mm256_loadu_pd( lrd_score_table_ptr+i+4);

        denom0_3 = _mm256_mul_pd(denom0_3, mul_k);
        denom4_7 = _mm256_mul_pd(denom4_7, mul_k);

        result0_3 = _mm256_div_pd(result0_3, denom0_3);
        result4_7 = _mm256_div_pd(result4_7, denom4_7);

        _mm256_storeu_pd(lof_score_table_ptr+i, result0_3);
        _mm256_storeu_pd(lof_score_table_ptr+i+4, result4_7);

        /*

        if(i==0) {
            printf("k=%i, j=%i\n", k, j);
            double print_vec_1[4], print_vec_2[4];
            _mm256_store_pd(print_vec_1, result0_3);
            _mm256_store_pd(print_vec_2, result4_7);
            printf("Result - %f, %f, %f, %f\n", print_vec_1[0], print_vec_1[1], print_vec_1[2], print_vec_1[3] );
            printf("Actual - %f, %f, %f, %f\n", lof_score_table_ptr[i], lof_score_table_ptr[i+1], lof_score_table_ptr[i+2], lof_score_table_ptr[i+3] );
            printf("Result - %f, %f, %f, %f\n", print_vec_2[0], print_vec_2[1], print_vec_2[2], print_vec_2[3] );
            printf("Actual - %f, %f, %f, %f\n", lof_score_table_ptr[i+4], lof_score_table_ptr[i+5], lof_score_table_ptr[i+6], lof_score_table_ptr[i+7] );
        }*/



        /*
        lof_score_table_ptr[i] = a / (lrd_score_table_ptr[i] * k);
        lof_score_table_ptr[i+1] = b / (lrd_score_table_ptr[i+1] * k);
        lof_score_table_ptr[i+2] = c / (lrd_score_table_ptr[i+2] * k);
        lof_score_table_ptr[i+3] = d / (lrd_score_table_ptr[i+3] * k);
        lof_score_table_ptr[i+4] = e / (lrd_score_table_ptr[i+4] * k);
        lof_score_table_ptr[i+5] = f / (lrd_score_table_ptr[i+5] * k);
        lof_score_table_ptr[i+6] = g / (lrd_score_table_ptr[i+6] * k);
        lof_score_table_ptr[i+7] = h / (lrd_score_table_ptr[i+7] * k);*/

        //lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

    //Remaining i
    for(; i<num_pts; ++i){

        double lrd_neighs_sum = 0;
        for(int j=0; j<k; ++j){
            lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j
        lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

}   // ComputeLocalOutlierFactor_Pipeline2



void ComputeLocalOutlierFactor_2_AVX_Faster(int k, int num_pts,
                                             const double* lrd_score_table_ptr,
                                             const double* lrd_score_neigh_table_ptr,
                                             double* lof_score_table_ptr){

    /**
     *@param lrd_score_table_ptr_tmp: an array of size num_pts
     *                                lrd_score_table_ptr[ i ] contains lrd score of point i
     *
     *@param lrd_score_neigh_table_tmp_ptr: an array of size num_pts * k
     *                                 lrd_score_neigh_table_ptr[ i*k + j ], where i < num_pts, j < k
     *                                 stores the lrd score of the jth (of k nearest neighbors) of the point i
     */

    //printf("New Vectorized LOF fix Function Run\n\n");

    int idx_base_0, idx_base_1, idx_base_2, idx_base_3, idx_base_4, idx_base_5, idx_base_6, idx_base_7;
    int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;

    __m256d vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec, temp1, temp2;
    __m256d res0, res1, res2, res3, sum0, sum1, sum2, sum3;
    __m256d mul_k = _mm256_set1_pd(k);
    __m256d denom0_3, denom4_7;
    __m128i idx_128_0, idx_128_1, idx_128_2, idx_128_3, idx_128_4, idx_128_5, idx_128_6, idx_128_7;

    int i=0;

    __m256d result0_3, result4_7;

    for(; i+7<num_pts; i+=8){

        result0_3 = _mm256_set1_pd(0.0);
        result4_7 = _mm256_set1_pd(0.0);

        double a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0;

        idx_base_0 = i * k;
        idx_base_1 = (i + 1) * k;
        idx_base_2 = (i + 2) * k;
        idx_base_3 = (i + 3) * k;
        idx_base_4 = (i + 4) * k;
        idx_base_5 = (i + 5) * k;
        idx_base_6 = (i + 6) * k;
        idx_base_7 = (i + 7) * k;

        double lrd_neighs_sum = 0;
        int j=0;
        for(; j+3<k; j+=4){

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            vec0 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx0);
            vec1 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx1);
            vec2 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx2);
            vec3 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx3);

            sum0 = _mm256_hadd_pd(vec0, vec1);
            sum1 = _mm256_hadd_pd(vec2, vec3);

            res0 = _mm256_blend_pd(sum0, sum1, 12);
            res1 = _mm256_permute2f128_pd(sum0, sum1, 33);

            result0_3 = _mm256_add_pd(result0_3, _mm256_add_pd(res0, res1));

            vec4 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx4);
            vec5 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx5);
            vec6 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx6);
            vec7 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx7);

            sum2 = _mm256_hadd_pd(vec4, vec5);
            sum3 = _mm256_hadd_pd(vec6, vec7);

            res2 = _mm256_blend_pd(sum2, sum3, 12);
            res3 = _mm256_permute2f128_pd(sum2, sum3, 33);

            result4_7 = _mm256_add_pd(result4_7, _mm256_add_pd(res2, res3));

            //lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j

        // collect remaining stuff
        for (; j < k; ++j) {

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            __m128i offset1 = _mm_set_epi32( idx3, idx2, idx1, idx0);
            temp1 =  _mm256_i32gather_pd( lrd_score_neigh_table_ptr, offset1, sizeof(double) );
            result0_3 = _mm256_add_pd(result0_3, temp1);

            __m128i offset2 = _mm_set_epi32(idx7, idx6, idx5, idx4);
            temp2 =  _mm256_i32gather_pd( lrd_score_neigh_table_ptr, offset2, sizeof(double) );
            result4_7 = _mm256_add_pd(result4_7, temp2);
        }

        denom0_3 = _mm256_loadu_pd( lrd_score_table_ptr+i);
        denom4_7 = _mm256_loadu_pd( lrd_score_table_ptr+i+4);

        denom0_3 = _mm256_mul_pd(denom0_3, mul_k);
        denom4_7 = _mm256_mul_pd(denom4_7, mul_k);

        result0_3 = _mm256_div_pd(result0_3, denom0_3);
        result4_7 = _mm256_div_pd(result4_7, denom4_7);

        _mm256_storeu_pd(lof_score_table_ptr+i, result0_3);
        _mm256_storeu_pd(lof_score_table_ptr+i+4, result4_7);

        //lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

    //Remaining i
    for(; i<num_pts; ++i){

        double lrd_neighs_sum = 0;
        for(int j=0; j<k; ++j){
            lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j
        lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

}   // ComputeLocalOutlierFactor_Pipeline2




double ComputeLocalOutlierFactor_2_AVX_Fastest(int k, int num_pts,
                                                  const double* lrd_score_table_ptr,
                                                  const double* lrd_score_neigh_table_ptr,
                                                  double* lof_score_table_ptr){

    /**
     *@param lrd_score_table_ptr_tmp: an array of size num_pts
     *                                lrd_score_table_ptr[ i ] contains lrd score of point i
     *
     *@param lrd_score_neigh_table_tmp_ptr: an array of size num_pts * k
     *                                 lrd_score_neigh_table_ptr[ i*k + j ], where i < num_pts, j < k
     *                                 stores the lrd score of the jth (of k nearest neighbors) of the point i
     */

    //printf("New Vectorized LOF fix Function Run\n\n");

    int idx_base_0, idx_base_1, idx_base_2, idx_base_3, idx_base_4, idx_base_5, idx_base_6, idx_base_7;
    int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;

    __m256d vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, temp1, temp2;
    __m256d res0, res1, res2, res3, sum0, sum1, sum2, sum3;
    __m256d mul_k = _mm256_set1_pd(k);
    __m256d denom0_3, denom4_7;
    __m128i idx_128_0, idx_128_1, idx_128_2, idx_128_3, idx_128_4, idx_128_5, idx_128_6, idx_128_7;

    __m256d vec0_sum, vec1_sum, vec2_sum, vec3_sum, vec4_sum, vec5_sum, vec6_sum, vec7_sum;

    int i=0;

    __m256d result0_3, result4_7;

    for(; i+7<num_pts; i+=8){

        vec0_sum = _mm256_set1_pd(0.0);
        vec1_sum = _mm256_set1_pd(0.0);
        vec2_sum = _mm256_set1_pd(0.0);
        vec3_sum = _mm256_set1_pd(0.0);
        vec4_sum = _mm256_set1_pd(0.0);
        vec5_sum = _mm256_set1_pd(0.0);
        vec6_sum = _mm256_set1_pd(0.0);
        vec7_sum = _mm256_set1_pd(0.0);

        double a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0;

        idx_base_0 = i * k;
        idx_base_1 = (i + 1) * k;
        idx_base_2 = (i + 2) * k;
        idx_base_3 = (i + 3) * k;
        idx_base_4 = (i + 4) * k;
        idx_base_5 = (i + 5) * k;
        idx_base_6 = (i + 6) * k;
        idx_base_7 = (i + 7) * k;

        double lrd_neighs_sum = 0;
        int j=0;
        for(; j+3<k; j+=4){

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            vec0 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx0);
            vec1 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx1);
            vec2 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx2);
            vec3 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx3);

            vec0_sum = _mm256_add_pd(vec0_sum, vec0);
            vec1_sum = _mm256_add_pd(vec1_sum, vec1);
            vec2_sum = _mm256_add_pd(vec2_sum, vec2);
            vec3_sum = _mm256_add_pd(vec3_sum, vec3);


            vec4 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx4);
            vec5 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx5);
            vec6 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx6);
            vec7 = _mm256_loadu_pd(lrd_score_neigh_table_ptr+idx7);

            vec4_sum = _mm256_add_pd(vec4_sum, vec4);
            vec5_sum = _mm256_add_pd(vec5_sum, vec5);
            vec6_sum = _mm256_add_pd(vec6_sum, vec6);
            vec7_sum = _mm256_add_pd(vec7_sum, vec7);

            //lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j

        sum0 = _mm256_hadd_pd(vec0_sum, vec1_sum);
        sum1 = _mm256_hadd_pd(vec2_sum, vec3_sum);

        res0 = _mm256_blend_pd(sum0, sum1, 12);
        res1 = _mm256_permute2f128_pd(sum0, sum1, 33);

        result0_3 = _mm256_add_pd(res0, res1);


        sum2 = _mm256_hadd_pd(vec4_sum, vec5_sum);
        sum3 = _mm256_hadd_pd(vec6_sum, vec7_sum);

        res2 = _mm256_blend_pd(sum2, sum3, 12);
        res3 = _mm256_permute2f128_pd(sum2, sum3, 33);

        result4_7 = _mm256_add_pd(res2, res3);

        // collect remaining stuff
        for (; j < k; ++j) {

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;
            idx4 = idx_base_4 + j;
            idx5 = idx_base_5 + j;
            idx6 = idx_base_6 + j;
            idx7 = idx_base_7 + j;

            __m128i offset1 = _mm_set_epi32( idx3, idx2, idx1, idx0);
            temp1 =  _mm256_i32gather_pd( lrd_score_neigh_table_ptr, offset1, sizeof(double) );
            result0_3 = _mm256_add_pd(result0_3, temp1);

            __m128i offset2 = _mm_set_epi32(idx7, idx6, idx5, idx4);
            temp2 =  _mm256_i32gather_pd( lrd_score_neigh_table_ptr, offset2, sizeof(double) );
            result4_7 = _mm256_add_pd(result4_7, temp2);
        }

        denom0_3 = _mm256_loadu_pd( lrd_score_table_ptr+i);
        denom4_7 = _mm256_loadu_pd( lrd_score_table_ptr+i+4);

        denom0_3 = _mm256_mul_pd(denom0_3, mul_k);
        denom4_7 = _mm256_mul_pd(denom4_7, mul_k);

        result0_3 = _mm256_div_pd(result0_3, denom0_3);
        result4_7 = _mm256_div_pd(result4_7, denom4_7);

        _mm256_storeu_pd(lof_score_table_ptr+i, result0_3);
        _mm256_storeu_pd(lof_score_table_ptr+i+4, result4_7);

        //lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

    //Remaining i
    for(; i<num_pts; ++i){

        double lrd_neighs_sum = 0;
        for(int j=0; j<k; ++j){
            lrd_neighs_sum += lrd_score_neigh_table_ptr[ i*k + j ];
        }  // for j
        lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

    return num_pts * (2 + k);
}   // ComputeLocalOutlierFactor_Pipeline2


void lof_verify_an( int num_pts, int pnt_idx, int k){

    // INITIALIZE INPUT
    double* dist_k_neighborhood_index = XmallocMatrixDoubleRandom( num_pts, k );
    double* k_distance_index = XmallocVectorDoubleRandom( num_pts );
    int* k_neighborhood_index = XmallocMatrixIntRandom( num_pts, k, num_pts-1 );

    double* lrd_score_table_ptr = XmallocVectorDoubleRandom( num_pts );

    double* lrd_score_neigh_table_ptr = XmallocMatrixDoubleRandom( num_pts, k );

    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);
    double* lof_score_table_ptr_true = XmallocVectorDouble(num_pts);

    double tol = 0.0001;

    //Baseline
    ComputeLocalOutlierFactor_2(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr_true);

    //Half AVX
    //ComputeLocalOutlierFactor_2_unroll_AVX_fixed(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    //AVX Slow
    //ComputeLocalOutlierFactor_2_AVX_working(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    //AVX Faster
    //ComputeLocalOutlierFactor_2_AVX_Faster(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    //AVX Fastest
    ComputeLocalOutlierFactor_2_AVX_Fastest(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

    int ver_lof = test_double_arrays( num_pts, tol, lof_score_table_ptr_true, lof_score_table_ptr);

    if( ver_lof == 0 ){
        printf("Resulting LOF scores are different\n");
    } else {
        printf( "LOF passed verification\n" );
    }

    if(fabs( lof_score_table_ptr[pnt_idx] - lof_score_table_ptr_true[pnt_idx] ) < 0.001){
        printf("PASSED VERIFICATION");
    } else {
        printf("DID NOT PASS VERIFICATION: should %lf is %lf", lof_score_table_ptr_true[pnt_idx], lof_score_table_ptr[pnt_idx]);
    }
}



void lof_driver( int num_pts, int pnt_idx, int k){

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &ComputeLocalOutlierFactor_2;
    fun_array[1] = &ComputeLocalOutlierFactor_2_unroll_AVX_fixed;
    fun_array[2] = &ComputeLocalOutlierFactor_2_AVX_working;
    fun_array[3] = &ComputeLocalOutlierFactor_2_AVX_Faster;
    fun_array[4] = &ComputeLocalOutlierFactor_2_AVX_Fastest;

    char* fun_names[NUM_FUNCTIONS] = {"Baseline", "Half AVX", "AVX Slow", "AVX Faster", "AVX Fastest"};

    myInt64 start, end;
    double cycles;


//    FILE* results_file = open_with_error_check("../lrd_benchmark_k10_full.txt", "a");

    // INITIALIZE RANDOM INPUT

    // INITIALIZE INPUT
    double* dist_k_neighborhood_index = XmallocMatrixDoubleRandom( num_pts, k );
    double* k_distance_index = XmallocVectorDoubleRandom( num_pts );
    int* k_neighborhood_index = XmallocMatrixIntRandom( num_pts, k, num_pts-1 );

    double* lrd_score_table_ptr = XmallocVectorDoubleRandom( num_pts );

    double* lrd_score_neigh_table_ptr = XmallocMatrixDoubleRandom( num_pts, k );

    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);
    double* lof_score_table_ptr_true = XmallocVectorDouble(num_pts);

    double tol = 0.0001;

    //Baseline
    ComputeLocalOutlierFactor_2(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr_true);

    double totalCycles = 0;

    for (int fun_index = 0; fun_index < NUM_FUNCTIONS; fun_index++) {
        free(lof_score_table_ptr);
        lof_score_table_ptr = XmallocVectorDouble(num_pts);


        // VERIFICATION : ------------------------------------------------------------------------
        (*fun_array[fun_index])(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);

        int ver_lof = test_double_arrays( num_pts, tol, lof_score_table_ptr_true, lof_score_table_ptr);

        if( ver_lof == 0 ){
            printf("Resulting LOF scores are different\n");
        } else {
            //printf( "LOF passed verification\n" );
        }

        if(fabs( lof_score_table_ptr[pnt_idx] - lof_score_table_ptr_true[pnt_idx] ) < 0.001){
            //printf("PASSED VERIFICATION\n");
        } else {
            printf("DID NOT PASS VERIFICATION: should %lf is %lf\n", lof_score_table_ptr_true[pnt_idx], lof_score_table_ptr[pnt_idx]);
        }

        double numRuns = 100;

        start = start_tsc();
        for (size_t i = 0; i < numRuns; i++) {
            (*fun_array[fun_index])(k, num_pts, lrd_score_table_ptr, lrd_score_neigh_table_ptr, lof_score_table_ptr);
        }
        end = stop_tsc(start);

        cycles = (double) end/numRuns;

        totalCycles += cycles;


        double flops = num_pts * (2 + k);
        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        printf("%s:  num_pts:%d k:%d cycles:%lf perf:%lf \n\n", fun_names[fun_index], num_pts, k, cycles, perf);
//        fprintf(results_file, "%s, %d, %d, %lf, %lf\n", fun_names[fun_index], num_pts, k, cycles, perf);
    }

    printf("-------------\n");

    free(lrd_score_table_ptr);
    free(lrd_score_neigh_table_ptr);
    free(lof_score_table_ptr);
    free(lof_score_table_ptr_true);
//    fclose(results_file);
}