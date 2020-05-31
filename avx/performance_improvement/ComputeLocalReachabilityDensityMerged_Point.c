//
// Created by Sycheva  Anastasia on 21.05.20.
//
#include <math.h>
#include <immintrin.h>

#include "../../include/performance_measurement.h"
#include "../../include/utils.h"
#include "../../include/lof_baseline.h"
#include "../include/ComputeLocalReachabilityDensityMerged_Point.h"
#include "../include/AVX_utils.h"
#include "../../include/tests.h"


void ComputeLocalReachabilityDensityMerged_Point_AVX4( int pnt_idx, int k,
                                                 const double* dist_k_neighborhood_index,
                                                 const double* k_distance_index,
                                                 const int* k_neighborhood_index,
                                                 double* lrd_score_table_ptr ){
    /**
     * Compute lrd score for @param pnt_idx
     * See ComputeLocalReachabilityDensityMerged_Pipeline2 for argument description
     */

    int base_idx, base_idx_j, neigh_idx;
    int neigh_idx0, neigh_idx1, neigh_idx2, neigh_idx3;
    double kdistO, distPo, sum;

    __m128i idx_j_128, neigh_idx_128;
    __m256d kdistO_265, distPo_256, sum_256;

    unsigned int j;
    sum = 0;
    base_idx = pnt_idx * k;

    sum_256 = _mm256_set1_pd(0.0);
    for (j = 0; j + 3 < k; j += 4) {

        int base_idx_j0 =  base_idx + j;

        /*
        // CORRECT BUT SHOULD BE SLOW
        neigh_idx0 = k_neighborhood_index[ base_idx_j0 ];
        neigh_idx1 = k_neighborhood_index[ base_idx_j0+1 ];
        neigh_idx2 = k_neighborhood_index[ base_idx_j0+2 ];
        neigh_idx3 = k_neighborhood_index[ base_idx_j0+3 ];

        kdistO_265 = _mm256_set_pd( k_distance_index[ neigh_idx3 ],
                                    k_distance_index[ neigh_idx2 ],
                                    k_distance_index[ neigh_idx1 ],
                                    k_distance_index[ neigh_idx0 ]);

        distPo_256 = _mm256_set_pd( dist_k_neighborhood_index[base_idx_j0+3],
                                    dist_k_neighborhood_index[base_idx_j0+2],
                                    dist_k_neighborhood_index[base_idx_j0+1],
                                       dist_k_neighborhood_index[base_idx_j0]);
        */
        // POTENTIALLY FASTER, DOES NOT COMPILE
        //idx_j_128 = _mm_set_epi32( base_idx_j0+3, base_idx_j0+2, base_idx_j0+1, base_idx_j0);
        //neigh_idx_128 = _mm_set_epi32( neigh_idx3, neigh_idx2, neigh_idx1, neigh_idx0);
        //neigh_idx_128 = _mm_i32gather_epi32(k_neighborhood_index, idx_j_128, sizeof(int));
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0);
        //distPo_256 = _mm256_i32gather_pd( dist_k_neighborhood_index, idx_j_128, sizeof(double) );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

    }

    sum += sum_double_avx(sum_256); // sum_double_avx( _mm256_max_pd( kdistO_265, distPo_256) );

    for( ; j < k; j++ ){
        base_idx_j = base_idx + j;

        neigh_idx = k_neighborhood_index[ base_idx_j ];
        kdistO = k_distance_index[neigh_idx];   //  dist_k_neighborhood_index[ neigh_idx*k + k-1]
        distPo = dist_k_neighborhood_index[ base_idx_j ];

        sum += kdistO >= distPo ? kdistO : distPo;

    }

    double result = k / sum;
    lrd_score_table_ptr[pnt_idx] = result;

}   // ComputeLocalReachabilityDensityMerged_Point_AVX4


void ComputeLocalReachabilityDensityMerged_Point_AVX4_V2( int pnt_idx, int k,
                                                       const double* dist_k_neighborhood_index,
                                                       const double* k_distance_index,
                                                       const int* k_neighborhood_index,
                                                       double* lrd_score_table_ptr ){
    /**
     * Compute lrd score for @param pnt_idx
     * See ComputeLocalReachabilityDensityMerged_Pipeline2 for argument description
     */

    int base_idx, base_idx_j, neigh_idx;
    int neigh_idx0, neigh_idx1, neigh_idx2, neigh_idx3;
    double kdistO, distPo, sum;

    __m128i idx_j_128, neigh_idx_128;
    __m256d kdistO_265, distPo_256, sum_256;

    unsigned int j;
    sum = 0;
    base_idx = pnt_idx * k;

    sum_256 = _mm256_set1_pd(0.0);
    for (j = 0; j + 3 < k; j += 4) {

        int base_idx_j0 =  base_idx + j;

        // POTENTIALLY FASTER, DOES NOT COMPILE
        //idx_j_128 = _mm_set_epi32( base_idx_j0+3, base_idx_j0+2, base_idx_j0+1, base_idx_j0);
        //neigh_idx_128 = _mm_set_epi32( neigh_idx3, neigh_idx2, neigh_idx1, neigh_idx0);
        //neigh_idx_128 = _mm_i32gather_epi32(k_neighborhood_index, idx_j_128, sizeof(int));
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0);
        //distPo_256 = _mm256_i32gather_pd( dist_k_neighborhood_index, idx_j_128, sizeof(double) );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

    }

    sum += sum_double_avx_2(sum_256); // sum_double_avx( _mm256_max_pd( kdistO_265, distPo_256) );

    for( ; j < k; j++ ){
        base_idx_j = base_idx + j;

        neigh_idx = k_neighborhood_index[ base_idx_j ];
        kdistO = k_distance_index[neigh_idx];   //  dist_k_neighborhood_index[ neigh_idx*k + k-1]
        distPo = dist_k_neighborhood_index[ base_idx_j ];

        sum += kdistO >= distPo ? kdistO : distPo;

    }

    double result = k / sum;
    lrd_score_table_ptr[pnt_idx] = result;

}   // ComputeLocalReachabilityDensityMerged_Point_AVX4

void ComputeLocalReachabilityDensityMerged_Point_AVX8( int pnt_idx, int k,
                                                       const double* dist_k_neighborhood_index,
                                                       const double* k_distance_index,
                                                       const int* k_neighborhood_index,
                                                       double* lrd_score_table_ptr ){
    /**
     * Compute lrd score for @param pnt_idx
     * See ComputeLocalReachabilityDensityMerged_Pipeline2 for argument description
     */

    int base_idx, base_idx_j, neigh_idx;
    double kdistO, distPo, sum;

    __m128i idx_j_128, neigh_idx_128;
    __m256d kdistO_265, distPo_256, sum_256;

    unsigned int j;
    sum = 0;
    base_idx = pnt_idx * k;
    sum_256 = _mm256_set1_pd(0.0);
    for (j = 0; j + 7 < k; j += 8) {

        int base_idx_j0 =  base_idx + j;
        // POTENTIALLY FASTER, DOES NOT COMPILE
        //idx_j_128 = _mm_set_epi32( base_idx_j0+3, base_idx_j0+2, base_idx_j0+1, base_idx_j0);
        //neigh_idx_128 = _mm_i32gather_epi32( k_neighborhood_index, idx_j_128, sizeof(int));
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0);
        //distPo_256 = _mm256_i32gather_pd( dist_k_neighborhood_index, idx_j_128, sizeof(double) );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        //sum += sum_double_avx( _mm256_max_pd( kdistO_265, distPo_256) );
        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        //idx_j_128 = _mm_set_epi32( base_idx_j0+7, base_idx_j0+6, base_idx_j0+5, base_idx_j0+4);
        //neigh_idx_128 = _mm_i32gather_epi32(k_neighborhood_index, idx_j_128, sizeof(int));
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 4);
        //distPo_256 = _mm256_i32gather_pd( dist_k_neighborhood_index, idx_j_128, sizeof(double) );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 4); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );


        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));
    }

    sum += sum_double_avx_2(sum_256);// sum_double_avx( _mm256_max_pd( kdistO_265, distPo_256) );

    for( ; j < k; j++ ){
        base_idx_j = base_idx + j;

        neigh_idx = k_neighborhood_index[ base_idx_j ];
        kdistO = k_distance_index[neigh_idx];   //  dist_k_neighborhood_index[ neigh_idx*k + k-1]
        distPo = dist_k_neighborhood_index[ base_idx_j ];

        sum += kdistO >= distPo ? kdistO : distPo;

    }

    double result = k / sum;
    lrd_score_table_ptr[pnt_idx] = result;

}   // ComputeLocalReachabilityDensityMerged_Point_AVX8


void ComputeLocalReachabilityDensityMerged_Point_AVX16( int pnt_idx, int k,
                                                       const double* dist_k_neighborhood_index,
                                                       const double* k_distance_index,
                                                       const int* k_neighborhood_index,
                                                       double* lrd_score_table_ptr ){
    /**
     * Compute lrd score for @param pnt_idx
     * See ComputeLocalReachabilityDensityMerged_Pipeline2 for argument description
     */

    int base_idx, base_idx_j, neigh_idx;
    double kdistO, distPo, sum;

    __m128i idx_j_128, neigh_idx_128;
    __m256d kdistO_265, distPo_256, sum_256;

    unsigned int j;
    sum = 0;
    base_idx = pnt_idx * k;
    sum_256 = _mm256_set1_pd(0.0);

    for (j = 0; j + 15 < k; j += 16) {

        int base_idx_j0 =  base_idx + j;
        // POTENTIALLY FASTER, DOES NOT COMPILE
        //idx_j_128 = _mm_set_epi32( base_idx_j0+3, base_idx_j0+2, base_idx_j0+1, base_idx_j0);
        //neigh_idx_128 = _mm_i32gather_epi32( k_neighborhood_index, idx_j_128, sizeof(int));
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 );
        //distPo_256 = _mm256_i32gather_pd( dist_k_neighborhood_index, idx_j_128, sizeof(double) );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        //sum += sum_double_avx( _mm256_max_pd( kdistO_265, distPo_256) );
        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        //idx_j_128 = _mm_set_epi32( base_idx_j0+7, base_idx_j0+6, base_idx_j0+5, base_idx_j0+4);
        //neigh_idx_128 = _mm_i32gather_epi32(k_neighborhood_index, idx_j_128, sizeof(int));
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 4 );
        //distPo_256 = _mm256_i32gather_pd( dist_k_neighborhood_index, idx_j_128, sizeof(double) );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 4);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        //idx_j_128 = _mm_set_epi32( base_idx_j0+11, base_idx_j0+10, base_idx_j0+9, base_idx_j0+8);
        //neigh_idx_128 = _mm_i32gather_epi32( k_neighborhood_index, idx_j_128, sizeof(int));
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 8 );
        //distPo_256 = _mm256_i32gather_pd( dist_k_neighborhood_index, idx_j_128, sizeof(double) );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 8);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        //sum += sum_double_avx( _mm256_max_pd( kdistO_265, distPo_256) );
        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        //idx_j_128 = _mm_set_epi32( base_idx_j0+15, base_idx_j0+14, base_idx_j0+13, base_idx_j0+12);
        //neigh_idx_128 = _mm_i32gather_epi32(k_neighborhood_index, idx_j_128, sizeof(int));
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 12 );
        //distPo_256 = _mm256_i32gather_pd( dist_k_neighborhood_index, idx_j_128, sizeof(double) );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 12);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));
    }

    sum += sum_double_avx_2(sum_256);// sum_double_avx( _mm256_max_pd( kdistO_265, distPo_256) );

    for( ; j < k; j++ ){
        base_idx_j = base_idx + j;

        neigh_idx = k_neighborhood_index[ base_idx_j ];
        kdistO = k_distance_index[neigh_idx];   //  dist_k_neighborhood_index[ neigh_idx*k + k-1]
        distPo = dist_k_neighborhood_index[ base_idx_j ];

        sum += kdistO >= distPo ? kdistO : distPo;

    }

    double result = k / sum;
    lrd_score_table_ptr[pnt_idx] = result;

}   // ComputeLocalReachabilityDensityMerged_Point_AVX16


void ComputeLocalReachabilityDensityMerged_Point_AVX32_FASTEST( int pnt_idx, int k,
                                                        const double* dist_k_neighborhood_index,
                                                        const double* k_distance_index,
                                                        const int* k_neighborhood_index,
                                                        double* lrd_score_table_ptr ){
    /**
     * Compute lrd score for @param pnt_idx
     * See ComputeLocalReachabilityDensityMerged_Pipeline2 for argument description
     */

    int base_idx, base_idx_j, neigh_idx;
    double kdistO, distPo, sum;

    __m128i idx_j_128, neigh_idx_128;
    __m256d kdistO_265, distPo_256, sum_256;

    unsigned int j;
    sum = 0;
    base_idx = pnt_idx * k;
    sum_256 = _mm256_set1_pd(0.0);

    for (j = 0; j + 31 < k; j += 32) {

        int base_idx_j0 =  base_idx + j;
        // POTENTIALLY FASTER, DOES NOT COMPILE
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 4 );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 4);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 8 );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 8);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 12 );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 12);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 16 );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 16); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 20 );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 20);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 24 );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 24);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 28 );
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 28);
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));
    }

    for ( ; j + 7 < k; j += 8) {

        int base_idx_j0 =  base_idx + j;
        // POTENTIALLY FASTER, DOES NOT COMPILE
        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0);
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));

        neigh_idx_128 = _mm_loadu_si128(k_neighborhood_index + base_idx_j0 + 4);
        distPo_256 = _mm256_loadu_pd( dist_k_neighborhood_index + base_idx_j0 + 4); // , sizeof(double)
        kdistO_265 = _mm256_i32gather_pd( k_distance_index, neigh_idx_128, sizeof(double) );

        sum_256 = _mm256_add_pd(sum_256, _mm256_max_pd( kdistO_265, distPo_256));
    }

    sum += sum_double_avx_2(sum_256);// sum_double_avx( _mm256_max_pd( kdistO_265, distPo_256) );

    for( ; j < k; j++ ){
        base_idx_j = base_idx + j;

        neigh_idx = k_neighborhood_index[ base_idx_j ];
        kdistO = k_distance_index[neigh_idx];   //  dist_k_neighborhood_index[ neigh_idx*k + k-1]
        distPo = dist_k_neighborhood_index[ base_idx_j ];

        sum += kdistO >= distPo ? kdistO : distPo;

    }

    double result = k / sum;
    lrd_score_table_ptr[pnt_idx] = result;

}   // ComputeLocalReachabilityDensityMerged_Point_AVX32


void verify_function( int num_pts, int pnt_idx, int k, my_fun fun_to_verify ){

    // INITIALIZE INPUT
    double* dist_k_neighborhood_index = XmallocMatrixDoubleRandom( num_pts, k );
    double* k_distance_index = XmallocVectorDoubleRandom( num_pts );
    int* k_neighborhood_index = XmallocMatrixIntRandom( num_pts, k, num_pts-1 );

    double* lrd_score_table_ptr_true = XmallocVectorDouble( num_pts );
    double* lrd_score_table_ptr = XmallocVectorDouble( num_pts );

    ComputeLocalReachabilityDensityMerged_Point(pnt_idx, k, dist_k_neighborhood_index, k_distance_index,
                                                k_neighborhood_index, lrd_score_table_ptr_true);

    fun_to_verify(pnt_idx, k, dist_k_neighborhood_index, k_distance_index, k_neighborhood_index,
                  lrd_score_table_ptr);

    double tol = 0.0001;
    int ver_lof = test_double_arrays( num_pts, tol, lrd_score_table_ptr_true, lrd_score_table_ptr);

    if( ver_lof == 0 ){
        printf("Resulting LOF scores are different\n");
    } else {
        printf( "LOF passed verification\n" );
    }

    if(fabs( lrd_score_table_ptr[pnt_idx] - lrd_score_table_ptr_true[pnt_idx] ) < 0.001){
        printf("PASSED VERIFICATION\n");
    } else {
        printf("DID NOT PASS VERIFICATION: should %lf is %lf", lrd_score_table_ptr_true[pnt_idx], lrd_score_table_ptr[pnt_idx]);
    }
}

void lrdm_point_perf_driver(int num_pts, int k, int num_reps){


    //fast_test_improvement_lrdm_avx(k, num_pts, num_reps);
    int pnt_idx = 100;
    printf("Original function\n");
    performance_measure_fast_for_lrdm_point_pipeline2( pnt_idx, num_pts, k, num_reps,
                                                       ComputeLocalReachabilityDensityMerged_Point);

    printf("ComputeLocalReachabilityDensityMerged_Point_AVX4\n");
    verify_function( num_pts, pnt_idx, k, ComputeLocalReachabilityDensityMerged_Point_AVX4);
    performance_measure_fast_for_lrdm_point_pipeline2( pnt_idx, num_pts, k, num_reps,
                                                       ComputeLocalReachabilityDensityMerged_Point_AVX4);

    printf("ComputeLocalReachabilityDensityMerged_Point_AVX4_V2\n");
    verify_function( num_pts, pnt_idx, k, ComputeLocalReachabilityDensityMerged_Point_AVX4);
    performance_measure_fast_for_lrdm_point_pipeline2( pnt_idx, num_pts, k, num_reps,
                                                       ComputeLocalReachabilityDensityMerged_Point_AVX4_V2);

    printf("ComputeLocalReachabilityDensityMerged_Point_AVX8\n");
    verify_function( num_pts, pnt_idx, k, ComputeLocalReachabilityDensityMerged_Point_AVX8);
    performance_measure_fast_for_lrdm_point_pipeline2( pnt_idx, num_pts, k, num_reps,
                                                       ComputeLocalReachabilityDensityMerged_Point_AVX8);

    printf("ComputeLocalReachabilityDensityMerged_Point_AVX16\n");
    verify_function( num_pts, pnt_idx, k, ComputeLocalReachabilityDensityMerged_Point_AVX16);
    performance_measure_fast_for_lrdm_point_pipeline2( pnt_idx, num_pts, k, num_reps,
                                                       ComputeLocalReachabilityDensityMerged_Point_AVX16);

    printf("ComputeLocalReachabilityDensityMerged_Point_AVX32\n");
    verify_function( num_pts, pnt_idx, k, ComputeLocalReachabilityDensityMerged_Point_AVX16);
    performance_measure_fast_for_lrdm_point_pipeline2( pnt_idx, num_pts, k, num_reps,
                                                       ComputeLocalReachabilityDensityMerged_Point_AVX32_FASTEST);
}