/**
 * TODO: 1. inner / outer unroll by 8
 *       2. improvements for version with a hash table
 * CURRENTLY there are 5 functions
 *
 * TODO:
 *       _mm256_min_epi32 8 intgers -> to load need 4 longs
 *       Try to _mm256_min_epi32 - > use results for loading ?
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <xmmintrin.h>

#include "../../include/lof_baseline.h"
#include "../../include/tsc_x86.h"
#include "../../include/performance_measurement.h"

#include "../include/ComputeLocalReachabilityDensityMerged.h"
#include "../include/AVX_utils.h"

#define MIN(a,b) a ^ ((b ^ a) & -(b < a))
#define NUM_FUNCTIONS_AVX_RDM 5

// ------------------------------------------------------------------------------- <AVX improvements>

void ComputeLocalReachabilityDensityMerged_OUTER4_INNER1_AVX_128(int k, int num_pts, const double* distances_indexed_ptr,
                                                                 const double* k_distances_indexed_ptr,
                                                                 const int* neighborhood_index_table_ptr,
                                                                 double* lrd_score_table_ptr) {
    /**
     * Outer Loop Unroll by 4 + No Inner Loop Unroll
     * Load indexes from neighborhood_index_table_ptr as _m128i numbers
     *
     */

    unsigned int i, j;

    __m256d sum_avx, lrd_vec, k_avx, kdistO_avx, distPo_avx;
    __m128i neigh_idx_128, vindex_dist_128_1, vindex_dist_128_2, vindex_dist_128, i_avx_128, incr_128, n_128_avx, n_i_avx_128;

    int idx_base, idx_base_0, idx_base_1, idx_base_2, idx_base_3;
    int idx_base_k, idx_base_k_0, idx_base_k_1, idx_base_k_2, idx_base_k_3;
    double sum;

    k_avx = _mm256_set1_pd( k );
    n_128_avx = _mm_set1_epi32( num_pts );
    i_avx_128 = _mm_set_epi32( 3, 2, 1, 0 );
    incr_128 = _mm_set1_epi32( 4 );

    for ( i = 0; i + 3 < num_pts; i += 4 ) {

        idx_base_k_0 = i*k;
        idx_base_k_1 = (i + 1)*k;
        idx_base_k_2 = (i + 2)*k;
        idx_base_k_3 = (i + 3)*k;

        sum_avx = _mm256_setzero_pd();
        n_i_avx_128 = _mm_mullo_epi32(i_avx_128, n_128_avx);

        for ( j = 0; j < k; j ++ ) {

            neigh_idx_128 = _mm_set_epi32( neighborhood_index_table_ptr[idx_base_k_3 + j], neighborhood_index_table_ptr[idx_base_k_2 + j], neighborhood_index_table_ptr[idx_base_k_1 + j], neighborhood_index_table_ptr[idx_base_k_0 + j]);

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128); // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            // TRY _mm_blend_epi32 instead !!!
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );

            kdistO_avx = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128, sizeof(double) );
            distPo_avx = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx, distPo_avx) );

        }

        lrd_vec =  _mm256_div_pd ( k_avx, sum_avx );
        _mm256_store_pd(lrd_score_table_ptr+i, lrd_vec);
        // increment
        i_avx_128 = _mm_add_epi32(i_avx_128, incr_128);

    }   // i + 4;

    // COLLECT REMAINING ELEMENTS I:
    for (  ; i < num_pts; i++ ) {

        sum = 0;
        idx_base = num_pts * i;
        idx_base_k = i * k;

        for (int j = 0; j < k; j++) {

            int neigh_idx = neighborhood_index_table_ptr[idx_base_k + j];
            int idx_dist;

            if (i < neigh_idx) {
                idx_dist = idx_base + neigh_idx;
            } else {
                idx_dist = num_pts * neigh_idx + i;
            }

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];

            sum += kdistO >= distPo ? kdistO : distPo;
        }
        lrd_score_table_ptr[i] = k / sum;
    }
}

void ComputeLocalReachabilityDensityMerged_OUTER1_INNER4_AVX_128(int k, int num_pts, const double* distances_indexed_ptr,
                                                                 const double* k_distances_indexed_ptr,
                                                                 const int* neighborhood_index_table_ptr,
                                                                 double* lrd_score_table_ptr){

    unsigned int i, j;
    double sum;
    int idx;

    __m256d sum_avx, lrd_vec, k_avx, kdistO_avx, distPo_avx;
    __m128i neigh_idx_128, vindex_dist_128_1, vindex_dist_128_2, vindex_dist_128, i_avx_128, incr_128, n_128_avx, n_i_avx_128;

    i_avx_128 = _mm_set1_epi32( 0 );
    incr_128 = _mm_set1_epi32( 1 );
    n_128_avx = _mm_set1_epi32( num_pts );

    for ( i = 0; i < num_pts; i++ ) {

        sum = 0;
        n_i_avx_128 = _mm_set1_epi32( i*num_pts );
        // n_i_avx_128 = _mm_mullo_epi32( i_avx_128, n_128_avx );
        idx = i*k;

        for ( j = 0; j + 3 < k; j += 4 ) {

            neigh_idx_128 = _mm_loadu_si128( neighborhood_index_table_ptr + idx + j );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128); // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            // TRY _mm_blend_epi32 instead !!!
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );

            kdistO_avx = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128, sizeof(double) );
            distPo_avx = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            sum += sum_double_avx(_mm256_max_pd(kdistO_avx, distPo_avx));

        }
        // COLLECT REMAINING J
        for( ; j < k; ++j ){

            int neigh_idx = neighborhood_index_table_ptr[i * k + j];
            int idx_dist;

            if (i < neigh_idx) {
                idx_dist = num_pts * i + neigh_idx;
            } else {
                idx_dist = num_pts * neigh_idx + i;
            }

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];

            sum += kdistO >= distPo ? kdistO : distPo;

        }

        lrd_score_table_ptr[i] = k / sum;
        i_avx_128 = _mm_add_epi32(i_avx_128, incr_128);

    }
}

double ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128(int k, int num_pts, const double* distances_indexed_ptr,
                                                                 const double* k_distances_indexed_ptr,
                                                                 const int* neighborhood_index_table_ptr,
                                                                 double* lrd_score_table_ptr) {
    /**
     *
     * Outer Loop Unroll by 4
     * Load indexes from neighborhood_index_table_ptr as _128i numbers
     *
     */

    unsigned int i, j;

    __m256d sum_avx, lrd_vec, k_avx, kdistO_avx, distPo_avx;
    __m256d kdistO_avx_0, kdistO_avx_1, kdistO_avx_2, kdistO_avx_3;
    __m256d distPo_avx_0, distPo_avx_1, distPo_avx_2, distPo_avx_3;

    __m128i neigh_idx_128, vindex_dist_128_1, vindex_dist_128_2, vindex_dist_128, incr_128, n_128_avx, n_i_avx_128;
    __m128i neigh_idx_128_0, neigh_idx_128_1, neigh_idx_128_2, neigh_idx_128_3;
    __m128i i_avx_128, i_avx_128_0, i_avx_128_1, i_avx_128_2, i_avx_128_3;

    int idx_base, idx_base_0, idx_base_1, idx_base_2, idx_base_3;
    int idx_base_k, idx_base_k_0, idx_base_k_1, idx_base_k_2, idx_base_k_3;
    double sum;

    k_avx = _mm256_set1_pd( k );
    n_128_avx = _mm_set1_epi32( num_pts );
    i_avx_128 = _mm_set_epi32( 3, 2, 1, 0 );
    incr_128 = _mm_set1_epi32( 4 );

    for ( i = 0; i + 3 < num_pts; i += 4 ) {

        idx_base_k_0 = i*k;
        idx_base_k_1 = (i + 1)*k;
        idx_base_k_2 = (i + 2)*k;
        idx_base_k_3 = (i + 3)*k;

        sum_avx = _mm256_setzero_pd();
        n_i_avx_128 = _mm_mullo_epi32(i_avx_128, n_128_avx);

        for ( j = 0; j + 3 < k; j += 4 ) {

            neigh_idx_128_0 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_0 + j );
            neigh_idx_128_1 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_1 + j );
            neigh_idx_128_2 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_2 + j );
            neigh_idx_128_3 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_3 + j );

            _MM_TRANSPOSE4_PS(neigh_idx_128_0, neigh_idx_128_1, neigh_idx_128_2, neigh_idx_128_3);

            // Gather data:
            kdistO_avx_0 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_0, sizeof(double) );
            kdistO_avx_1 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_1, sizeof(double) );
            kdistO_avx_2 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_2, sizeof(double) );
            kdistO_avx_3 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_3, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_0, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_0 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_0 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_1, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_1 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_1 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_2, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_2 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_2 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_3, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_3 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_3 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_0, distPo_avx_0));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_1, distPo_avx_1));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_2, distPo_avx_2));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_3, distPo_avx_3));

        }

        // COLLECT REMAINING ELEMENTS J
        for ( ; j < k; j ++ ) {

            neigh_idx_128 = _mm_set_epi32( neighborhood_index_table_ptr[idx_base_k_3 + j], neighborhood_index_table_ptr[idx_base_k_2 + j], neighborhood_index_table_ptr[idx_base_k_1 + j], neighborhood_index_table_ptr[idx_base_k_0 + j]);

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128); // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            // TRY _mm_blend_epi32 instead !!!
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );

            kdistO_avx = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128, sizeof(double) );
            distPo_avx = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx, distPo_avx) );

        }

        lrd_vec =  _mm256_div_pd ( k_avx, sum_avx );
        _mm256_store_pd( lrd_score_table_ptr+i, lrd_vec );
        // increment
        i_avx_128 = _mm_add_epi32( i_avx_128, incr_128 );

    }   // i + 4;

    // COLLECT REMAINING ELEMENTS I:
    for (  ; i < num_pts; i++ ) {

        sum = 0;
        idx_base = num_pts * i;
        idx_base_k = i * k;

        for (int j = 0; j < k; j++) {

            int neigh_idx = neighborhood_index_table_ptr[idx_base_k + j];
            int idx_dist;

            if (i < neigh_idx) {
                idx_dist = idx_base + neigh_idx;
            } else {
                idx_dist = num_pts * neigh_idx + i;
            }

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];

            sum += kdistO >= distPo ? kdistO : distPo;
        }
        lrd_score_table_ptr[i] = k / sum;
    }

    return num_pts + 3 * k * num_pts;
}

void ComputeLocalReachabilityDensityMerged_OUTER4_INNER8_AVX_128(int k, int num_pts, const double* distances_indexed_ptr,
                                                                 const double* k_distances_indexed_ptr,
                                                                 const int* neighborhood_index_table_ptr,
                                                                 double* lrd_score_table_ptr) {
    /**
     *
     * Outer Loop Unroll by 4 Inner loop unroll by 8
     * Load indexes from neighborhood_index_table_ptr as _128i numbers
     *
     */

    unsigned int i, j;

    __m256d sum_avx, lrd_vec, k_avx, kdistO_avx, distPo_avx;
    __m256d kdistO_avx_0, kdistO_avx_1, kdistO_avx_2, kdistO_avx_3;
    __m256d distPo_avx_0, distPo_avx_1, distPo_avx_2, distPo_avx_3;

    __m128i neigh_idx_128, vindex_dist_128_1, vindex_dist_128_2, vindex_dist_128, incr_128, n_128_avx, n_i_avx_128;
    __m128i neigh_idx_128_0, neigh_idx_128_1, neigh_idx_128_2, neigh_idx_128_3;
    __m128i i_avx_128, i_avx_128_0, i_avx_128_1, i_avx_128_2, i_avx_128_3;

    int idx_base, idx_base_0, idx_base_1, idx_base_2, idx_base_3;
    int idx_base_k, idx_base_k_0, idx_base_k_1, idx_base_k_2, idx_base_k_3;
    double sum;

    k_avx = _mm256_set1_pd( k );
    n_128_avx = _mm_set1_epi32( num_pts );
    i_avx_128 = _mm_set_epi32( 3, 2, 1, 0 );
    incr_128 = _mm_set1_epi32( 4 );

    for ( i = 0; i + 3 < num_pts; i += 4 ) {

        idx_base_k_0 = i*k;
        idx_base_k_1 = (i + 1)*k;
        idx_base_k_2 = (i + 2)*k;
        idx_base_k_3 = (i + 3)*k;

        sum_avx = _mm256_setzero_pd();
        n_i_avx_128 = _mm_mullo_epi32(i_avx_128, n_128_avx);

        for ( j = 0; j + 8 < k; j += 8 ) {

            neigh_idx_128_0 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_0 + j );
            neigh_idx_128_1 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_1 + j );
            neigh_idx_128_2 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_2 + j );
            neigh_idx_128_3 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_3 + j );

            _MM_TRANSPOSE4_PS(neigh_idx_128_0, neigh_idx_128_1, neigh_idx_128_2, neigh_idx_128_3);

            // Gather data:
            kdistO_avx_0 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_0, sizeof(double) );
            kdistO_avx_1 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_1, sizeof(double) );
            kdistO_avx_2 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_2, sizeof(double) );
            kdistO_avx_3 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_3, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_0, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_0 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_0 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_1, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_1 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_1 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_2, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_2 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_2 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_3, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_3 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_3 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_0, distPo_avx_0));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_1, distPo_avx_1));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_2, distPo_avx_2));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_3, distPo_avx_3));

            // ------------------- second set of indexes
            neigh_idx_128_0 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_0 + j + 4);
            neigh_idx_128_1 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_1 + j + 4 );
            neigh_idx_128_2 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_2 + j + 4 );
            neigh_idx_128_3 = _mm_loadu_si128( neighborhood_index_table_ptr + idx_base_k_3 + j + 4 );

            _MM_TRANSPOSE4_PS(neigh_idx_128_0, neigh_idx_128_1, neigh_idx_128_2, neigh_idx_128_3);

            // Gather data:
            kdistO_avx_0 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_0, sizeof(double) );
            kdistO_avx_1 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_1, sizeof(double) );
            kdistO_avx_2 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_2, sizeof(double) );
            kdistO_avx_3 = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128_3, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_0, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_0 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_0 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_1, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_1 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_1 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_2, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_2 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_2 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128_3, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128_3 );      // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );
            distPo_avx_3 = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );

            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_0, distPo_avx_0));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_1, distPo_avx_1));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_2, distPo_avx_2));
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx_3, distPo_avx_3));

        }

        // COLLECT REMAINING ELEMENTS J
        for ( ; j < k; j ++ ) {

            neigh_idx_128 = _mm_set_epi32( neighborhood_index_table_ptr[idx_base_k_3 + j], neighborhood_index_table_ptr[idx_base_k_2 + j], neighborhood_index_table_ptr[idx_base_k_1 + j], neighborhood_index_table_ptr[idx_base_k_0 + j]);

            vindex_dist_128_1 = _mm_add_epi32( _mm_mullo_epi32( neigh_idx_128, n_128_avx ),  i_avx_128);
            vindex_dist_128_2 = _mm_add_epi32( n_i_avx_128,  neigh_idx_128); // _mm_mullo_epi32( n_128_avx, i_avx_128 )
            vindex_dist_128 = _mm_min_epi32( vindex_dist_128_1, vindex_dist_128_2 );

            kdistO_avx = _mm256_i32gather_pd( k_distances_indexed_ptr, neigh_idx_128, sizeof(double) );
            distPo_avx = _mm256_i32gather_pd( distances_indexed_ptr, vindex_dist_128, sizeof(double) );
            sum_avx = _mm256_add_pd( sum_avx, _mm256_max_pd(kdistO_avx, distPo_avx) );

        }

        lrd_vec =  _mm256_div_pd ( k_avx, sum_avx );
        _mm256_store_pd( lrd_score_table_ptr+i, lrd_vec );
        // increment
        i_avx_128 = _mm_add_epi32( i_avx_128, incr_128 );

    }   // i + 4;

    // COLLECT REMAINING ELEMENTS I:
    for (  ; i < num_pts; i++ ) {

        sum = 0;
        idx_base = num_pts * i;
        idx_base_k = i * k;

        for (int j = 0; j < k; j++) {

            int neigh_idx = neighborhood_index_table_ptr[idx_base_k + j];
            int idx_dist;

            if (i < neigh_idx) {
                idx_dist = idx_base + neigh_idx;
            } else {
                idx_dist = num_pts * neigh_idx + i;
            }

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];

            sum += kdistO >= distPo ? kdistO : distPo;
        }
        lrd_score_table_ptr[i] = k / sum;
    }
}


// ------------------------------------------------------------------------------- </AVX improvements>

void fast_test_improvement_lrdm_avx( int k , int num_pts, int num_reps ){


    printf("ComputeLocalReachabilityDensityMerged\n");
    performance_measure_fast_for_lrd( k, num_pts, num_reps, ComputeLocalReachabilityDensityMerged, ComputeFlopsReachabilityDensityMerged);

    printf("\nComputeLocalReachabilityDensityMerged_OUTER1_INNER4_AVX_128\n");
    performance_measure_fast_for_lrd( k, num_pts, num_reps, ComputeLocalReachabilityDensityMerged_OUTER1_INNER4_AVX_128, ComputeFlopsReachabilityDensityMerged);

    printf("\nComputeLocalReachabilityDensityMerged_OUTER4_INNER1_AVX_128\n");
    performance_measure_fast_for_lrd( k, num_pts, num_reps, ComputeLocalReachabilityDensityMerged_OUTER4_INNER1_AVX_128, ComputeFlopsReachabilityDensityMerged);

    printf("\nComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128\n");
    performance_measure_fast_for_lrd( k, num_pts, num_reps, ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128, ComputeFlopsReachabilityDensityMerged);

    printf("\nComputeLocalReachabilityDensityMerged_OUTER4_INNER8_AVX_128\n");
    performance_measure_fast_for_lrd( k, num_pts, num_reps, ComputeLocalReachabilityDensityMerged_OUTER4_INNER8_AVX_128, ComputeFlopsReachabilityDensityMerged);

    /*
    performance_measure_hot_for_lrd(k, num_pts, ComputeLocalReachabilityDensityMerged_OUTER4_INNER1_AVX_128, 1, 1, 1);

    performance_measure_hot_for_lrd(k, num_pts, ComputeLocalReachabilityDensityMerged_OUTER1_INNER4_AVX_128, 1, 1, 1);

    performance_measure_hot_for_lrd(k, num_pts, ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128, 1, 1, 1);

    performance_measure_hot_for_lrd(k, num_pts, ComputeLocalReachabilityDensityMerged_OUTER4_INNER8_AVX_128, 1, 1, 1);

     */

}

int lrdm_driver_avx( int k_ref, int num_pts_ref ) {
    /**
     * functions for producing the measurements
     * @param k_ref: value of k fixed, when performance for different num_pts is compared
     * @param num_pts_ref: value of num_pts fixed, when performance for different k is compared
    */

    my_lof_fnc* fun_array_lof = (my_lof_fnc*) calloc(NUM_FUNCTIONS_AVX_RDM, sizeof(my_lof_fnc));
    printf("Start n");
    fun_array_lof[0] = &ComputeLocalOutlierFactor;
    fun_array_lof[1] = &ComputeLocalReachabilityDensityMerged_OUTER1_INNER4_AVX_128;
    fun_array_lof[2] = &ComputeLocalReachabilityDensityMerged_OUTER4_INNER1_AVX_128;
    fun_array_lof[3] = &ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128;
    fun_array_lof[4] = &ComputeLocalReachabilityDensityMerged_OUTER4_INNER8_AVX_128;


    char* fun_names[NUM_FUNCTIONS_AVX_RDM] = { "baseline", "avx OUTER1_INNER4", "avx OUTER4_INNER1", "avx OUTER4_INNER4", "avx OUTER4_INNER8" };

    // Performance for different k

    performance_plot_lrdm_to_file_k( num_pts_ref, "../avx/performance_improvement/measurements/lrdm_results_k.txt",
                                    fun_names[0], "w", fun_array_lof[0]);
    for(int i = 1; i < NUM_FUNCTIONS_AVX_RDM; ++i){
        performance_plot_lrdm_to_file_k( num_pts_ref, "../avx/performance_improvement/measurements/lrdm_results_k.txt",
                                        fun_names[i], "a", fun_array_lof[i]);
    }
    // Performance for different num_pts
    performance_plot_lrdm_to_file_num_pts( k_ref, "../avx/performance_improvement/measurements/lrdm_results_num_pts.txt",
                                         fun_names[0], "w", fun_array_lof[0]);
    for(int i = 1; i < NUM_FUNCTIONS_AVX_RDM; ++i){
        performance_plot_lrdm_to_file_num_pts( k_ref, "../avx/performance_improvement/measurements/lrdm_results_num_pts.txt",
                                             fun_names[i], "a", fun_array_lof[i]);
    }

    return 1;
}

