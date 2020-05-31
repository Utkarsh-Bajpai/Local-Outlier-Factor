//
// @asycheva: had errors using intrinsics for converting, and other operations on __m256i numbers
// (i.e. _mm256_cvtepu64_pd, _mm256_min_epi64, _mm256_cmp_epi64_mask,  _mm256_mullo_epi64 ...)
// In the first function ComputeLocalReachabilityDensity_Outer1 I have the code that should work 
// if these intrinsics are present. 
//

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <immintrin.h>

#include "../../include/lof_baseline.h"
#include "../../include/tsc_x86.h"
#include "../../include/performance_measurement.h"
#include "../../include/tests.h"

#include "../include/ComputeLocalReachabilityDensityMerged.h"

/**
* OBSERVATIONS
*            1) in baseline only elements of distances_indexed_ptr are accessed twice
 *              everything else -> once
 *
* TRADEOFF:
*            a) i-j => strided access for reachability_distances_indexed_ptr,
 *                     contigeous for neighborhood_index_table_ptr
*            b) j-i => strided access for neighborhood_index_table_ptr,
 *                     subsequently accessed elements of reachability_distances_indexed_ptr are closer together
 *
 * POTENTIAL IMPROVEMENTS:
 *           1) change the storage layout for neighborhood_index_table_ptr to avoid the TRADEOFF
 *
 * TODO:
 *      1) write a routine for matrix transpose for verification
*/
// ------------------------------------------------------------------------------- </Memory improvements>

void ComputeLocalReachabilityDensityMerged_Reversed(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {
    /**
     * Reverse the loop order / should be suboptimal compared to the baseline
     * since results in a strided access
     *
     * @note originally taken from
     */
    for (int j = 0; j < k; j++) {

        for (int i = 0; i < num_pts; i++) {

            int neigh_idx = neighborhood_index_table_ptr[i * k + j];

            int idx_dist;

            if (i < neigh_idx) {
                idx_dist = num_pts * i + neigh_idx;
            } else {
                idx_dist = num_pts * neigh_idx + i;
            }

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];

            lrd_score_table_ptr[i] += kdistO >= distPo ? kdistO : distPo;
        }
    }
    for (int i = 0; i < num_pts; i++) {
        lrd_score_table_ptr[i] = k / lrd_score_table_ptr[i];
    }
}


void ComputeLocalReachabilityDensityMerged_Reversed_Layout(int k, int num_pts, const double* distances_indexed_ptr,
                                                    const double* k_distances_indexed_ptr,
                                                    const int* neighborhood_index_table_transposed_ptr,
                                                    double* lrd_score_table_ptr) {
    /**
     * @param: neighborhood_index_table_transposed_ptr:
     *                  [num_pts x k] matrix stored
     *                  k-th neighbor of point i is stored at index k*num_pts + i
     *
     */
    for (int j = 0; j < k; j++) {

        for (int i = 0; i < num_pts; i++) {

            int neigh_idx = neighborhood_index_table_transposed_ptr[ j*num_pts + i ];

            int idx_dist;

            if (i < neigh_idx) {
                idx_dist = num_pts * i + neigh_idx;
            } else {
                idx_dist = num_pts * neigh_idx + i;
            }

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];

            lrd_score_table_ptr[i] += kdistO >= distPo ? kdistO : distPo;
        }
    }

    for (int i = 0; i < num_pts; i++) {
        lrd_score_table_ptr[i] = k / lrd_score_table_ptr[i];
    }
}

// ------------------------------------------------------------------------------- </Memory improvements>

#define NUM_FUNCTIONS_AVX_MEM 3
int lrdm_driver_memory( int k_ref, int num_pts_ref ) {

    my_lrdm_fnc* fun_array_lrd = (my_lrdm_fnc*) calloc(NUM_FUNCTIONS_AVX_MEM, sizeof(my_lrdm_fnc));
    fun_array_lrd[0] = &ComputeLocalReachabilityDensityMerged;
    fun_array_lrd[1] = &ComputeLocalReachabilityDensityMerged_Reversed;
    fun_array_lrd[2] = &ComputeLocalReachabilityDensityMerged_Reversed_Layout;
    //fun_array_lrd[3] = &ComputeLocalReachabilityDensityMerged_Inner2;


    //char* fun_names[NUM_FUNCTIONS_AVX_RDM] = {"baseline", "avx outer 4", "avx outer 4 inner 4", "avx outer 4 inner 8"};
    char* fun_names[NUM_FUNCTIONS_AVX_MEM] = { "baseline", "avx OUTER1_INNER4", "avx OUTER4_INNER1", "avx OUTER4_INNER4", "avx OUTER4_INNER8" };

    // Performance for different k

    performance_plot_lrdm_to_file_k( num_pts_ref, "../memory/performance_improvement/measurements/lrdm_results_k.txt",
                                     fun_names[0], "w", fun_array_lrd[0]);
    for(int i = 1; i < NUM_FUNCTIONS_AVX_MEM; ++i){
        performance_plot_lrdm_to_file_k( num_pts_ref, "../memory/performance_improvement/measurements/lrdm_results_k.txt",
                                         fun_names[i], "a", fun_array_lrd[i]);
    }
    // Performance for different num_pts
    performance_plot_lrdm_to_file_num_pts( k_ref, "../memory/performance_improvement/measurements/lrdm_results_num_pts.txt",
                                           fun_names[0], "w", fun_array_lrd[0]);
    for(int i = 1; i < NUM_FUNCTIONS_AVX_MEM; ++i){
        performance_plot_lrdm_to_file_num_pts( k_ref, "../memory/performance_improvement/measurements/lrdm_results_num_pts.txt",
                                               fun_names[i], "a", fun_array_lrd[i]);
    }



    return 1;
}
