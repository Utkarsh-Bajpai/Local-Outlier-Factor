//
//int k_ref = 100, num_pts_ref = 2000;
//lof_driver( k_ref, num_pts_ref);
//

#include <stdio.h>
#include "stdlib.h"

#include "../../include/utils.h"
#include "../../include/tests.h"
#include "../../include/tsc_x86.h"
#include "../../include/lof_baseline.h"
#include "../../include/performance_measurement.h"

#include "../include/Algorithm.h"
#include "../include/ComputeLocalOutlierFactor.h"

#define CYCLES_REQUIRED 1e8
#define NUM_FUNCTIONS 7


void ComputeLocalOutlierFactor_1(int k, int num_pts, const double* lrd_score_table_ptr,
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
        double denom = lrd_score_table_ptr[i] * k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }
}   // faster_function_1


void ComputeLocalOutlierFactor_2(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr) {
    /**
     * ComputeLocalOutlierFactor_1 + inner loop unrolling by 4
     */

    unsigned int i, j;

    for (i = 0; i < num_pts; i++) {
        double lrd_neighs_sum = 0;

        for (j = 0; j + 4 < k; j += 4) {

            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 1]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 2]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 3]];

        }

        // collect remaining stuff
        for (; j < k; ++j) {
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j]];
        }

        double denom = lrd_score_table_ptr[i] * k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }
} // faster_function_2

void ComputeLocalOutlierFactor_3(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr) {
    /**
     * ComputeLocalOutlierFactor_2 + outer loop unrolling by 4
     */

    double lrd_neighs_sum, lrd_neighs_sum_0, lrd_neighs_sum_1, lrd_neighs_sum_2, lrd_neighs_sum_3;
    double denom, denom_0, denom_1, denom_2, denom_3;

    unsigned int i, j;
    i = 0;
    for (; i + 4 < num_pts; i += 4) {

        lrd_neighs_sum_0 = 0;
        lrd_neighs_sum_1 = 0;
        lrd_neighs_sum_2 = 0;
        lrd_neighs_sum_3 = 0;

        j = 0;
        for (; j + 4 < k; j += 4) {

            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 1]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 2]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 3]];

            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 1) * k + j]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 1) * k + j + 1]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 1) * k + j + 2]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 1) * k + j + 3]];

            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 2) * k + j]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 2) * k + j + 1]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 2) * k + j + 2]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 2) * k + j + 3]];

            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 3) * k + j]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 3) * k + j + 1]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 3) * k + j + 2]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 3) * k + j + 3]];

        }

        // collect remaining stuff
        for (; j < k; ++j) {

            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 1) * k + j]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 2) * k + j]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[(i + 3) * k + j]];

        }

        denom_0 = lrd_score_table_ptr[i] * k;
        denom_1 = lrd_score_table_ptr[i + 1] * k;
        denom_2 = lrd_score_table_ptr[i + 2] * k;
        denom_3 = lrd_score_table_ptr[i + 3] * k;

        lof_score_table_ptr[i] = lrd_neighs_sum_0 / denom_0;
        lof_score_table_ptr[i + 1] = lrd_neighs_sum_1 / denom_1;
        lof_score_table_ptr[i + 2] = lrd_neighs_sum_2 / denom_2;
        lof_score_table_ptr[i + 3] = lrd_neighs_sum_3 / denom_3;

    }

    // collect remaining i: ---------------------------------------------

    for (; i < num_pts; i++) {

        lrd_neighs_sum = 0;

        j = 0;
        for (; j + 4 < k; j += 4) {

            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 1]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 2]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j + 3]];
        }

        // collect remaining stuff
        for (; j < k; ++j) {
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[i * k + j]];
        }

        denom = lrd_score_table_ptr[i] * k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }

} // faster_function_3


void ComputeLocalOutlierFactor_4(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr) {
    /**
     * ComputeLocalOutlierFactor_3 + better index calculation
     */

    double lrd_neighs_sum, lrd_neighs_sum_0, lrd_neighs_sum_1, lrd_neighs_sum_2, lrd_neighs_sum_3;
    double denom, denom_0, denom_1, denom_2, denom_3;

    int idx_base, idx_base_0, idx_base_1, idx_base_2, idx_base_3;
    int idx, idx0, idx1, idx2, idx3;

    unsigned int i, j;
    i = 0;
    for (; i + 4 < num_pts; i += 4) {

        lrd_neighs_sum_0 = 0;
        lrd_neighs_sum_1 = 0;
        lrd_neighs_sum_2 = 0;
        lrd_neighs_sum_3 = 0;

        idx_base_0 = i * k;
        idx_base_1 = (i + 1) * k;
        idx_base_2 = (i + 2) * k;
        idx_base_3 = (i + 3) * k;

        j = 0;
        for (; j + 4 < k; j += 4) {

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;

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

        }

        // collect remaining stuff
        for (; j < k; ++j) {

            idx0 = idx_base_0 + j;
            idx1 = idx_base_1 + j;
            idx2 = idx_base_2 + j;
            idx3 = idx_base_3 + j;

            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3]];

        }

        denom_0 = lrd_score_table_ptr[i] * k;
        denom_1 = lrd_score_table_ptr[i + 1] * k;
        denom_2 = lrd_score_table_ptr[i + 2] * k;
        denom_3 = lrd_score_table_ptr[i + 3] * k;

        lof_score_table_ptr[i] = lrd_neighs_sum_0 / denom_0;
        lof_score_table_ptr[i + 1] = lrd_neighs_sum_1 / denom_1;
        lof_score_table_ptr[i + 2] = lrd_neighs_sum_2 / denom_2;
        lof_score_table_ptr[i + 3] = lrd_neighs_sum_3 / denom_3;

    }

    // collect remaining i: ---------------------------------------------

    for (; i < num_pts; i++) {

        lrd_neighs_sum = 0;
        idx_base = i * k;

        j = 0;
        for (; j + 4 < k; j += 4) {

            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 1]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 2]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 3]];
        }

        // collect remaining stuff
        for (; j < k; ++j) {
            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx]];
        }

        denom = lrd_score_table_ptr[i] * k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }

} // faster_function_3_5


double ComputeLocalOutlierFactorUnroll_fastest(int k, int num_pts, const double* lrd_score_table_ptr,
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
    i = 0;
    j = 0;

    for (; i + 8 < num_pts; i += 8) {

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
        for (; j + 4 < k; j += 4) {

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

            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7]];

        }

        denom_0 = lrd_score_table_ptr[i] * k;
        denom_1 = lrd_score_table_ptr[i + 1] * k;
        denom_2 = lrd_score_table_ptr[i + 2] * k;
        denom_3 = lrd_score_table_ptr[i + 3] * k;
        denom_4 = lrd_score_table_ptr[i + 4] * k;
        denom_5 = lrd_score_table_ptr[i + 5] * k;
        denom_6 = lrd_score_table_ptr[i + 6] * k;
        denom_7 = lrd_score_table_ptr[i + 7] * k;

        lof_score_table_ptr[i] = lrd_neighs_sum_0 / denom_0;
        lof_score_table_ptr[i + 1] = lrd_neighs_sum_1 / denom_1;
        lof_score_table_ptr[i + 2] = lrd_neighs_sum_2 / denom_2;
        lof_score_table_ptr[i + 3] = lrd_neighs_sum_3 / denom_3;
        lof_score_table_ptr[i + 4] = lrd_neighs_sum_4 / denom_4;
        lof_score_table_ptr[i + 5] = lrd_neighs_sum_5 / denom_5;
        lof_score_table_ptr[i + 6] = lrd_neighs_sum_6 / denom_6;
        lof_score_table_ptr[i + 7] = lrd_neighs_sum_7 / denom_7;

    }

    // collect remaining i: ---------------------------------------------

    for (; i < num_pts; i++) {

        lrd_neighs_sum = 0;
        idx_base = i * k;
        j = 0;
        for (; j + 4 < k; j += 4) {

            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 1]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 2]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 3]];

        }

        // collect remaining stuff
        for (; j < k; ++j) {

            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx]];

        }

        denom = lrd_score_table_ptr[i] * k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }

    return num_pts * (2*k + 1);
}


void ComputeLocalOutlierFactor_6(int k, int num_pts, const double* lrd_score_table_ptr,
                                 const int* neighborhood_index_table_ptr,
                                 double* lof_score_table_ptr) {
    /**
     * ComputeLocalOutlierFactor_5 + inner loop unrolling by 8
     */

    double lrd_neighs_sum, lrd_neighs_sum_0, lrd_neighs_sum_1, lrd_neighs_sum_2, lrd_neighs_sum_3;
    double lrd_neighs_sum_4, lrd_neighs_sum_5, lrd_neighs_sum_6, lrd_neighs_sum_7;
    double denom, denom_0, denom_1, denom_2, denom_3, denom_4, denom_5, denom_6, denom_7;

    int idx_base, idx_base_0, idx_base_1, idx_base_2, idx_base_3, idx_base_4, idx_base_5, idx_base_6, idx_base_7;
    int idx, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7;

    unsigned int i, j;
    i = 0;
    j = 0;

    for (; i + 8 < num_pts; i += 8) {

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
        for (; j + 4 < k; j += 8) {

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
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0 + 4]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0 + 5]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0 + 6]];
            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0 + 7]];

            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 1]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 2]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 3]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 4]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 5]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 6]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1 + 7]];

            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 1]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 2]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 3]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 4]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 5]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 6]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2 + 7]];

            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 1]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 2]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 3]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 4]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 5]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 6]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3 + 7]];

            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 1]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 2]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 3]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 4]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 5]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 6]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4 + 7]];

            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 1]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 2]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 3]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 4]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 5]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 6]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5 + 7]];

            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 1]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 2]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 3]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 4]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 5]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 6]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6 + 7]];

            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 1]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 2]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 3]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 4]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 5]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 6]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7 + 7]];

        }

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

            lrd_neighs_sum_0 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx0]];
            lrd_neighs_sum_1 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx1]];
            lrd_neighs_sum_2 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx2]];
            lrd_neighs_sum_3 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx3]];
            lrd_neighs_sum_4 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx4]];
            lrd_neighs_sum_5 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx5]];
            lrd_neighs_sum_6 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx6]];
            lrd_neighs_sum_7 += lrd_score_table_ptr[neighborhood_index_table_ptr[idx7]];

        }

        denom_0 = lrd_score_table_ptr[i] * k;
        denom_1 = lrd_score_table_ptr[i + 1] * k;
        denom_2 = lrd_score_table_ptr[i + 2] * k;
        denom_3 = lrd_score_table_ptr[i + 3] * k;
        denom_4 = lrd_score_table_ptr[i + 4] * k;
        denom_5 = lrd_score_table_ptr[i + 5] * k;
        denom_6 = lrd_score_table_ptr[i + 6] * k;
        denom_7 = lrd_score_table_ptr[i + 7] * k;

        lof_score_table_ptr[i] = lrd_neighs_sum_0 / denom_0;
        lof_score_table_ptr[i + 1] = lrd_neighs_sum_1 / denom_1;
        lof_score_table_ptr[i + 2] = lrd_neighs_sum_2 / denom_2;
        lof_score_table_ptr[i + 3] = lrd_neighs_sum_3 / denom_3;
        lof_score_table_ptr[i + 4] = lrd_neighs_sum_4 / denom_4;
        lof_score_table_ptr[i + 5] = lrd_neighs_sum_5 / denom_5;
        lof_score_table_ptr[i + 6] = lrd_neighs_sum_6 / denom_6;
        lof_score_table_ptr[i + 7] = lrd_neighs_sum_7 / denom_7;

    }

    // collect remaining i: ---------------------------------------------

    for (; i < num_pts; i++) {

        lrd_neighs_sum = 0;
        idx_base = i * k;

        j = 0;
        for (; j + 8 < k; j += 8) {

            idx = idx_base + j;

            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 1]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 2]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 3]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 4]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 5]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 6]];
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx + 7]];
        }

        // collect remaining stuff
        for (; j < k; ++j) {
            idx = idx_base + j;
            lrd_neighs_sum += lrd_score_table_ptr[neighborhood_index_table_ptr[idx]];
        }

        denom = lrd_score_table_ptr[i] * k;
        lof_score_table_ptr[i] = lrd_neighs_sum / denom;
    }

}

// --------------------------------------------------------------------------- Drivers


int lof_driver_unrolled(int k_ref, int num_pts_ref) {
    /**
     * functions for producing the measurements
     * @param k_ref: value of k fixed, when performance for different num_pts is compared
     * @param num_pts_ref: value of num_pts fixed, when performance for different k is compared
    */

    my_lof_fnc* fun_array_lof = (my_lof_fnc*) calloc(NUM_FUNCTIONS, sizeof(my_lof_fnc));

    fun_array_lof[0] = &ComputeLocalOutlierFactor;
    fun_array_lof[1] = &ComputeLocalOutlierFactor_1;
    fun_array_lof[2] = &ComputeLocalOutlierFactor_2;
    fun_array_lof[3] = &ComputeLocalOutlierFactor_3;
    fun_array_lof[4] = &ComputeLocalOutlierFactor_4;
    fun_array_lof[5] = &ComputeLocalOutlierFactorUnroll_fastest;
    fun_array_lof[6] = &ComputeLocalOutlierFactor_6;

    char* fun_names[NUM_FUNCTIONS] = {"baseline", "strength reduction", "inner unroll 4", "inner 4 outer 4",
                                      "inner 4 outer 4 + index", "inner 4 outer 8 + index", "inner 8 outer 8 + index"};

    // Performance for different k
    performance_plot_lof_to_file_k(num_pts_ref, "../unrolled/performance_improvement/measurements/lof_results_k.txt",
                                   fun_names[0], "w", fun_array_lof[0], fun_array_lof[0]);
    for (int i = 1; i < NUM_FUNCTIONS; ++i) {
        performance_plot_lof_to_file_k(num_pts_ref,
                                       "../unrolled/performance_improvement/measurements/lof_results_k.txt",
                                       fun_names[i], "a", fun_array_lof[i], fun_array_lof[0]);
    }
    // Performance for different num_pts
    performance_plot_lof_to_file_num_pts(k_ref,
                                         "../unrolled/performance_improvement/measurements/lof_results_num_pts.txt",
                                         fun_names[0], "w", fun_array_lof[0], fun_array_lof[0]);
    for (int i = 1; i < NUM_FUNCTIONS; ++i) {
        performance_plot_lof_to_file_num_pts(k_ref,
                                             "../unrolled/performance_improvement/measurements/lof_results_num_pts.txt",
                                             fun_names[i], "a", fun_array_lof[i], fun_array_lof[0]);
    }

    return 1;
}



