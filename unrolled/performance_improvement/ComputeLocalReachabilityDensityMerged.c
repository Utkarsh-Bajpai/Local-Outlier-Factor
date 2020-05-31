//
// Created by pasca on 08.05.2020.
//

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "../../include/utils.h"
#include "../../include/tsc_x86.h"
#include "../../include/tests.h"
#include "../../include/file_utils.h"
#include "../../include/lof_baseline.h"


typedef void (* my_fun)(int, int, const double*, const double*, const int*, double*);

#define CYCLES_REQUIRED 1e8
int* hashes_linear;
int* hashes_matrix;


void SeparateBaseline(int k, int num_pts, const double* distances_indexed_ptr,
                      const double* k_distances_indexed_ptr,
                      const int* neighborhood_index_table_ptr,
                      double* lrd_score_table_ptr) {

    double* reachability_distances_indexed_ptr_true = XmallocVectorDouble(num_pts * num_pts);


    ComputeReachabilityDistanceAll(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                   reachability_distances_indexed_ptr_true);


    ComputeLocalReachabilityDensity(k, num_pts, reachability_distances_indexed_ptr_true, neighborhood_index_table_ptr,
                                    lrd_score_table_ptr);

    free(reachability_distances_indexed_ptr_true);
}

/**
 *
 * Merged the 2 functionalities
 *
 */
void ComputeLocalReachabilityDensityMerged_1(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {
    // IS IT THE SAME AS BASELINE ?
    for (int i = 0; i < num_pts; i++) {
        double sum = 0;

        for (int j = 0; j < k; j++) {
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
    }

}

/**
 *
 * Outer unroll 4, performs worse. Probably messes up cache
 *
 */
void ComputeLocalReachabilityDensityMerged_2(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {
    int i, j;

    for (i = 0; i + 4 < num_pts; i += 4) {

        double sum_0 = 0;
        double sum_1 = 0;
        double sum_2 = 0;
        double sum_3 = 0;


        int np_base_0 = i * k;
        int np_base_1 = np_base_0 + k;
        int np_base_2 = np_base_1 + k;
        int np_base_3 = np_base_2 + k;


        for (j = 0; j < k; j++) {
            int np_0 = neighborhood_index_table_ptr[np_base_0 + j];
            int np_1 = neighborhood_index_table_ptr[np_base_1 + j];
            int np_2 = neighborhood_index_table_ptr[np_base_2 + j];
            int np_3 = neighborhood_index_table_ptr[np_base_3 + j];

            int idx_dist_0;
            int idx_dist_1;
            int idx_dist_2;
            int idx_dist_3;

            if (i < np_0) {
                idx_dist_0 = num_pts * i + np_0;
            } else {
                idx_dist_0 = num_pts * np_0 + i;
            }
            if (i + 1 < np_1) {
                idx_dist_1 = num_pts * (i + 1) + np_1;
            } else {
                idx_dist_1 = num_pts * np_1 + i + 1;
            }
            if (i + 2 < np_2) {
                idx_dist_2 = num_pts * (i + 2) + np_2;
            } else {
                idx_dist_2 = num_pts * np_2 + (i + 2);
            }
            if (i + 3 < np_3) {
                idx_dist_3 = num_pts * (i + 3) + np_3;
            } else {
                idx_dist_3 = num_pts * np_3 + i + 3;
            }

            double k_dist_0 = k_distances_indexed_ptr[np_0];
            double dist_p_0 = distances_indexed_ptr[idx_dist_0];
            sum_0 += k_dist_0 >= dist_p_0 ? k_dist_0 : dist_p_0;

            double k_dist_1 = k_distances_indexed_ptr[np_1];
            double dist_p_1 = distances_indexed_ptr[idx_dist_1];
            sum_1 += k_dist_1 >= dist_p_1 ? k_dist_1 : dist_p_1;

            double k_dist_2 = k_distances_indexed_ptr[np_2];
            double dist_p_2 = distances_indexed_ptr[idx_dist_2];
            sum_2 += k_dist_2 >= dist_p_2 ? k_dist_2 : dist_p_2;

            double k_dist_3 = k_distances_indexed_ptr[np_3];
            double dist_p_3 = distances_indexed_ptr[idx_dist_3];
            sum_3 += k_dist_3 >= dist_p_3 ? k_dist_3 : dist_p_3;


        }
        lrd_score_table_ptr[i] = k / sum_0;
        lrd_score_table_ptr[i + 1] = k / sum_1;
        lrd_score_table_ptr[i + 2] = k / sum_2;
        lrd_score_table_ptr[i + 3] = k / sum_3;
    }

    for (i = 0; i < num_pts; i++) {
        double sum = 0;

        for (j = 0; j < k; j++) {
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
    }

}

/**
 *
 * Inner unroll 4.
 *
 */
void ComputeLocalReachabilityDensityMerged_3(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {

    int i, j;
    for (i = 0; i < num_pts; i++) {
        double sum = 0;

        int neigh_base = i * k;
        int dist_base = num_pts * i;

        for (j = 0; j + 3 < k; j += 4) {
            int neigh_idx_0 = neighborhood_index_table_ptr[neigh_base + j];
            int neigh_idx_1 = neighborhood_index_table_ptr[neigh_base + j + 1];
            int neigh_idx_2 = neighborhood_index_table_ptr[neigh_base + j + 2];
            int neigh_idx_3 = neighborhood_index_table_ptr[neigh_base + j + 3];

            int idx_dist_0;
            int idx_dist_1;
            int idx_dist_2;
            int idx_dist_3;

            if (i < neigh_idx_0) {
                idx_dist_0 = dist_base + neigh_idx_0;
            } else {
                idx_dist_0 = num_pts * neigh_idx_0 + i;
            }
            double k_dist_0 = k_distances_indexed_ptr[neigh_idx_0];
            double dist_p_0 = distances_indexed_ptr[idx_dist_0];
            sum += k_dist_0 >= dist_p_0 ? k_dist_0 : dist_p_0;

            if (i < neigh_idx_1) {
                idx_dist_1 = dist_base + neigh_idx_1;
            } else {
                idx_dist_1 = num_pts * neigh_idx_1 + i;
            }
            double k_dist_1 = k_distances_indexed_ptr[neigh_idx_1];
            double dist_p_1 = distances_indexed_ptr[idx_dist_1];
            sum += k_dist_1 >= dist_p_1 ? k_dist_1 : dist_p_1;

            if (i < neigh_idx_2) {
                idx_dist_2 = dist_base + neigh_idx_2;
            } else {
                idx_dist_2 = num_pts * neigh_idx_2 + i;
            }
            double k_dist_2 = k_distances_indexed_ptr[neigh_idx_2];
            double dist_p_2 = distances_indexed_ptr[idx_dist_2];
            sum += k_dist_2 >= dist_p_2 ? k_dist_2 : dist_p_2;

            if (i < neigh_idx_3) {
                idx_dist_3 = dist_base + neigh_idx_3;
            } else {
                idx_dist_3 = num_pts * neigh_idx_3 + i;
            }
            double k_dist_3 = k_distances_indexed_ptr[neigh_idx_3];
            double dist_p_3 = distances_indexed_ptr[idx_dist_3];
            sum += k_dist_3 >= dist_p_3 ? k_dist_3 : dist_p_3;
        }

        for (; j < k; j++) {
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
    }

}

/**
 *
 * Inner unroll 8
 *
 */
double ComputeLocalReachabilityDensityMergedUnroll_Fastest(int k, int num_pts, const double* distances_indexed_ptr,
                                                           const double* k_distances_indexed_ptr,
                                                           const int* neighborhood_index_table_ptr,
                                                           double* lrd_score_table_ptr) {

    int i, j;
    for (i = 0; i < num_pts; i++) {
        double sum = 0;

        int neigh_base = i * k;
        int dist_base = num_pts * i;

        for (j = 0; j + 7 < k; j += 8) {
            int neigh_idx_0 = neighborhood_index_table_ptr[neigh_base + j];
            int neigh_idx_1 = neighborhood_index_table_ptr[neigh_base + j + 1];
            int neigh_idx_2 = neighborhood_index_table_ptr[neigh_base + j + 2];
            int neigh_idx_3 = neighborhood_index_table_ptr[neigh_base + j + 3];
            int neigh_idx_4 = neighborhood_index_table_ptr[neigh_base + j + 4];
            int neigh_idx_5 = neighborhood_index_table_ptr[neigh_base + j + 5];
            int neigh_idx_6 = neighborhood_index_table_ptr[neigh_base + j + 6];
            int neigh_idx_7 = neighborhood_index_table_ptr[neigh_base + j + 7];

            int idx_dist_0;
            int idx_dist_1;
            int idx_dist_2;
            int idx_dist_3;
            int idx_dist_4;
            int idx_dist_5;
            int idx_dist_6;
            int idx_dist_7;

            if (i < neigh_idx_0) {
                idx_dist_0 = dist_base + neigh_idx_0;
            } else {
                idx_dist_0 = num_pts * neigh_idx_0 + i;
            }
            double k_dist_0 = k_distances_indexed_ptr[neigh_idx_0];
            double dist_p_0 = distances_indexed_ptr[idx_dist_0];
            sum += k_dist_0 >= dist_p_0 ? k_dist_0 : dist_p_0;

            if (i < neigh_idx_1) {
                idx_dist_1 = dist_base + neigh_idx_1;
            } else {
                idx_dist_1 = num_pts * neigh_idx_1 + i;
            }
            double k_dist_1 = k_distances_indexed_ptr[neigh_idx_1];
            double dist_p_1 = distances_indexed_ptr[idx_dist_1];
            sum += k_dist_1 >= dist_p_1 ? k_dist_1 : dist_p_1;

            if (i < neigh_idx_2) {
                idx_dist_2 = dist_base + neigh_idx_2;
            } else {
                idx_dist_2 = num_pts * neigh_idx_2 + i;
            }
            double k_dist_2 = k_distances_indexed_ptr[neigh_idx_2];
            double dist_p_2 = distances_indexed_ptr[idx_dist_2];
            sum += k_dist_2 >= dist_p_2 ? k_dist_2 : dist_p_2;

            if (i < neigh_idx_3) {
                idx_dist_3 = dist_base + neigh_idx_3;
            } else {
                idx_dist_3 = num_pts * neigh_idx_3 + i;
            }
            double k_dist_3 = k_distances_indexed_ptr[neigh_idx_3];
            double dist_p_3 = distances_indexed_ptr[idx_dist_3];
            sum += k_dist_3 >= dist_p_3 ? k_dist_3 : dist_p_3;


            if (i < neigh_idx_4) {
                idx_dist_4 = dist_base + neigh_idx_4;
            } else {
                idx_dist_4 = num_pts * neigh_idx_4 + i;
            }
            double k_dist_4 = k_distances_indexed_ptr[neigh_idx_4];
            double dist_p_4 = distances_indexed_ptr[idx_dist_4];
            sum += k_dist_4 >= dist_p_4 ? k_dist_4 : dist_p_4;


            if (i < neigh_idx_5) {
                idx_dist_5 = dist_base + neigh_idx_5;
            } else {
                idx_dist_5 = num_pts * neigh_idx_5 + i;
            }
            double k_dist_5 = k_distances_indexed_ptr[neigh_idx_5];
            double dist_p_5 = distances_indexed_ptr[idx_dist_5];
            sum += k_dist_5 >= dist_p_5 ? k_dist_5 : dist_p_5;


            if (i < neigh_idx_6) {
                idx_dist_6 = dist_base + neigh_idx_6;
            } else {
                idx_dist_6 = num_pts * neigh_idx_6 + i;
            }
            double k_dist_6 = k_distances_indexed_ptr[neigh_idx_6];
            double dist_p_6 = distances_indexed_ptr[idx_dist_6];
            sum += k_dist_6 >= dist_p_6 ? k_dist_6 : dist_p_6;


            if (i < neigh_idx_7) {
                idx_dist_7 = dist_base + neigh_idx_7;
            } else {
                idx_dist_7 = num_pts * neigh_idx_7 + i;
            }
            double k_dist_7 = k_distances_indexed_ptr[neigh_idx_7];
            double dist_p_7 = distances_indexed_ptr[idx_dist_7];
            sum += k_dist_7 >= dist_p_7 ? k_dist_7 : dist_p_7;
        }

        for (; j < k; j++) {
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
    }

    return num_pts * (2.0 * k + 1);
}

/**
 *
 * Change loop order
 *
 */



/**
 * Linear indexing simple
 */
void ComputeLocalReachabilityDensityMerged_6(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {

    for (int i = 0; i < num_pts; i++) {
        double sum = 0;

        for (int j = 0; j < k; j++) {
            int neigh_idx = neighborhood_index_table_ptr[i * k + j];

            int idx_dist;

            if (i < neigh_idx) {
                idx_dist = GetLinearIndex(i, neigh_idx, num_pts);
            } else {
                idx_dist = GetLinearIndex(neigh_idx, i, num_pts);
            }

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];
            sum += kdistO >= distPo ? kdistO : distPo;
        }
        lrd_score_table_ptr[i] = k / sum;
    }
}

/**
 * Linear indexing hashes_linear
 */
void ComputeLocalReachabilityDensityMerged_7(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {

    for (int i = 0; i < num_pts; i++) {
        double sum = 0;
        int base = i * k;

        for (int j = 0; j < k; j++) {
            int neigh_idx = neighborhood_index_table_ptr[base + j];

            int idx_dist = hashes_linear[i, neigh_idx];
            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];
            sum += kdistO >= distPo ? kdistO : distPo;
        }
        lrd_score_table_ptr[i] = k / sum;
    }
}


/**
 *
 * Inner unroll 8 hashes_linear linear
 *
 */
void ComputeLocalReachabilityDensityMerged_8(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {

    int i, j;
    for (i = 0; i < num_pts; i++) {
        double sum = 0;

        int neigh_base = i * k;

        for (j = 0; j + 7 < k; j += 8) {
            int neigh_idx_0 = neighborhood_index_table_ptr[neigh_base + j];
            int neigh_idx_1 = neighborhood_index_table_ptr[neigh_base + j + 1];
            int neigh_idx_2 = neighborhood_index_table_ptr[neigh_base + j + 2];
            int neigh_idx_3 = neighborhood_index_table_ptr[neigh_base + j + 3];
            int neigh_idx_4 = neighborhood_index_table_ptr[neigh_base + j + 4];
            int neigh_idx_5 = neighborhood_index_table_ptr[neigh_base + j + 5];
            int neigh_idx_6 = neighborhood_index_table_ptr[neigh_base + j + 6];
            int neigh_idx_7 = neighborhood_index_table_ptr[neigh_base + j + 7];

            int idx_dist_0;
            int idx_dist_1;
            int idx_dist_2;
            int idx_dist_3;
            int idx_dist_4;
            int idx_dist_5;
            int idx_dist_6;
            int idx_dist_7;

            idx_dist_0 = hashes_matrix[i * num_pts + neigh_idx_0];
            double k_dist_0 = k_distances_indexed_ptr[neigh_idx_0];
            double dist_p_0 = distances_indexed_ptr[idx_dist_0];
            sum += k_dist_0 >= dist_p_0 ? k_dist_0 : dist_p_0;

            idx_dist_1 = hashes_matrix[i * num_pts + neigh_idx_1];
            double k_dist_1 = k_distances_indexed_ptr[neigh_idx_1];
            double dist_p_1 = distances_indexed_ptr[idx_dist_1];
            sum += k_dist_1 >= dist_p_1 ? k_dist_1 : dist_p_1;

            idx_dist_2 = hashes_matrix[i * num_pts + neigh_idx_2];
            double k_dist_2 = k_distances_indexed_ptr[neigh_idx_2];
            double dist_p_2 = distances_indexed_ptr[idx_dist_2];
            sum += k_dist_2 >= dist_p_2 ? k_dist_2 : dist_p_2;

            idx_dist_3 = hashes_matrix[i * num_pts + neigh_idx_3];
            double k_dist_3 = k_distances_indexed_ptr[neigh_idx_3];
            double dist_p_3 = distances_indexed_ptr[idx_dist_3];
            sum += k_dist_3 >= dist_p_3 ? k_dist_3 : dist_p_3;


            idx_dist_4 = hashes_matrix[i * num_pts + neigh_idx_4];
            double k_dist_4 = k_distances_indexed_ptr[neigh_idx_4];
            double dist_p_4 = distances_indexed_ptr[idx_dist_4];
            sum += k_dist_4 >= dist_p_4 ? k_dist_4 : dist_p_4;


            idx_dist_5 = hashes_matrix[i * num_pts + neigh_idx_5];
            double k_dist_5 = k_distances_indexed_ptr[neigh_idx_5];
            double dist_p_5 = distances_indexed_ptr[idx_dist_5];
            sum += k_dist_5 >= dist_p_5 ? k_dist_5 : dist_p_5;


            idx_dist_6 = hashes_matrix[i * num_pts + neigh_idx_6];
            double k_dist_6 = k_distances_indexed_ptr[neigh_idx_6];
            double dist_p_6 = distances_indexed_ptr[idx_dist_6];
            sum += k_dist_6 >= dist_p_6 ? k_dist_6 : dist_p_6;


            idx_dist_7 = hashes_matrix[i * num_pts + neigh_idx_7];
            double k_dist_7 = k_distances_indexed_ptr[neigh_idx_7];
            double dist_p_7 = distances_indexed_ptr[idx_dist_7];
            sum += k_dist_7 >= dist_p_7 ? k_dist_7 : dist_p_7;


        }

        for (; j < k; j++) {
            int neigh_idx = neighborhood_index_table_ptr[i * k + j];

            int idx_dist;

            idx_dist = hashes_matrix[i * num_pts + neigh_idx];

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];
            sum += kdistO >= distPo ? kdistO : distPo;
        }

        lrd_score_table_ptr[i] = k / sum;
    }
}

/**
 *
 * Inner unroll 8 hashes_linear linear
 *
 */
void ComputeLocalReachabilityDensityMergedHashed(int k, int num_pts, const double* distances_indexed_ptr,
                                                 const double* k_distances_indexed_ptr,
                                                 const int* neighborhood_index_table_ptr,
                                                 const int* hashes_index,
                                                 double* lrd_score_table_ptr) {

    int i, j;
    for (i = 0; i < num_pts; i++) {
        double sum = 0;

        int neigh_base = i * k;

        for (j = 0; j + 7 < k; j += 8) {
            int neigh_idx_0 = neighborhood_index_table_ptr[neigh_base + j];
            int neigh_idx_1 = neighborhood_index_table_ptr[neigh_base + j + 1];
            int neigh_idx_2 = neighborhood_index_table_ptr[neigh_base + j + 2];
            int neigh_idx_3 = neighborhood_index_table_ptr[neigh_base + j + 3];
            int neigh_idx_4 = neighborhood_index_table_ptr[neigh_base + j + 4];
            int neigh_idx_5 = neighborhood_index_table_ptr[neigh_base + j + 5];
            int neigh_idx_6 = neighborhood_index_table_ptr[neigh_base + j + 6];
            int neigh_idx_7 = neighborhood_index_table_ptr[neigh_base + j + 7];

            int idx_dist_0;
            int idx_dist_1;
            int idx_dist_2;
            int idx_dist_3;
            int idx_dist_4;
            int idx_dist_5;
            int idx_dist_6;
            int idx_dist_7;

            idx_dist_0 = hashes_index[i * num_pts + neigh_idx_0];
            double k_dist_0 = k_distances_indexed_ptr[neigh_idx_0];
            double dist_p_0 = distances_indexed_ptr[idx_dist_0];
            sum += k_dist_0 >= dist_p_0 ? k_dist_0 : dist_p_0;

            idx_dist_1 = hashes_index[i * num_pts + neigh_idx_1];
            double k_dist_1 = k_distances_indexed_ptr[neigh_idx_1];
            double dist_p_1 = distances_indexed_ptr[idx_dist_1];
            sum += k_dist_1 >= dist_p_1 ? k_dist_1 : dist_p_1;

            idx_dist_2 = hashes_index[i * num_pts + neigh_idx_2];
            double k_dist_2 = k_distances_indexed_ptr[neigh_idx_2];
            double dist_p_2 = distances_indexed_ptr[idx_dist_2];
            sum += k_dist_2 >= dist_p_2 ? k_dist_2 : dist_p_2;

            idx_dist_3 = hashes_index[i * num_pts + neigh_idx_3];
            double k_dist_3 = k_distances_indexed_ptr[neigh_idx_3];
            double dist_p_3 = distances_indexed_ptr[idx_dist_3];
            sum += k_dist_3 >= dist_p_3 ? k_dist_3 : dist_p_3;


            idx_dist_4 = hashes_index[i * num_pts + neigh_idx_4];
            double k_dist_4 = k_distances_indexed_ptr[neigh_idx_4];
            double dist_p_4 = distances_indexed_ptr[idx_dist_4];
            sum += k_dist_4 >= dist_p_4 ? k_dist_4 : dist_p_4;


            idx_dist_5 = hashes_index[i * num_pts + neigh_idx_5];
            double k_dist_5 = k_distances_indexed_ptr[neigh_idx_5];
            double dist_p_5 = distances_indexed_ptr[idx_dist_5];
            sum += k_dist_5 >= dist_p_5 ? k_dist_5 : dist_p_5;


            idx_dist_6 = hashes_index[i * num_pts + neigh_idx_6];
            double k_dist_6 = k_distances_indexed_ptr[neigh_idx_6];
            double dist_p_6 = distances_indexed_ptr[idx_dist_6];
            sum += k_dist_6 >= dist_p_6 ? k_dist_6 : dist_p_6;


            idx_dist_7 = hashes_index[i * num_pts + neigh_idx_7];
            double k_dist_7 = k_distances_indexed_ptr[neigh_idx_7];
            double dist_p_7 = distances_indexed_ptr[idx_dist_7];
            sum += k_dist_7 >= dist_p_7 ? k_dist_7 : dist_p_7;


        }

        for (; j < k; j++) {
            int neigh_idx = neighborhood_index_table_ptr[i * k + j];

            int idx_dist;

            idx_dist = hashes_index[i * num_pts + neigh_idx];

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];
            sum += kdistO >= distPo ? kdistO : distPo;
        }

        lrd_score_table_ptr[i] = k / sum;
    }
}


/**
 *
 * Inner unroll 8 hashes_matrix matrix
 *
 */
void ComputeLocalReachabilityDensityMerged_9(int k, int num_pts, const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {

    int i, j;
    for (i = 0; i < num_pts; i++) {
        double sum = 0;

        int neigh_base = i * k;

        for (j = 0; j + 7 < k; j += 8) {
            int neigh_idx_0 = neighborhood_index_table_ptr[neigh_base + j];
            int neigh_idx_1 = neighborhood_index_table_ptr[neigh_base + j + 1];
            int neigh_idx_2 = neighborhood_index_table_ptr[neigh_base + j + 2];
            int neigh_idx_3 = neighborhood_index_table_ptr[neigh_base + j + 3];
            int neigh_idx_4 = neighborhood_index_table_ptr[neigh_base + j + 4];
            int neigh_idx_5 = neighborhood_index_table_ptr[neigh_base + j + 5];
            int neigh_idx_6 = neighborhood_index_table_ptr[neigh_base + j + 6];
            int neigh_idx_7 = neighborhood_index_table_ptr[neigh_base + j + 7];

            int idx_dist_0;
            int idx_dist_1;
            int idx_dist_2;
            int idx_dist_3;
            int idx_dist_4;
            int idx_dist_5;
            int idx_dist_6;
            int idx_dist_7;

            idx_dist_0 = hashes_matrix[i * num_pts + neigh_idx_0];
            double k_dist_0 = k_distances_indexed_ptr[neigh_idx_0];
            double dist_p_0 = distances_indexed_ptr[idx_dist_0];
            sum += k_dist_0 >= dist_p_0 ? k_dist_0 : dist_p_0;

            idx_dist_1 = hashes_matrix[i * num_pts + neigh_idx_1];
            double k_dist_1 = k_distances_indexed_ptr[neigh_idx_1];
            double dist_p_1 = distances_indexed_ptr[idx_dist_1];
            sum += k_dist_1 >= dist_p_1 ? k_dist_1 : dist_p_1;

            idx_dist_2 = hashes_matrix[i * num_pts + neigh_idx_2];
            double k_dist_2 = k_distances_indexed_ptr[neigh_idx_2];
            double dist_p_2 = distances_indexed_ptr[idx_dist_2];
            sum += k_dist_2 >= dist_p_2 ? k_dist_2 : dist_p_2;

            idx_dist_3 = hashes_matrix[i * num_pts + neigh_idx_3];
            double k_dist_3 = k_distances_indexed_ptr[neigh_idx_3];
            double dist_p_3 = distances_indexed_ptr[idx_dist_3];
            sum += k_dist_3 >= dist_p_3 ? k_dist_3 : dist_p_3;


            idx_dist_4 = hashes_matrix[i * num_pts + neigh_idx_4];
            double k_dist_4 = k_distances_indexed_ptr[neigh_idx_4];
            double dist_p_4 = distances_indexed_ptr[idx_dist_4];
            sum += k_dist_4 >= dist_p_4 ? k_dist_4 : dist_p_4;


            idx_dist_5 = hashes_matrix[i * num_pts + neigh_idx_5];
            double k_dist_5 = k_distances_indexed_ptr[neigh_idx_5];
            double dist_p_5 = distances_indexed_ptr[idx_dist_5];
            sum += k_dist_5 >= dist_p_5 ? k_dist_5 : dist_p_5;


            idx_dist_6 = hashes_matrix[i * num_pts + neigh_idx_6];
            double k_dist_6 = k_distances_indexed_ptr[neigh_idx_6];
            double dist_p_6 = distances_indexed_ptr[idx_dist_6];
            sum += k_dist_6 >= dist_p_6 ? k_dist_6 : dist_p_6;


            idx_dist_7 = hashes_matrix[i * num_pts + neigh_idx_7];
            double k_dist_7 = k_distances_indexed_ptr[neigh_idx_7];
            double dist_p_7 = distances_indexed_ptr[idx_dist_7];
            sum += k_dist_7 >= dist_p_7 ? k_dist_7 : dist_p_7;


        }

        for (; j < k; j++) {
            int neigh_idx = neighborhood_index_table_ptr[i * k + j];

            int idx_dist;

            idx_dist = hashes_matrix[i * num_pts + neigh_idx];

            double kdistO = k_distances_indexed_ptr[neigh_idx];
            double distPo = distances_indexed_ptr[idx_dist];
            sum += kdistO >= distPo ? kdistO : distPo;
        }

        lrd_score_table_ptr[i] = k / sum;
    }
}

#define NUM_FUNCTIONS 10

int lrdensity_merged_driver_unrolled(int num_pts, int k, int dim, int num_reps) {

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &SeparateBaseline;
    fun_array[1] = &ComputeLocalReachabilityDensityMerged_1;
    fun_array[2] = &ComputeLocalReachabilityDensityMerged_2;
    fun_array[3] = &ComputeLocalReachabilityDensityMerged_3;
    fun_array[4] = &ComputeLocalReachabilityDensityMergedUnroll_Fastest;
    // fun_array[5] = &ComputeLocalReachabilityDensityMerged_5;
    fun_array[5] = &ComputeLocalReachabilityDensityMerged_6;
    fun_array[6] = &ComputeLocalReachabilityDensityMerged_7;
    fun_array[7] = &ComputeLocalReachabilityDensityMerged_8;
    fun_array[8] = &ComputeLocalReachabilityDensityMerged_9;

    char* fun_names[NUM_FUNCTIONS] = {"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9"};

    hashes_linear = XmallocMatrixInt(num_pts, num_pts);
    for (int i = 0; i < num_pts - 1; i++) {
        for (int j = i + 1; j < num_pts; j++) {
            hashes_linear[i * num_pts + j] = GetLinearIndex(i, j, num_pts);
            hashes_linear[j * num_pts + i] = hashes_linear[i * num_pts + j];
        }
    }

    hashes_matrix = XmallocMatrixInt(num_pts, num_pts);
    for (int i = 0; i < num_pts - 1; i++) {
        for (int j = i + 1; j < num_pts; j++) {
            hashes_matrix[i * num_pts + j] = num_pts * i + j;
            hashes_matrix[j * num_pts + i] = num_pts * j + i;
        }
    }

    myInt64 start, end;
    double cycles1;


//    FILE* results_file = open_with_error_check("../lrd_benchmark_k10_full.txt", "a");

    // INITIALIZE RANDOM INPUT
    int* neighborhood_index_table_ptr = XmallocMatrixIntRandom(num_pts, k, num_pts);
    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDoubleRandom(num_pts);

    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    double* lrd_score_table_ptr_true = XmallocVectorDouble(num_pts);


    for (int fun_index = 1; fun_index < NUM_FUNCTIONS; fun_index++) {
        free(lrd_score_table_ptr);
        lrd_score_table_ptr = XmallocVectorDouble(num_pts);

        SeparateBaseline(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr, neighborhood_index_table_ptr,
                         lrd_score_table_ptr_true);
        // VERIFICATION : ------------------------------------------------------------------------
        (*fun_array[fun_index])(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                neighborhood_index_table_ptr,
                                lrd_score_table_ptr);

        int ver = test_double_arrays(num_pts, 1e-3, lrd_score_table_ptr, lrd_score_table_ptr_true);
        if (ver != 1) {
            printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
//            exit(-1);
        }
        double multiplier = 1;
        double numRuns = 10;

        // Warm-up phase: we determine a number of executions that allows
        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[fun_index])(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                        neighborhood_index_table_ptr, lrd_score_table_ptr);

            }
            end = stop_tsc(start);

            cycles1 = (double) end;
            multiplier = (CYCLES_REQUIRED) / (cycles1);

        } while (multiplier > 2);


        double totalCycles = 0;
        double* cyclesPtr = XmallocVectorDouble(num_reps);

        for (size_t j = 0; j < num_reps; j++) {
            free(lrd_score_table_ptr);
            lrd_score_table_ptr = XmallocVectorDouble(num_pts);

            start = start_tsc();
            for (size_t i = 0; i < numRuns; ++i) {
                (*fun_array[fun_index])(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                        neighborhood_index_table_ptr, lrd_score_table_ptr);

            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;

            totalCycles += cycles1;
        }

        qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[(int) num_reps / 2 + 1];


        double flops = fun_index > 0 ? num_pts * (2 * k + 1) : (num_pts * (num_pts - 1 + 2 + k));
        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        free(cyclesPtr);
        printf("%s n:%d cycles:%lf perf:%lf \n", fun_names[fun_index], num_pts, cycles, perf);
//        fprintf(results_file, "%s, %d, %d, %lf, %lf\n", fun_names[fun_index], num_pts, k, cycles, perf);
    }

    printf("-------------\n");
    free(lrd_score_table_ptr_true);
    free(lrd_score_table_ptr);
    free(distances_indexed_ptr);
    free(k_distances_indexed_ptr);
    free(neighborhood_index_table_ptr);
//    fclose(results_file);
    return 0;
}


