#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

//#include <tests.h>
#include "../../include/tests.h"
#include "../../include/tsc_x86.h"
#include "../../include/file_utils.h"

#include "../include/lof.h"
#include "../include/utils.h"
#include "../include/metrics.h"


#define NUM_RUNS 2000


// -------------------------------------------------------------------------------------------------------------------------------------------

void ComputePairwiseDistances(int dim, int num_pts, const double* input_points_ptr, double (* fct)(const double*,
                                                                                                   const double*, int),
                              double* distances_indexed_ptr) {
    /**
     * @param fct: type of metrics function used, see metrics.h
     * @param dim: dimensionality of the input array
     * @param num_pts: number of points in the input array
     *
     * @return: fill distances_indexed_ptr
     *          an array with n * n/2 elements such that
     *          distances_indexed_ptr[i * n  + j] = dist(p_i, p_j) for i < j
     */
    int j, i;
    for (j = 0; j < num_pts; j++) {
        for (i = 0; i < j; i++) {
//            printf("%d \n", i * num_pts + j);
            distances_indexed_ptr[i * num_pts + j] = (*fct)(&input_points_ptr[i * dim], &input_points_ptr[j * dim],
                                                            dim);
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------------

void ComputeKDistanceObject(int obj_idx, int k, int num_pts, const double* distances_indexed_ptr,
                            double* k_distances_indexed_ptr) {
    /** Implement Definition 3 from the paper
     * @param obj_idx: index of the object
     *
     * @return: fill k_distances_indexed_ptr
    */

    int i;
    double* dist_to_obj = XmallocVectorDouble(num_pts - 1); // collect distances to obj_idx

    for (i = 0; i < obj_idx; i++) {
        // distances_indexed_ptr[i * n  + j] = dist(p_i, p_j) for i < j
        dist_to_obj[i] = distances_indexed_ptr[i * num_pts + obj_idx]; // check this later !!!
    }

    for (i = obj_idx + 1; i < num_pts; i++) {
        dist_to_obj[i - 1] = distances_indexed_ptr[obj_idx * num_pts + i]; // check this later !!!
    }

    qsort(dist_to_obj, num_pts - 1, sizeof(double), compare_double);
    while (dist_to_obj[k] == dist_to_obj[k - 1] && k > 1) {
        k = k - 1;
    }
    k_distances_indexed_ptr[obj_idx] = dist_to_obj[k - 1];
}


void ComputeKDistanceAll(int k, int num_pts, const double* distances_indexed_ptr, double* k_distances_indexed_ptr) {
    /** Implement Definition 3 from the paper i.e. for each point it's distance to the farthest k'th neighbor
   * @param obj_idx: index of the object
   *
   * @return: fill k_distances_indexed_ptr
  */
    int idx;
    for (idx = 0; idx < num_pts; idx++) {
        ComputeKDistanceObject(idx, k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------------

void
ComputeKDistanceNeighborhoodAll(int num_pts, int k, const double* k_distances_indexed_ptr,
                                const double* distances_indexed_ptr, int* neighborhood_index_table_ptr) {
    /** Implement Definition 4 from the paper
    *
    * @param obj_idx: index of an object, with respect to which the distance is computed
    * @param n: number of points in the dataset
    *
    * @return: fill the elements of the table table neighborhood_index_table_ptr, corresponding to obj_idx
    *          s.t. the k entries to the i'th line correspond to the k nearest neighbors i.e the points with
     *          distance <= kdist
    * @note:
   */

    int idx, i;
    for (idx = 0; idx < num_pts; idx++) {
        double kdist = k_distances_indexed_ptr[idx];
        int k_so_far = 0;

        for (i = 0; i < idx; i++) {
            if (distances_indexed_ptr[num_pts * i + idx] <= kdist) {

                neighborhood_index_table_ptr[idx * k + k_so_far] = i;
                k_so_far++;
            }
        }

        for (i = idx + 1; i < num_pts; i++) {
            if (distances_indexed_ptr[idx * num_pts + i] <= kdist) {

                neighborhood_index_table_ptr[idx * k + k_so_far] = i;
                k_so_far++;
            }
        }

    }

}

// -------------------------------------------------------------------------------------------------------------------------------------------

void ComputeReachabilityDistanceObject(int obj_from_idx, int obj_to_idx, int k, int num_pts,
                                       const double* distances_indexed_ptr, const double* k_distances_indexed_ptr,
                                       double* reachability_distances_indexed_ptr) {
    /** Implement Definition 5 from the paper
     *
     *
     * @param obj_from_idx: obj o in def 5
     * @param obj_to_idx: obj p in def 5
     *
     * @return: fill reachability_distances_indexed_ptr
     *          reachability_distances_indexed_ptr[obj_from_idx * num_pts  + obj_to_idx]
     *          contains the reachability distance of an object obj_to_idx w.r.t obj_from_idx
    */

    int idx_dist = obj_from_idx < obj_to_idx ? num_pts * obj_from_idx + obj_to_idx : num_pts * obj_to_idx +
                                                                                     obj_from_idx;
    int idx_reach = obj_from_idx * num_pts + obj_to_idx;

    double kdist_o = k_distances_indexed_ptr[obj_to_idx];
    double dist_p_o = distances_indexed_ptr[idx_dist];

    reachability_distances_indexed_ptr[idx_reach] = kdist_o >= dist_p_o ? kdist_o : dist_p_o;

}


void ComputeReachabilityDistanceAll(int k, int num_pts, const double* distances_indexed_ptr,
                                    const double* k_distances_indexed_ptr, double* reachability_distances_indexed_ptr) {
    int idx_from, idx_to;
    for (idx_from = 0; idx_from < num_pts; ++idx_from) {  // Q:
        for (idx_to = 0; idx_to < num_pts; ++idx_to) {
            if (idx_from != idx_to)
                ComputeReachabilityDistanceObject(idx_from, idx_to, k, num_pts, distances_indexed_ptr,
                                                  k_distances_indexed_ptr, reachability_distances_indexed_ptr);
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------------------

void ComputeLocalReachabilityDensity(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                     const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr) {
    /** Implement Definition 6 from the paper
     *
     * @param distances_indexed_ptr: Indexed distances
     * @param neighborhood_index_table_ptr: Neighbors of all the points
     * @param n: number of points in the dataset
     * @param k: neighbors
     *
     * @return: Void (lrd_score_table_ptr containing Local Reachability Density Score)
    */

    for (int i = 0; i < num_pts; i++) {
        double sum = 0;

        for (int j = 0; j < k; j++) {
            int neigh_point = neighborhood_index_table_ptr[i * k + j];
            sum += reachability_distances_indexed_ptr[i * num_pts + neigh_point];
        }
        double mean = sum / k;
        lrd_score_table_ptr[i] = 1.0 / (mean);
    }

}

// -------------------------------------------------------------------------------------------------------------------------------------------

void ComputeLocalOutlierFactor(int k, int num_pts, const double* lrd_score_table_ptr,
                               const int* neighborhood_index_table_ptr,
                               double* lof_score_table_ptr) {
    /**
     * Implement Definition 7
     *
     * @return:  fill lof_score_table_ptr
     */

    //    for each point I want to compute LOF
    for (int i = 0; i < num_pts; i++) {
        //        first I need the neighbours index with their lrd_score
        double lrd_neighs_sum = 0;
        for (int j = 0; j < k; j++) {
            //Store in array -> Slower
            int neigh_index = neighborhood_index_table_ptr[i * k + j];
            lrd_neighs_sum += lrd_score_table_ptr[neigh_index];
        }
        lof_score_table_ptr[i] = (lrd_neighs_sum / lrd_score_table_ptr[i]) / k;
    }

}

// -------------------------------------------------------------------------------------------------------------------------------------------

double*
LofBaseline(int num_pts, int k, int dim, enum Mode mode,
            FILE* results_file, FILE* exec_file) { //( char* input_file_name, char* lof_results_file_nam ) {
    /**
     *@param num_pts: number of points in the dataset
     *@param k: number of neighbors to consider
     *@param dim: dimensionality of the input
     *
    */



    char dir_full[50];
    sprintf(dir_full, "/n%d_k%d_dim%d/", num_pts, k, dim);
    char* input_file_name = concat(dir_full, "dataset.txt");
    char* lof_results_file_name = concat(dir_full, "lof_results.txt");

    // Step 1: read the input
    FILE* input_file = open_with_error_check(input_file_name, "r");
    FILE* lof_results_file = open_with_error_check(lof_results_file_name, "r");

    // just a multidimensional test
    char* input_meta_file_name = concat(dir_full, "metadata.txt");
    FILE* input_meta_file = open_with_error_check(input_meta_file_name, "r");
    double* min_range = NULL;
    double* max_range = NULL;

    double* input_points_ptr = LoadGenericDataFromFile(input_file, input_meta_file,
                                                       min_range, max_range, &num_pts, &dim, &k);

    double* lof_true_ptr = LoadResultsFromFile(lof_results_file, num_pts);


    // Initialize all important matrices / vectors used
    double* distances_indexed_ptr = XmallocMatrixDouble(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDouble(num_pts);
    double* reachability_distances_indexed_ptr = XmallocVectorDouble(num_pts * num_pts);
    double* lrd_score_table_ptr = XmallocVectorDouble(num_pts);
    double* lof_score_table_ptr = XmallocVectorDouble(num_pts);
    int* neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);

    int i, num_runs;
    double cyclesStep2, cyclesStep3, cyclesStep4, cyclesStep5, cyclesStep6, cyclesStep7;
    myInt64 start;
    double temp;
    num_runs = NUM_RUNS;

    //calculating cycles
    start = start_tsc();
    // Step 2: compute pairwise distance between points
    for (i = 1; i < num_runs; ++i) {
        ComputePairwiseDistances(dim, num_pts, input_points_ptr, UnrolledEuclideanDistance, distances_indexed_ptr);
    }
    cyclesStep2 = stop_tsc(start) / (double) num_runs;


    start = start_tsc();
    // Step 3: compute K distances
    for (i = 0; i < num_runs; ++i) {
        ComputeKDistanceAll(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    }
    cyclesStep3 = stop_tsc(start) / (double) num_runs;


    //calculating cycles
    start = start_tsc();
    //Step 4:  compute K neighborhoods
    for (i = 0; i < num_runs; ++i) {
        ComputeKDistanceNeighborhoodAll(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr,
                                        neighborhood_index_table_ptr);
    }
    cyclesStep4 = stop_tsc(start) / (double) num_runs;


    start = start_tsc();
    // Step 5: compute pairwise Reachability distance
    for (i = 0; i < num_runs; ++i) {
        ComputeReachabilityDistanceAll(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr,
                                       reachability_distances_indexed_ptr);
    }
    cyclesStep5 = stop_tsc(start) / (double) num_runs;


    start = start_tsc();
    // Step 6: compute reachability density
    for (i = 0; i < num_runs; ++i) {
        ComputeLocalReachabilityDensity(k, num_pts, reachability_distances_indexed_ptr, neighborhood_index_table_ptr,
                                        lrd_score_table_ptr);
    }
    cyclesStep6 = stop_tsc(start) / (double) num_runs;

    //calculating cycles
    start = start_tsc();
    // Step 7: compute lof
    for (i = 1; i < num_runs; ++i) {
        ComputeLocalOutlierFactor(k, num_pts, lrd_score_table_ptr, neighborhood_index_table_ptr, lof_score_table_ptr);
    }
    cyclesStep7 = stop_tsc(start) / (double) num_runs;

    double dif = 0;
    for (i = 0; i < num_pts; i++) {
        dif += fabs(lof_score_table_ptr[i] - lof_true_ptr[i]);
    }


    switch (mode) {
        case BASIC:
            assert(fabs(dif) < 1e-3);
            break;
        case TESTING: {
            // TODO: put it into a separate test file ?
            FILE* neigh_ind_results_file = open_with_error_check(concat(dir_full, "neigh_ind_results.txt"), "r");
            FILE* neigh_dist_results_file = open_with_error_check(concat(dir_full, "neigh_dist_results.txt"), "r");
            FILE* lrd_results_file = open_with_error_check(concat(dir_full, "lrd_results.txt"), "r");
            int* neigh_ind_true_ptr = LoadNeighIndResultsFromFile(neigh_ind_results_file, num_pts, k);
            double* neigh_dist_true_ptr = LoadNeighDistResultsFromFile(neigh_dist_results_file, num_pts, k);
            double* lrd_true_ptr = LoadLRDResultsFromFile(lrd_results_file, num_pts);

            int test_neigh_dist_result = test_neigh_dist(num_pts, k, 1e-3, k_distances_indexed_ptr,
                                                         neigh_dist_true_ptr);
            assert(test_neigh_dist_result == 1);

            int test_neigh_ind_result = test_neigh_ind(num_pts, k, neighborhood_index_table_ptr, neigh_ind_true_ptr);
            assert(test_neigh_ind_result == 1);

            int test_lrd_result = test_lrd(num_pts, 1e-5, lrd_score_table_ptr, lrd_true_ptr);
            assert(test_lrd_result == 1);

            assert(fabs(dif) < 1e-3);
        }
            break;
        case BENCHMARK: {
            double total = cyclesStep2 + cyclesStep3 + cyclesStep4 + cyclesStep5 + cyclesStep6 + cyclesStep7;
            fprintf(results_file, "%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", num_pts, k, dim, cyclesStep2, cyclesStep3,
                    cyclesStep4, cyclesStep5, cyclesStep6, cyclesStep7, total);
            printf("%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n", num_pts, k, dim, cyclesStep2, cyclesStep3,
                   cyclesStep4, cyclesStep5, cyclesStep6, cyclesStep7, total, dif);
            double freq = 2.6e9;
            fprintf(exec_file, "%d,%d,%d,%lf\n", num_pts, k, dim, total/freq, dif);
            //assert(fabs(dif) < 1e-3);
            break;
        }
    }


}



