//
// Created by Sycheva  Anastasia on 15.05.20.
//
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "../include/lof_baseline.h"
#include "../include/utils.h"


double ComputePairwiseDistances(int dim, int num_pts, const double* input_points_ptr, double (* fct)(const double*,
                                                                                                     const double*,
                                                                                                     int),
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
            distances_indexed_ptr[i * num_pts + j] = (*fct)(&input_points_ptr[i * dim], &input_points_ptr[j * dim],
                                                            dim);
        }
    }
    return (num_pts / 2.0 * num_pts) * (4.0 * dim + 1.0);
}
// Definition 3 ------------------------------------------------------>

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
    free(dist_to_obj);
}


double ComputeKDistanceAll(int k, int num_pts, const double* distances_indexed_ptr, double* k_distances_indexed_ptr) {
    /** Implement Definition 3 from the paper i.e. for each point it's distance to the farthest k'th neighbor
   * @param obj_idx: index of the object
   *
   * @return: fill k_distances_indexed_ptr
  */
    int idx;
    for (idx = 0; idx < num_pts; idx++) {
        ComputeKDistanceObject(idx, k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    }
    return num_pts * (num_pts - 1.0) * log(num_pts - 1.0);
}

// Definition 4 ------------------------------------------------------>

double
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

    return num_pts * (num_pts - 1.0);
}

// Definition 5 ------------------------------------------------------>

void ComputeReachabilityDistanceAll(int k, int num_pts, const double* distances_indexed_ptr,
                                    const double* k_distances_indexed_ptr, double* reachability_distances_indexed_ptr) {
    int idx_from, idx_to;
    for (idx_from = 0; idx_from < num_pts; ++idx_from) {
        for (idx_to = 0; idx_to < num_pts; ++idx_to) {
            if (idx_from != idx_to) {

                int idxDist = idx_from < idx_to ? num_pts * idx_from + idx_to : num_pts * idx_to +
                                                                                idx_from;
                int idxReach = idx_from * num_pts + idx_to;

                double kdistO = k_distances_indexed_ptr[idx_to];
                double distPo = distances_indexed_ptr[idxDist];

                reachability_distances_indexed_ptr[idxReach] = kdistO >= distPo ? kdistO : distPo;
            }
        }
    }
}

// Definition 5 + 6 -------------------------------------------------->

double ComputeFlopsReachabilityDensityMerged(int k, int num_pts) {
    // num_pts * div + k * num_pts * ( add + 2*comp )
    assert(num_pts >= k);
    return num_pts + 3 * k * num_pts;
}

double ComputeLocalReachabilityDensityMerged(int k, int num_pts,
                                             const double* distances_indexed_ptr,
                                             const double* k_distances_indexed_ptr,
                                             const int* neighborhood_index_table_ptr,
                                             double* lrd_score_table_ptr) {

    double sum;
    for (int i = 0; i < num_pts; i++) {
        sum = 0;

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

    return num_pts + 3 * k * num_pts;
}

// Definition 6 ------------------------------------------------------>

void ComputeLocalReachabilityDensity(int k, int num_pts, const double* reachability_distances_indexed_ptr,
                                     const int* neighborhood_index_table_ptr, double* lrd_score_table_ptr) {
    /** Implement Definition 6 from the paper
     *
     * @param reachability_distances_indexed_ptr: Indexed reachability distances
     * @param neighborhood_index_table_ptr: Neighbors of all the points
     * @param n: number of points in the dataset
     * @param k: neighbors
     *
     * @return: Void (lrd_score_table_ptr containing Local Reachability Density Score)
     *
    */
    // BITMAP ?
    for (int i = 0; i < num_pts; i++) {

        double sum = 0;
        for (int j = 0; j < k; j++) {
//            int neigh_point = neighborhood_index_table_ptr[i * k + j];
            sum += reachability_distances_indexed_ptr[i * num_pts + j];
        }
        double mean = sum / k;
        lrd_score_table_ptr[i] = 1.0 / (mean);

    }
}

// Definition 7 ------------------------------------------------------>

double ComputeFlopsLocalOutlierFactor(int k, int num_pts) {
    return num_pts * (2 + k);
}

double ComputeLocalOutlierFactor(int k, int num_pts, const double* lrd_score_table_ptr,
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
    return num_pts * (2 + k);
}

// ------------------------------------------------------------------------------------------->
// DOWNSTREAM FUNCTIONS FOR PIPELINE 2 & 3

/**
 * GENERAL COMMENTS
 *
 * 1. I hope I understood the intended restructuring correctly, otherwise will adjust tomorrow
 * 2. I do not see a lot of room for vectorization of ComputeLocalReachabilityDensityMerged_Ver2
 * 3. I do not think the total number of random memory accesses has decreased compared to the previous version of the pipleline
 * 4. This modification hopefully improves access pattern and can be viewed as a memory optimization ?
 *
 * TODO
 * 1. VERIFY PIPELINE 2 !
 */

void ComputeLocalReachabilityDensityMerged_Point(int pnt_idx, int k,
                                                 const double* dist_k_neighborhood_index,
                                                 const double* k_distance_index,
                                                 const int* k_neighborhood_index,
                                                 double* lrd_score_table_ptr) {
    /**
     * Compute lrd score for @param pnt_idx
     * See ComputeLocalReachabilityDensityMerged_Pipeline2 for argument description
     */
    double sum = 0;
    for (int j = 0; j < k; j++) {

        // get index of j-th neighbor of pnt_idx
        int neigh_idx = k_neighborhood_index[pnt_idx * k + j];
        double kdistO = k_distance_index[neigh_idx];   //  dist_k_neighborhood_index[ neigh_idx*k + k-1]
        // pariwise distance between pnt_idx and j-th neighbor
        double distPo = dist_k_neighborhood_index[pnt_idx * k + j];
        sum += kdistO >= distPo ? kdistO : distPo;

    }  // 2*k + 1

    double result = k / sum;
    //printf("Compute lrd for %d result %lf\n", pnt_idx, result);
    lrd_score_table_ptr[pnt_idx] = result; // move division for later ? => does not seem worth it ...
}   // ComputeLocalReachabilityDensityMerged_Point


double ComputeLocalReachabilityDensityMerged_Pipeline2(int k, int num_pts,
                                                       my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                                       const double* dist_k_neighborhood_index,
                                                       const int* k_neighborhood_index,
                                                       double* k_distance_index,
                                                       double* lrd_score_table_ptr,
                                                       double* lrd_score_neigh_table_ptr) {
    /**
     *
     * @param dist_k_neighborhood_index: an array of size num_pts * k
     *                              k_dist_index[ i * k + j ], where i < num_pts, j < k
     *                              stores the **pairwise distance** of neighbor j of point i
     *                              returned by the function KNN
     *
     *
     * @param k_neighborhood_index: an array of size num_pts * k
     *                              k_neighborhood_index[ i*k + j ], where i < num_pts, j < k
     *                              stores the index of the jth (of k nearest neighbors) of the point i
     *                              returned by the function KNN
     *
     * @return fill the following objects:
     *
     *         (potentially move to KNN)
     *         k_distance_index: an array of size num_pts
     *                              k_distance_index[ i ] stores the k-distance of point i
     *
     *         lrd_score_table_ptr: an array of size num_pts
     *                              lrd_score_table_ptr[ i ] contains the lrd score of point i
     *                              IMPORTAT: should be initialized with negative numbers
     *
     *         lrd_score_neigh_table_ptr: an array of size num_pts * k
     *                               lrd_score_neigh_table_ptr[ i*k + j ], where i < num_pts, j < k
     *                               stores the lrd score of the jth (of k nearest neighbors) of the point i
     *
     * @important: lrd_score_table_ptr has to be initialized with 0!
     */
    // Compute K distance for every object

    //  SLOW !!!
    /*
    for(int i=0; i < num_pts; i++){
        double max = -1.0;     // K distance is the max distance among k nearest neighbors
        for( int j=0; j < k; j++ ){
            if( dist_k_neighborhood_index[ i*k + j ] > max ){
                // LAST ELEMENT !
                max = dist_k_neighborhood_index[ i*k + j ];
            }
        } // for j
        // k_distance_index[ i ] = dist_k_neighborhood_index[ i*k + k-2 ]; // max;
        k_distance_index[ i ] = max;
    } // for i
     */

    // FILL IN k_distance_index
    for (int i = 0; i < num_pts; i++) {
        k_distance_index[i] = dist_k_neighborhood_index[i * k + k - 1]; // CHECK !!!!!!!!!!!
    } // for i

    // (2*k + 1) every time  go in ComputeLocalReachabilityDensityMerged_Point
    double total_flops = 0;
    //int count=0;
    for (int i = 0; i < num_pts; i++) {   // iterate over all points
        for (int j = 0; j < k; j++) {      // iterate over all their neighbors

            total_flops += 1.0; // add for comparison

            int idx_j_neighbor = k_neighborhood_index[i * k + j];
            // printf("Process Point %d\n", idx_j_neighbor);
            if (lrd_score_table_ptr[idx_j_neighbor] > 0.0) {    // if already computed
                lrd_score_neigh_table_ptr[i * k + j] = lrd_score_table_ptr[idx_j_neighbor];
            } else {
                lrdm2_pnt_fnc(idx_j_neighbor, k, dist_k_neighborhood_index, k_distance_index,
                              k_neighborhood_index, lrd_score_table_ptr);

                total_flops += 2 * k + 1; // computations inside

                lrd_score_neigh_table_ptr[i * k + j] = lrd_score_table_ptr[idx_j_neighbor];
                // count ++;
            }
        } // for j

        // SOME POINTS THEY ARE NOT IN THE NEIGHBORHOOD OF ANY OTHER POINT
        // THEIR LRDs won't be computed unless we explicitely ensure they are
        // CAN SPOIL THE LOCALITY ... (hopefully not in many cases)
        if (lrd_score_table_ptr[i] < 0.0) {

            lrdm2_pnt_fnc(i, k, dist_k_neighborhood_index, k_distance_index,
                          k_neighborhood_index, lrd_score_table_ptr);

            total_flops += 2 * k + 1;
            // count ++;
        }
        total_flops += 1; // for comparison with lrd_score_table_ptr[ i ]
    } // for i

    return total_flops;
    //printf("\nComputed LOF %d times\n\n", count);
}  // ComputeLocalReachabilityDensityMerged_Pipeline2


double ComputeLocalOutlierFactor_Pipeline2(int k, int num_pts,
                                           const double* lrd_score_table_ptr,
                                           const double* lrd_score_neigh_table_ptr,
                                           double* lof_score_table_ptr) {
    /**
     *@param lrd_score_table_ptr_tmp: an array of size num_pts
     *                                lrd_score_table_ptr[ i ] contains lrd score of point i
     *
     *@param lrd_score_neigh_table_tmp_ptr: an array of size num_pts * k
     *                                 lrd_score_neigh_table_ptr[ i*k + j ], where i < num_pts, j < k
     *                                 stores the lrd score of the jth (of k nearest neighbors) of the point i
     */

    for (int i = 0; i < num_pts; ++i) {

        double lrd_neighs_sum = 0;
        for (int j = 0; j < k; ++j) {
            lrd_neighs_sum += lrd_score_neigh_table_ptr[i * k + j];
        }  // for j
        lof_score_table_ptr[i] = lrd_neighs_sum / (lrd_score_table_ptr[i] * k);
    } // for i

    return num_pts * k + 2.0 * num_pts;
}   // ComputeLocalOutlierFactor_Pipeline2


// ***********************************************************************************
// PIPELINE 4:
//