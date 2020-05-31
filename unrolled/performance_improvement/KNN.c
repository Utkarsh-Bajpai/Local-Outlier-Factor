
// Created by fvasluia on 5/11/20.
//

# include <stdlib.h>
# include <math.h>

# include "../../include/sort.h"
# include "../../include/utils.h"
# include "../../include/file_utils.h"
# include "../../include/Algorithm.h"
# include "../../include/tsc_x86.h"
# include "../include/metrics.h"
# include "../include/ComputeKDistanceAll.h"
# include "../include/ComputeKDistanceNeighborhood.h"
//#include "../include/ComputePairwiseDistancesMMMUnroll.h"

typedef long (* test_fun)(double*, double*, double*, int*, int*, int, int, int);

#define CYCLES_REQUIRED 2e8

double KNN(int num_pts, int k, int* k_neighborhood_index, const double* pairwise_dist, double* k_dist_index) {
    // the pairwise_dist array will be a n(n-1)/2 array representing an upper diagonal matrix

    RECORD* entries = (RECORD*) malloc((num_pts - 1) * sizeof(RECORD));

    for (int i = 0; i < num_pts; ++i) {
        int local_idx = 0;
        for (int j = 0; j < num_pts; ++j) {
            if (j != i) {
                int idx;
                if (j > i) {
                    idx = j * num_pts + i;
                } else {
                    idx = i * num_pts + j;
                }
                RECORD r = {j, pairwise_dist[idx]};
                entries[local_idx++] = r;
            }
        }
        qsort(entries, num_pts - 1, sizeof(RECORD), compare_record);
        for (int l = 0; l < k; ++l) {
            RECORD local_record = entries[l];
            k_dist_index[i * k + l] = local_record.distance;
            k_neighborhood_index[i * k + l] = local_record.point_idx;
        }
    }
    return num_pts * num_pts * log(num_pts - 1);
}


double KNN_fastest(int n, int k, int* k_neighborhood_index, const double* pairwise_dist, double* k_dist_index) {

    int* entries_indices = (int*) malloc((n - 1) * sizeof(int));
    double* entries_distances = (double*) malloc((n - 1) * sizeof(double));

    for (int i = 0; i < n; ++i) {

        int local_idx = 0;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                int idx;
                if (j > i) {
                    idx = j * n + i;
                } else {
                    idx = i * n + j;
                }
                entries_indices[local_idx] = j;
                entries_distances[local_idx] = pairwise_dist[idx];
                ++local_idx;
            }
        }
        quickSortArray(entries_distances, entries_indices, 0, local_idx - 1);
        for (int l = 0; l < k; ++l) {
            k_dist_index[i * k + l] = entries_distances[l];
            k_neighborhood_index[i * k + l] = entries_indices[l];
        }
    }
    return 2 * n * (n - 1) * (log(n - 1));
}

void addEntry(int i, int j, int n, int* local_idx, const double* pairwise_dist, double* entries) {
    int li, lj;
    if (j != i) {
        li = i < j ? i : j;
        lj = i > j ? i : j;
        int linear_idx = li * n + lj;
        entries[*local_idx] = pairwise_dist[linear_idx];
        ++(*local_idx);
    }
}


void KNN4x4(int n, int k, int* k_neighborhood_index, const double* pairwise_dist, double* k_dist_index) {
    // the pairwise_dist array will be a n(n-1)/2 array representing an upper diagonal matrix

    double* entries1 = (double*) malloc((n - 1) * sizeof(double));
    double* entries2 = (double*) malloc((n - 1) * sizeof(double));
    double* entries3 = (double*) malloc((n - 1) * sizeof(double));
    double* entries4 = (double*) malloc((n - 1) * sizeof(double));

    int i, j;
    for (i = 0; (i + 4) < n; i += 4) {
        int local_idx1 = 0, local_idx2 = 0, local_idx3 = 0, local_idx4 = 0;
        int neigh_idx1 = 0, neigh_idx2 = 0, neigh_idx3 = 0, neigh_idx4 = 0;

        for (j = 0; (j + 4) < n; j += 4) {
            addEntry(i, j, n, &local_idx1, pairwise_dist, entries1);
            addEntry(i, j + 1, n, &local_idx1, pairwise_dist, entries1);
            addEntry(i, j + 2, n, &local_idx1, pairwise_dist, entries1);
            addEntry(i, j + 3, n, &local_idx1, pairwise_dist, entries1);

            addEntry(i + 1, j, n, &local_idx2, pairwise_dist, entries2);
            addEntry(i + 1, j + 1, n, &local_idx2, pairwise_dist, entries2);
            addEntry(i + 1, j + 2, n, &local_idx2, pairwise_dist, entries2);
            addEntry(i + 1, j + 3, n, &local_idx2, pairwise_dist, entries2);

            addEntry(i + 2, j, n, &local_idx3, pairwise_dist, entries3);
            addEntry(i + 2, j + 1, n, &local_idx3, pairwise_dist, entries3);
            addEntry(i + 2, j + 2, n, &local_idx3, pairwise_dist, entries3);
            addEntry(i + 2, j + 3, n, &local_idx3, pairwise_dist, entries3);

            addEntry(i + 3, j, n, &local_idx4, pairwise_dist, entries4);
            addEntry(i + 3, j + 1, n, &local_idx4, pairwise_dist, entries4);
            addEntry(i + 3, j + 2, n, &local_idx4, pairwise_dist, entries4);
            addEntry(i + 3, j + 3, n, &local_idx4, pairwise_dist, entries4);
        }
        if (j < n) {
            for (; j < n; ++j) {
                addEntry(i, j, n, &local_idx1, pairwise_dist, entries1);
                addEntry(i + 1, j, n, &local_idx2, pairwise_dist, entries2);
                addEntry(i + 2, j, n, &local_idx3, pairwise_dist, entries3);
                addEntry(i + 3, j, n, &local_idx4, pairwise_dist, entries4);
            }
        }

        qsort(entries1, n - 1, sizeof(double), compare_double);
        qsort(entries2, n - 1, sizeof(double), compare_double);
        qsort(entries3, n - 1, sizeof(double), compare_double);
        qsort(entries4, n - 1, sizeof(double), compare_double);

        double k_dist1 = entries1[k - 1];
        double k_dist2 = entries2[k - 1];
        double k_dist3 = entries3[k - 1];
        double k_dist4 = entries4[k - 1];

        for (int l = 0; l < k; ++l) {
            k_dist_index[i * k + l] = entries1[l];
            k_dist_index[(i + 1) * k + l] = entries2[l];
            k_dist_index[(i + 2) * k + l] = entries3[l];
            k_dist_index[(i + 3) * k + l] = entries4[l];
        }

        for (int l = 0; l < (n - 1); ++l) {
            if (entries1[l] < k_dist1) {
                if (l < i) {
                    k_neighborhood_index[i * k + neigh_idx1] = l;
                    ++neigh_idx1;
                } else {
                    k_neighborhood_index[i * k + neigh_idx1] = l + 1;
                    ++neigh_idx1;
                }
            }
            if (entries2[l] < k_dist2) {
                if (l < i) {
                    k_neighborhood_index[(i + 1) * k + neigh_idx1] = l;
                    ++neigh_idx2;
                } else {
                    k_neighborhood_index[(i + 1) * k + neigh_idx1] = l + 1;
                    ++neigh_idx2;
                }
            }
            if (entries3[l] < k_dist3) {
                if (l < i) {
                    k_neighborhood_index[(i + 2) * k + neigh_idx1] = l;
                    ++neigh_idx3;
                } else {
                    k_neighborhood_index[(i + 2) * k + neigh_idx1] = l + 1;
                    ++neigh_idx3;
                }
            }
            if (entries4[l] < k_dist4) {
                if (l < i) {
                    k_neighborhood_index[(i + 3) * k + neigh_idx1] = l;
                    ++neigh_idx4;
                } else {
                    k_neighborhood_index[(i + 3) * k + neigh_idx1] = l + 1;
                    ++neigh_idx4;
                }

            }

        }

    }
    if (i != n - 1) {
        double* entries = (double*) malloc((n - 1) * sizeof(double));
        for (; i < n; ++i) {
            int local_idx = 0;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    int linear_idx;
                    if (j > i) {
                        linear_idx = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
                    } else {
                        linear_idx = (n * (n - 1) / 2) - (n - j) * ((n - j) - 1) / 2 + i - j - 1;
                    }
                    entries[local_idx++] = pairwise_dist[linear_idx];
                }
            }
            qsort(entries, n - 1, sizeof(double), compare_double);
            double k_dist = entries[k - 1];
            int k_neigh_idx = 0;

            for (int l = 0; l < k; ++l) {
                k_dist_index[i * k + l] = entries[l];
            }

            for (int l = 0; l < n - 1; ++l) {
                if (entries[l] < k_dist) {
                    if (l < i) {
                        k_neighborhood_index[i * k + k_neigh_idx] = l;
                    } else {
                        k_neighborhood_index[i * k + k_neigh_idx] = l + 1;
                    }
                    ++k_neigh_idx;
                }
            }
        }
        free(entries);
    }
    free(entries1);
    free(entries2);
    free(entries3);
    free(entries4);

}

// USED JUST IN TESTING>>>>TO HAVE SOME IDEA ABOUT THE GAIN (optimized implementation is done by rpasca)
double* mock_pairwise_dist(double* points_matrix, double* dist_array, int n, int dim) {
    for (int i = 0; i < n ; ++i) {
        for (int j = 0; j < i; ++j) {
            int linear_idx = i * n + j;
            dist_array[linear_idx] = UnrolledEuclideanDistance(points_matrix + i * dim, points_matrix + j * dim, dim);
        }
    }
}

/**
 * Merge pairwise dist with Knn
 */
void KNN_1(double* points_matrix, double* dist_array, double* k_dist_index, int* k_neighborhood_index, int n, int k,
           int dim) {
    // the pairwise_dist array will be a n(n-1)/2 array representing an upper diagonal matrix

    int pre_comp = n * (n - 1) / 2;
    int* entries_indices = (int*) malloc((n - 1) * sizeof(int));
    double* entries_distances = (double*) malloc((n - 1) * sizeof(double));

    for (int i = 0; i < n; ++i) {

        int local_idx = 0;
        for (int j = i + 1; j < n; ++j) {
            int linear_idx = pre_comp - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
            double distance = UnrolledEuclideanDistance(points_matrix + i * dim, points_matrix + j * dim, dim);
            dist_array[linear_idx] = distance;
            entries_indices[local_idx] = j;
            entries_distances[local_idx] = distance;
            local_idx++;
        }

        for (int j = 0; j < i; j++) {
            int linear_idx = pre_comp - (n - j) * ((n - j) - 1) / 2 + i - j - 1;
            entries_indices[local_idx] = j;
            entries_distances[local_idx] = dist_array[linear_idx];
            local_idx++;
        }

        quickSortArray(entries_distances, entries_indices, 0, local_idx - 1);
//        qsort(entries, n - 1, sizeof(RECORD), compare_record);
        for (int l = 0; l < k; ++l) {
            k_dist_index[i * k + l] = entries_distances[l];
            k_neighborhood_index[i * k + l] = entries_indices[l];
        }
    }
}


/**
 * Old fashioned merge
 */
void KNN_2(double* points_matrix, double* dist_array, double* k_dist_index, int* k_neighborhood_index, int n, int k,
           int dim) {
    // the pairwise_dist array will be a n(n-1)/2 array representing an upper diagonal matrix

    int* entries_indices = (int*) malloc((n - 1) * sizeof(int));
    double* entries_distances = (double*) malloc((n - 1) * sizeof(double));

    for (int i = 0; i < n; ++i) {

        int local_idx = 0;
        for (int j = i + 1; j < n; ++j) {
            int index = n * i + j;
            double distance = UnrolledEuclideanDistance(points_matrix + i * dim, points_matrix + j * dim, dim);
            dist_array[index] = distance;
            entries_indices[local_idx] = j;
            entries_distances[local_idx] = distance;
            local_idx++;
        }

        for (int j = 0; j < i; j++) {
            int index = n * j + i;
            entries_indices[local_idx] = j;
            entries_distances[local_idx] = dist_array[index];
            local_idx++;
        }

        quickSortArray(entries_distances, entries_indices, 0, local_idx - 1);
        for (int l = 0; l < k; ++l) {
            k_dist_index[i * k + l] = entries_distances[l];
            k_neighborhood_index[i * k + l] = entries_indices[l];
        }
    }
}

/**
 * Classical Merge linear search indices
 */
void KNN_3(double* points_matrix, double* dist_array, double* k_dist_index, int* k_neighborhood_index, int n, int k,
           int dim) {
    // the pairwise_dist array will be a n(n-1)/2 array representing an upper diagonal matrix

    double* entries_distances = (double*) malloc((n - 1) * sizeof(double));

    for (int i = 0; i < n; ++i) {

        int local_idx = 0;
        for (int j = i + 1; j < n; ++j) {
            int index = n * i + j;
            double distance = UnrolledEuclideanDistance(points_matrix + i * dim, points_matrix + j * dim, dim);
            dist_array[index] = distance;
            entries_distances[local_idx] = distance;
            local_idx++;
        }

        for (int j = 0; j < i; j++) {
            int index = n * j + i;
            entries_distances[local_idx] = dist_array[index];
            local_idx++;
        }

        qsort(entries_distances, n - 1, sizeof(double), compare_double);
        for (int l = 0; l < k; ++l) {
            k_dist_index[i * k + l] = entries_distances[l];
        }

        int k_so_far = 0;
        for (int l = 0; l < i; l++) {
            int index = n * l + i;
            if (dist_array[index] <= entries_distances[k - 1]) {
                k_neighborhood_index[i * k + k_so_far] = l;
                k_so_far++;
            }
        }

        for (int l = i + 1; l < n; l++) {
            int index = n * i + l;
            if (dist_array[index] <= entries_distances[k - 1]) {
                k_neighborhood_index[i * k + k_so_far] = l;
                k_so_far++;
            }
        }
    }
}

/**
 * Linear Merge linear search indices
 */
void KNN_4(int n, int k, int dim, double* points_matrix, int* k_neighborhood_index, double* dist_array,
           double* k_dist_index) {
    // the pairwise_dist array will be a n(n-1)/2 array representing an upper diagonal matrix

    double* entries_distances = (double*) malloc((n - 1) * sizeof(double));
    int pre_comp = n * (n - 1) / 2;
    for (int i = 0; i < n; ++i) {

        int local_idx = 0;
        for (int j = i + 1; j < n; ++j) {
            int index = pre_comp - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
            double distance = UnrolledEuclideanDistance(points_matrix + i * dim, points_matrix + j * dim, dim);
            dist_array[index] = distance;
            entries_distances[local_idx] = distance;
            local_idx++;
        }

        for (int j = 0; j < i; j++) {
            int index = pre_comp - (n - j) * ((n - j) - 1) / 2 + i - j - 1;
            entries_distances[local_idx] = dist_array[index];
            local_idx++;
        }

        qsort(entries_distances, n - 1, sizeof(double), compare_double);
        for (int l = 0; l < k; ++l) {
            k_dist_index[i * k + l] = entries_distances[l];
        }

        int k_so_far = 0;
        for (int l = 0; l < i; l++) {
            int index = n * l + i;
            if (dist_array[index] <= entries_distances[k - 1]) {
                k_neighborhood_index[i * k + k_so_far] = l;
                k_so_far++;
            }
        }

        for (int l = i + 1; l < n; l++) {
            int index = n * i + l;
            if (dist_array[index] <= entries_distances[k - 1]) {
                k_neighborhood_index[i * k + k_so_far] = l;
                k_so_far++;
            }
        }
    }
}

/**
 * merge linear search index with hashing
 */
void
KNN_hash(int n, int k, int dim, double* points_matrix, int* k_neighborhood_index, int* hash_index, double* dist_array,
         double* k_dist_index) {

    double* entries_distances = (double*) malloc((n - 1) * sizeof(double));

    int pre_comp = n * (n - 1) / 2;

    for (int i = 0; i < n; ++i) {

        int local_idx = 0;
        for (int j = i + 1; j < n; ++j) {
            int index = pre_comp - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
            hash_index[i * n + j] = index;
            double distance = UnrolledEuclideanDistance(points_matrix + i * dim, points_matrix + j * dim, dim);
            dist_array[index] = distance;
            entries_distances[local_idx] = distance;
            local_idx++;
        }

        for (int j = 0; j < i; j++) {
            int index = pre_comp - (n - j) * ((n - j) - 1) / 2 + i - j - 1;
            hash_index[i * n + j] = index;
            entries_distances[local_idx] = dist_array[index];
            local_idx++;
        }

        qsort(entries_distances, n - 1, sizeof(double), compare_double);
//        for (int l = 0; l < k; ++l) {
//            k_dist_index[i * k + l] = entries_distances[l];
//        }

        k_dist_index[i] = entries_distances[k - 1];

        int k_so_far = 0;
        for (int l = 0; l < i; l++) {
            int index = hash_index[i * n + l];
            if (dist_array[index] <= entries_distances[k - 1]) {
                k_neighborhood_index[i * k + k_so_far] = l;
                k_so_far++;
            }
        }

        for (int l = i + 1; l < n; l++) {
            int index = hash_index[i * n + l];
            if (dist_array[index] <= entries_distances[k - 1]) {
                k_neighborhood_index[i * k + k_so_far] = l;
                k_so_far++;
            }
        }
    }
}

double test_knn_baseline(int num_pts, int k, int dim, double* dist_matrix, int* k_index_ptr, int* hash_index,
                         double* k_dist_ptr,
                         double* points_matrix) {
    ComputePairwiseDistances(dim, num_pts, points_matrix, UnrolledEuclideanDistance, dist_matrix);
    ComputeKDistanceAll(k, num_pts, dist_matrix, k_dist_ptr);
    ComputeKDistanceNeighborhoodAll(num_pts, k, k_dist_ptr, dist_matrix, k_index_ptr);
    return num_pts * num_pts / 2 * (4 * dim + 1) + num_pts * (num_pts - 1) * (1 + log(num_pts - 1)) +
           num_pts * (num_pts - 1);
}

long test_knn_records(double* points_matrix, double* dist_array, double* k_dist_ptr, int* k_index_ptr, int* hash_index,
                      int num_pts, int dim, int k) {
    mock_pairwise_dist(points_matrix, dist_array, num_pts, dim);
    KNN(num_pts, k, k_index_ptr, dist_array, k_dist_ptr);
    return num_pts * num_pts / 2 * (4 * dim + 1 + 12) + num_pts * (num_pts - 1) * (1 + log(num_pts - 1));
}

// some copy paste
long test_knn_arrays(double* points_matrix, double* dist_array, double* k_dist_ptr, int* k_index_ptr, int* hash_index,
                     int num_pts, int dim, int k) {
    mock_pairwise_dist(points_matrix, dist_array, num_pts, dim);
    double flops = KNN_fastest(num_pts, k, k_index_ptr, dist_array, k_dist_ptr);
    return flops + num_pts * num_pts / 2.0 * (4 * dim + 1 + 12);
}


// some copy paste
long test_knn_unrolled(double* points_matrix, double* dist_array, double* k_dist_ptr, int* k_index_ptr, int* hash_index,
                       int num_pts, int dim, int k) {
    mock_pairwise_dist(points_matrix, dist_array, num_pts, dim);
    KNN4x4(num_pts, k, k_index_ptr, dist_array, k_dist_ptr);
    return num_pts * num_pts / 2 * (4 * dim + 1 + 12) + num_pts * (num_pts - 1) * (1 + log(num_pts - 1));
}


long test_knn_merged(double* points_matrix, double* dist_array, double* k_dist_ptr, int* k_index_ptr, int* hash_index,
                     int num_pts, int dim, int k) {
    KNN_1(points_matrix, dist_array, k_dist_ptr, k_index_ptr, num_pts, k, dim);
    return num_pts * num_pts / 2 * (4 * dim + 1 + 12) + 2 * num_pts * (num_pts - 1) * (1 + log(num_pts - 1));
}


long test_knn_merged2(double* points_matrix, double* dist_array, double* k_dist_ptr, int* k_index_ptr, int* hash_index,
                      int num_pts, int dim, int k) {
    KNN_2(points_matrix, dist_array, k_dist_ptr, k_index_ptr, num_pts, k, dim);
    return num_pts * num_pts / 2 * (4 * dim + 1 + 2) + 2 * num_pts * (num_pts - 1) * (1 + log(num_pts - 1));

}


long test_knn_merged3(double* points_matrix, double* dist_array, double* k_dist_ptr, int* k_index_ptr, int* hash_index,
                      int num_pts, int dim, int k) {
    KNN_3(points_matrix, dist_array, k_dist_ptr, k_index_ptr, num_pts, k, dim);
    return num_pts * num_pts / 2 * (4 * dim + 1 + 2) + num_pts * (num_pts - 1) * (1 + log(num_pts - 1)) +
           num_pts * (num_pts - 1);
}

long test_knn_merged4(double* points_matrix, double* dist_array, double* k_dist_ptr, int* k_index_ptr, int* hash_index,
                      int num_pts, int dim, int k) {
    KNN_4(num_pts, k, dim, points_matrix, k_index_ptr, dist_array, k_dist_ptr);
    return num_pts * num_pts / 2 * (4 * dim + 1 + 12) + num_pts * (num_pts - 1) * (1 + log(num_pts - 1)) +
           num_pts * (num_pts - 1);
}


int test_results(double* dist_array, double* dist_matrix, int num_points, int k) {
    for (int i = 0; i < num_points; i++) {
        if (fabs(dist_array[i] - dist_matrix[i * k + k - 1]) > 1e-3) {
            printf("%lf != %lf at point %d \n", dist_array[i], dist_matrix[i * k + k - 1], i);
            return 0;
        }
    }
    return 1;
}

#define NUM_FUNCTIONS 2

void knn_driver(int num_pts, int k, int dim, int num_reps) {
    test_fun* fun_array = (test_fun*) calloc(NUM_FUNCTIONS, sizeof(test_fun));
//    fun_array[0] = &test_knn_unrolled;
    fun_array[0] = &test_knn_arrays;
    fun_array[1] = &test_knn_records;
//    fun_array[1] = &test_knn_merged2;
//    fun_array[4] = &test_knn_merged3;
//    fun_array[5] = &test_knn_merged4;
    char* fun_names[NUM_FUNCTIONS] = {"sort arrays", "sort records", "matrix merge",
                                      "matrix merge search", "linear merge search", "linear hash search"};
    myInt64 start, end;
    double cycles1;

    // INITIALIZE RANDOM INPUT
    double* points_matrix = XmallocMatrixDoubleRandom(num_pts, dim);
    double* dist_matrix = XmallocMatrixDouble(num_pts, num_pts);
    double* dist_array = XmallocVectorDouble(num_pts * num_pts);
    double* k_dist_array = XmallocVectorDouble(num_pts);
    double* k_dist_matrix = XmallocMatrixDouble(num_pts, k);
    int* neigh_index_matrix = XmallocMatrixInt(num_pts, k);
    int* k_neigh_index_matrix = XmallocMatrixInt(num_pts, k);
    int* hash_index = XmallocMatrixInt(num_pts, num_pts);

    long flops;

    for (int fun_index = 0; fun_index < NUM_FUNCTIONS; fun_index++) {
        test_knn_baseline(num_pts, k, dim, dist_matrix, neigh_index_matrix, hash_index, k_dist_array, points_matrix);
        (*fun_array[fun_index])(points_matrix, dist_array, k_dist_matrix, k_neigh_index_matrix, hash_index, num_pts,
                                dim, k);

        if (!test_results(k_dist_array, k_dist_matrix, num_pts, k)) {
            printf("BAD RESULTS.\n");
            for (int i = 0; i < num_pts; ++i) {
                printf("\n");
                for (int j = 0; j < k; ++j) {
                    int li, lj;
                    li = i < neigh_index_matrix[i * k + j] ? i : neigh_index_matrix[i * k + j];
                    lj = i > neigh_index_matrix[i * k + j] ? i : neigh_index_matrix[i * k + j];
                    double dist = dist_matrix[li * num_pts + lj];
                    printf("B(%d, %lf) ", neigh_index_matrix[i * k + j], dist);
                }
                printf("\n");
                for (int j = 0; j < k; ++j) {
                    int li = i;
                    int lj = k_neigh_index_matrix[i * k + j];
                    int linear_idx;
                    if (lj > li) {
                        linear_idx = lj * num_pts + li;
                    } else {
                        linear_idx = li * num_pts + lj;
                    }
                    printf("N(%d, %lf) ", k_neigh_index_matrix[i * k + j], dist_array[linear_idx]);
                }
                printf("\n");
            }
//            exit(0);
        }

        double multiplier = 1;
        double numRuns = 10;

        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[fun_index])(points_matrix, dist_array, k_dist_matrix, k_neigh_index_matrix, hash_index,
                                        num_pts, dim, k);
            }
            end = stop_tsc(start);

            cycles1 = (double) end;
            multiplier = (CYCLES_REQUIRED) / (cycles1);

        } while (multiplier > 2);

        double* cyclesPtr = XmallocVectorDouble(num_reps);

        CleanTheCache(500);

        for (size_t j = 0; j < num_reps; j++) {
            start = start_tsc();
            for (size_t i = 0; i < numRuns; ++i) {
                flops = (*fun_array[fun_index])(points_matrix, dist_array, k_dist_matrix, k_neigh_index_matrix,
                                                hash_index,
                                                num_pts, dim, k);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;
        }

        qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[((int) num_reps / 2) + 1];
        free(cyclesPtr);
        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        printf("NEW IMP: %s n:%d cycles:%lf perf:%lf \n", fun_names[fun_index], num_pts, cycles, perf);


        cyclesPtr = XmallocVectorDouble(num_reps);
        CleanTheCache(500);
        // MEASURE THE BASELINE AS WELL
        for (size_t j = 0; j < num_reps; j++) {

            start = start_tsc();
            for (size_t i = 0; i < numRuns; ++i) {
                flops = test_knn_baseline(num_pts, k, dim,
                                          dist_matrix, k_neigh_index_matrix, hash_index, k_dist_array, points_matrix);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;
        }

        qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
        double cycles_baseline = cyclesPtr[((int) num_reps / 2) + 1];

        double perf_baseline = round((1000.0 * flops) / cycles_baseline) / 1000.0;
        free(cyclesPtr);
        printf("BASELINE: %s n:%d cycles:%lf perf:%lf \n", "BASELINE", num_pts, cycles_baseline, perf_baseline);
    }
    printf("-------------\n");
    free(k_dist_matrix);
    free(k_dist_array);
    free(points_matrix);
    free(dist_array);
    free(dist_matrix);
    free(neigh_index_matrix);
    free(k_neigh_index_matrix);
}