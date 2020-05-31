//
// Created by pasca on 12.05.2020.
// TODO: find better location for distance computation ...

#include <stdlib.h>
#include "../../include/utils.h"
#include "../../unrolled/include/metrics.h"

void
KNN_hash(int n, int k, int dim, double* points_matrix,
         int* k_neighborhood_index, int* hash_index, double* dist_array,
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
