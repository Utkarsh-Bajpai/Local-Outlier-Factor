//
// Created by pasca on 12.05.2020.
//

#ifndef FASTLOF_KNN_H
#define FASTLOF_KNN_H

void
KNN_hash(int n, int k, int dim, double* points_matrix, int* k_neighborhood_index, int* hash_index, double* dist_array,
         double* k_dist_index);

#endif //FASTLOF_KNN_H
