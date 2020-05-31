//
// Created by pasca on 08.05.2020.
//

#ifndef FASTLOF_COMPUTEKDISTANCENEIGHBORHOOD_H
#define FASTLOF_COMPUTEKDISTANCENEIGHBORHOOD_H

#include "../../include/utils.h"

// Definition 4

// _____________________________IMPROVEMENTS_______________________________________
void ComputeKDistanceNeighborhoodAllInnerLoop(int num_pts, int k, const double* k_distances_indexed_ptr,
                                              const double* distances_indexed_ptr, int* neighborhood_index_table_ptr);


double ComputeKDistanceNeighborhoodAllFaster(int num_pts, int k, const double* k_distances_indexed_ptr,
                                           const double* distances_indexed_ptr, int* neighborhood_index_table_ptr);
//_______________________________________________________________________________________________________________



int Compute_K_Neighborhood_driver(int num_pts, int k, int dim, int num_reps);


#endif //FASTLOF_COMPUTEKDISTANCENEIGHBORHOOD_H
