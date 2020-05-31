//
// Created by pasca on 08.05.2020.
//


#ifndef FASTLOF_COMPUTEKDISTANCEALL_H
#define FASTLOF_COMPUTEKDISTANCEALL_H

#include "../../include/utils.h"

// Definition 3

//__________________________________________IMPROVEMENTS________________________________________________________
void ComputeKDistanceObjectFaster(int obj_idx, int k, int num_pts, const double* distances_indexed_ptr,
                                  double* k_distances_indexed_ptr);


double
ComputeKDistanceAllFaster(int k, int num_pts, const double* distances_indexed_ptr, double* k_distances_indexed_ptr);
//_____________________________________________________________________________________________________


#endif //FASTLOF_COMPUTEKDISTANCEALL_H
