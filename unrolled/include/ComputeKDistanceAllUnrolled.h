//
// Created by pasca on 08.05.2020.
//


#ifndef FASTLOF_COMPUTEKDISTANCEALLUNROLLED_H
#define FASTLOF_COMPUTEKDISTANCEALLUNROLLED_H

#include "../../include/utils.h"

// Definition 3

//__________________________________________IMPROVEMENTS________________________________________________________
void ComputeKDistanceObjectFaster(int obj_idx, int k, int num_pts, const double* distances_indexed_ptr,
                                  double* k_distances_indexed_ptr);


double
ComputeKDistanceAllUnroll_Fastest(int k, int num_pts, const double* distances_indexed_ptr, double* k_distances_indexed_ptr);
//_____________________________________________________________________________________________________


#endif //FASTLOF_COMPUTEKDISTANCEALLUNROLLED_H
