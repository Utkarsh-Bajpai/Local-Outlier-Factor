//
// Created by pasca on 08.05.2020.
//


#ifndef FASTLOF_COMPUTEKDISTANCEALL_H
#define FASTLOF_COMPUTEKDISTANCEALL_H


//__________________________________________IMPROVEMENTS OBJECT________________________________________________________
void ComputeKDistanceObjectFaster(int obj_idx, int k, int num_pts, const double* distances_indexed_ptr,
                                  double* k_distances_indexed_ptr);

//_________________________________IMPROVEMENTS ALL____________________________________________________________________
double
ComputeKDistanceAllFaster(int k, int num_pts, const double* distances_indexed_ptr, double* k_distances_indexed_ptr);


#endif //FASTLOF_COMPUTEKDISTANCEALL_H
