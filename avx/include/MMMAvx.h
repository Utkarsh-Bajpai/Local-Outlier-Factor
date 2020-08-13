//
// Created by pasca on 19.05.2020.
//

#ifndef FASTLOF_MMMAVX_H
#define FASTLOF_MMMAVX_H

typedef double (* my_mmm_dist_fnc) (int, int, int, int, int, const double*, double*);

void avx_mmm_driver(int n, int dim, int B0, int B1, int BK, int nr_reps, int fun_index);

double mmm_avx_fastest(int n, int dim, int B0, int B1, int BK, const double* input, double* result);

double avx_micro_2_4_4(int n, int dim, int B0, int B1, int BK, const double* input, double* result);

void avx_mmm_block_profiler(int n, int dim, int B0, int B1, int BK, int nr_reps);

#endif //FASTLOF_MMMAVX_H
