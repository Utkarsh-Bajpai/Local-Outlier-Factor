//
// Created by pasca on 18.05.2020.
//

#ifndef FASTLOF_MMM_H
#define FASTLOF_MMM_H


/**
 * Bloking of Pairwise Distance calculation
 *
 * Github repo with potentially relevant theory (did not read it yet)
 * https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0
 */


//_____________________________ BASELINE __________________________________

typedef void (* my_mmm)(int, int, const double*, double*);

int baseline_full(int n, int dim, int Bp, int Bd, const double* input_points_ptr, double* result);

double mmm_baseline(int n, int dim, const double* input_points_ptr, double* result);

//_____________________________ MINI  _____________________________________

typedef void (* my_mini_mmm)(int, int, int, int, const double*, double*);

int mini_mmm_full(int n, int dim, int Bp, int Bd, const double* input_points_ptr, double* result);

long mini_mm_1_block(int n, int dim, int B0, int B1, const double* input, double* result);

long mini_mm_2_blocks(int n, int dim, int B0, int B1, const double* input, double* result);

//_____________________________ MICRO _____________________________________

long micro_mm_2_blocks(int n, int dim, int B0, int B1, const double* input, double* result);

double mmm_unroll_fastest(int n, int dim, int B0, int B1, int BK, const double* input, double* result);


//_____________________________ POSTPROCESSING ____________________________

int test_mini_mmm_lower_triangular_single(int num_pts, int dim, int Bp, int Bd, my_mini_mmm mini_fct, my_mmm mmm_fct);

void mmm_driver(int n, int dim, int B0, int B1, int nr_reps);


#endif //FASTLOF_MMM_H
