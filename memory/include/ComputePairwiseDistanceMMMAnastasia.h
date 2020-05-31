/**
 * Bloking of Pairwise Distance calculation
 *
 * Github repo with potentially relevant theory (did not read it yet)
 * https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0
 */

#ifndef FASTLOF_COMPUTEPAIRWISEDISTANCE_H
#define FASTLOF_COMPUTEPAIRWISEDISTANCEMMMAVX_H

struct MMM_INPUT {
    int num_pts;
    int dim;
    int Bp;
    int Bd;
};

struct MMM_MICRO_INPUT {
    int num_pts;
    int dim;
    int Bp;
    int Bd;
    int Bpr;
    int Bdr;
};

//_____________________________ BASELINE __________________________________

typedef void (* my_mmm)(int, int, const double*, double*);

void baseline_full(int num_pts, int dim, const double* input_points_ptr, double* result);
double mmm_baseline(int num_pts, int dim, const double* input_points_ptr, double* result);

//_____________________________ MINI  _____________________________________

typedef void (* my_mini_mmm)(int, int, int, int, const double*, double*);

void mini_mmm_full(int num_pts, int dim, int Bp, int Bd, const double* input_points_ptr, double* result);
void mini_mmm_lower_triangular(int num_pts, int dim, int Bp, int Bd, const double* input_points_ptr, double* result );

//_____________________________ MICRO _____________________________________

// STILL FAILS SOME TESTS FROM test_micro_mmm_triangular_driver !
void micro_mmm_lower_triangular(int num_pts, int dim, int Bp, int Bd, int Bpr, int Bdr, const double* input_points_ptr, double* result, int verbose);

//_____________________________ POSTPROCESSING ____________________________

int test_mini_mmm_lower_triangular_single(int num_pts, int dim, int Bp, int Bd, my_mini_mmm mini_fct, my_mmm mmm_fct );
void test_mini_mmm_lower_triangular_driver( my_mini_mmm mini_fct, my_mmm mmm_fct );

int test_micro_mmm_lower_triangular_single(int num_pts, int dim, int Bp, int Bd, int Bpr, int Bdr, int verbose);
void test_micro_mmm_triangular_driver();

void cache_block_selection( int num_pts, int dim, int num_reps );

#endif //FASTLOF_COMPUTEPAIRWISEDISTANCE_H
