//
// Created by fvasluia on 5/19/20.
//
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include "../../include/lattice.h"
#include "../../include/utils.h"
#include "../../include/file_utils.h"
#include "../include/AVX_utils.h"

int AVXComputeCellIndex(MULTI_LATTICE* lattice, double* point, int split_dims, int num_dims) {
    // switch to double arithmetic without expensive casts
    int i = 0;
    int resol = lattice->resolution;
    double resol3 = (double) resol * resol * resol;
    double resol2 = (double) resol * resol;
    double resol4 = (double) resol3 * resol;

    __m256d zeros = _mm256_setzero_pd();
    __m256d ones = _mm256_set1_pd(1.0);
    __m256d multiplier = _mm256_set1_pd(resol4);

    __m256d acc = _mm256_setzero_pd();
    __m256d factor = _mm256_set_pd(resol3, resol2, (double)resol, 1.0);

    int runs;
    printf("\n");
    for (i=0; (i + 3) < split_dims; i += 4) {
        ++runs;
        __m256d p_s = _mm256_loadu_pd(point + i);
        __m256d min_s = _mm256_loadu_pd(lattice->min_range + i);
        __m256d axis_s = _mm256_loadu_pd(lattice->axis_step + i);
        __m256d nom = _mm256_sub_pd(p_s, min_s);
        __m256d dbl_idx = _mm256_div_pd(nom, axis_s);
        __m256d rounded_idx = _mm256_ceil_pd(dbl_idx);
        __m256d cell_idx = _mm256_sub_pd(rounded_idx, ones);
        __m256d vmask = _mm256_cmp_pd(cell_idx, zeros, _CMP_GE_OQ);
        __m256d shift = _mm256_andnot_pd(vmask, ones);
        cell_idx = _mm256_add_pd(cell_idx, shift);
        acc = _mm256_fmadd_pd(cell_idx, factor, acc);
        factor = _mm256_mul_pd(factor, multiplier);
    }
    int mult = pow(resol, 4 * runs);
    double index = sum_double_avx(acc);
    for(;i < split_dims; ++i) {
        int dim_cell = ceil((point[i] - lattice->min_range[i])/lattice->axis_step[i]) - 1;
        if(dim_cell < 0) dim_cell = 0;
        index += mult * dim_cell;
        mult *= lattice->resolution;
    }
    return (int) index;
}

// some copy paste for faster input loading
int AVXComputeCellIndexWithOutput(MULTI_LATTICE* lattice, double* point, double* cell_locations, int split_dims, int num_dims) {
    int i = 0;
    int resol = lattice->resolution;
    double resol3 = (double) resol * resol * resol;
    double resol2 = (double) resol * resol;
    double resol4 = resol3 * resol;

    __m256d zeros = _mm256_setzero_pd();
    __m256d ones = _mm256_set1_pd(1.0);
    __m256d multiplier = _mm256_set1_pd(resol4);

    __m256d acc = _mm256_setzero_pd();
    __m256d factor = _mm256_set_pd(resol3, resol2, (double)resol, 1.0);

    int runs;
    for (i=0; (i + 3) < split_dims; i += 4) {
        ++runs;
        __m256d p_s = _mm256_loadu_pd(point + i);
        __m256d min_s = _mm256_loadu_pd(lattice->min_range + i);
        __m256d axis_s = _mm256_loadu_pd(lattice->axis_step + i);
        __m256d nom = _mm256_sub_pd(p_s, min_s);
        __m256d dbl_idx = _mm256_div_pd(nom, axis_s);
        __m256d rounded_idx = _mm256_ceil_pd(dbl_idx);
        __m256d cell_idx = _mm256_sub_pd(rounded_idx, ones);
        __m256d vmask = _mm256_cmp_pd(cell_idx, zeros, _CMP_GE_OQ);
        __m256d shift = _mm256_andnot_pd(vmask, ones);
        cell_idx = _mm256_add_pd(cell_idx, shift);
        //printf("%lf \n", sum_double_avx(cell_idx));

        _mm256_storeu_pd(cell_locations + i, cell_idx);

        acc = _mm256_fmadd_pd(cell_idx, factor, acc);
        factor = _mm256_mul_pd(factor, multiplier);
    }
    int mult = pow(resol, 4 * runs);
    double index = sum_double_avx(acc);
    for(;i < split_dims; ++i) {
        int dim_cell = ceil((point[i] - lattice->min_range[i])/lattice->axis_step[i]) - 1;
        if(dim_cell < 0) dim_cell = 0;
        cell_locations[i] = dim_cell;
        index += mult * dim_cell;
        mult *= lattice->resolution;
    }
    return (int) index;
}

void test_idx_comp(int num_pts, int k) {
    int resolution = 5;
    int dim = 128;
    int split_dim = 8;
    // just to test it
    char dir_full[50];
    sprintf(dir_full, "/n%d_k%d_dim%d/", num_pts, k, dim);
    char* input_file_name = concat(dir_full, "dataset.txt");
    char* meta_file_name = concat(dir_full, "metadata.txt");

    FILE* input_file = open_with_error_check(input_file_name, "r");
    FILE* meta_file = open_with_error_check(meta_file_name, "r");
    double* min_range;
    double* max_range;
    MULTI_LATTICE lattice;
    double* buffer = XmallocVectorDouble(split_dim);
    double* points = LoadTopologyInfo(input_file, meta_file, &lattice, min_range, max_range, resolution, &num_pts, &dim,
                                      &k, split_dim);
    for(int i = 0; i < num_pts; ++i) {
        int AVX_idx = AVXComputeCellIndexWithOutput(&lattice, points + i * dim, buffer, split_dim, dim);
        int idx = ComputeCellIndex(&lattice, points + i * dim, split_dim, dim);
        printf(" %d -> %d - %d\n", i, AVX_idx, idx);
    }
}
