//
// Created by fvasluia on 5/14/20.
//
#include <immintrin.h>
#include <math.h>
#include "../../include/file_utils.h"
#include "../../include/utils.h"
#include "../../include/metrics.h"
#include "../../include/tsc_x86.h"
#include "../include/AVX_utils.h"
#include "../include/AVXMetrics.h"

#define CYCLES_REQUIRED 2e8


typedef int (* test_fun)(double*, double*, int, int);

/*
double sum_double_avx(__m256d v) { // claimed to be faster
    __m128d low  = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low  = _mm_add_pd(low, high);

    __m128d high64 = _mm_unpackhi_pd(low, low);
    return  _mm_cvtsd_f64(_mm_add_sd(low, high64));
}*/

double AVXEuclideanDistance(const double v1_ptr[], const double v2_ptr[], int n_dim)  {

    double dist = 0.0;
    if(n_dim < 4) {
        for(int i = 0; i < n_dim; ++i) {
            dist += (v1_ptr[i] - v2_ptr[i]) * (v1_ptr[i] - v2_ptr[i]);
        }
    } else {
        int i;
        __m256d acc = _mm256_setzero_pd();
        for(i = 0; (i + 3) < n_dim; i += 4) {
            __m256d v1_i = _mm256_loadu_pd(v1_ptr + i);
            __m256d v2_i = _mm256_loadu_pd(v2_ptr + i);
            __m256d diff = _mm256_sub_pd(v1_i, v2_i);
            //__m256d sq_diff = _mm256_mul_pd(diff, diff);
            acc = _mm256_fmadd_pd(diff, diff, acc);
        }
        dist = sum_double_avx(acc);
        // take care of the rest
        for(; i < n_dim; ++i) {
            dist += (v1_ptr[i] - v2_ptr[i]) * (v1_ptr[i] - v2_ptr[i]);
        }
    }
    return sqrt(dist);
}


double AVXCosineSimilarity(const double *v1_ptr, const double *v2_ptr, int n_dim) {
    double dot_prod = 0;
    double denom = 0;
    double norm_v1_s = 0, norm_v2_s = 0;
    if(n_dim < 4) {
        for(int i = 0; i < n_dim; ++i) {
            dot_prod += v1_ptr[i] * v2_ptr[i];
            norm_v1_s += v1_ptr[i] * v1_ptr[i];
            norm_v2_s += v2_ptr[i] * v2_ptr[i];
        }
        return dot_prod/sqrt(norm_v1_s * norm_v2_s);
    } else {
        int i;
        __m256d dot_acc = _mm256_setzero_pd();
        __m256d v1_nrm_acc = _mm256_setzero_pd();
        __m256d v2_nrm_acc = _mm256_setzero_pd();
        for(i = 0; (i + 3) < n_dim; i += 4) {
            __m256d v1_i = _mm256_loadu_pd(v1_ptr + i);
            __m256d v2_i = _mm256_loadu_pd(v2_ptr + i);
            dot_acc = _mm256_fmadd_pd(v1_i, v2_i, dot_acc);
            v1_nrm_acc = _mm256_fmadd_pd(v1_i, v1_i, v1_nrm_acc);
            v2_nrm_acc = _mm256_fmadd_pd(v2_i, v2_i, v2_nrm_acc);

        }
        dot_prod = sum_double_avx(dot_acc);
        norm_v1_s = sum_double_avx(v1_nrm_acc);
        norm_v2_s = sum_double_avx(v2_nrm_acc);
        // take care of the rest
        for(; i < n_dim; ++i) {
            dot_prod += v1_ptr[i] * v2_ptr[i];
            norm_v1_s += v1_ptr[i] * v1_ptr[i];
            norm_v2_s += v2_ptr[i] * v2_ptr[i];
        }
    }
    return dot_prod/sqrt(norm_v1_s * norm_v2_s);

}

int unrolled_euclid_test(double* points, double* dists, int n, int dim) {
    for(int i = 0; i < n; ++i) {
        for(int j = i + 1; j < n; ++j) {
            dists[i * n + j] = UnrolledEuclideanDistance(points + i * dim, points + j * dim, dim);
        }
    }
    return n * n / 2 * (4 * dim + 1);
}

int avx_euclid_test(double* points, double* dists, int n, int dim) {
    for(int i = 0; i < n; ++i) {
        for(int j = i + 1; j < n; ++j) {
            dists[i * n + j] = AVXEuclideanDistance(points + i * dim, points + j * dim, dim);
        }
    }
    return n * n / 2 * (4 * dim + 1);
}

int unrolled_cosine_test(double* points, double* dists, int n, int dim) {
    for(int i = 0; i < n; ++i) {
        for(int j = i + 1; j < n; ++j) {
            dists[i * n + j] = CosineSimilarityFaster1(points + i * dim, points + j * dim, dim);
        }
    }
    return n * n / 2 * (4 * dim + 1);
}

int avx_cosine_test(double* points, double* dists, int n, int dim) {
    for(int i = 0; i < n; ++i) {
        for(int j = i + 1; j < n; ++j) {
            dists[i * n + j] = AVXCosineSimilarity(points + i * dim, points + j * dim, dim);
        }
    }
    return n * n / 2 * (4 * dim + 1);
}

void metrics_testbench(int num_pts,int dim, int num_reps) {

//    double* v1 = XmallocVectorDouble( dim);
//    double* v2 = XmallocVectorDouble( dim);
//    for(int i=0; i < dim; ++i) {
//        v1[i] = 1;
//        v2[2] = 0;
//    }
//
//    printf("%lf - > %lf \n", AVXEuclideanDistance(v1, v2, 24), UnrolledEuclideanDistance(v1, v2, 24));
    double *points = XmallocMatrixDoubleRandom(num_pts, dim);
    double *dist_base = XmallocMatrixDouble(num_pts, dim);
    double *dist_avx = XmallocMatrixDouble(num_pts, dim);

    unrolled_cosine_test(points, dist_base, num_pts, dim);
    avx_cosine_test(points, dist_avx, num_pts, dim);
    for(int i = 0; i< num_pts; i++) {
        for(int j = i + 1; j < num_pts; j++) {
            printf("%lf == %lf \n", dist_base[i * num_pts + j], dist_avx[i * num_pts + j]);
        }
    }
    test_fun* fun_array = (test_fun*) calloc(4, sizeof(test_fun));
    fun_array[0] = &unrolled_euclid_test;
    fun_array[1] = &avx_euclid_test;
    fun_array[2] = &unrolled_cosine_test;
    fun_array[3] = &avx_cosine_test;

    char* names[4] = {"unrolled_euclid", "avx_euclid", "unrolled_cosine", "avx_cosine"};


    myInt64 start, end;
    double cycles1;

    for(int k = 0; k < 4; ++k) {
        double multiplier = 1;
        double numRuns = 10;

        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[k])(points, dist_base, num_pts, dim);
            }
            end = stop_tsc(start);

            cycles1 = (double) end;
            multiplier = (CYCLES_REQUIRED) / (cycles1);

        } while (multiplier > 2);

        double* cyclesPtr = XmallocVectorDouble(num_reps);

        CleanTheCache(500);
        int flops;
        for (size_t j = 0; j < num_reps; j++) {
            start = start_tsc();
            for (size_t i = 0; i < numRuns; ++i) {
                flops = (*fun_array[k])(points, dist_base, num_pts, dim);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;
        }

        qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[((int) num_reps / 2) + 1];
        free(cyclesPtr);
        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        printf("%s - %lf\n", names[k], perf);
    }
}

