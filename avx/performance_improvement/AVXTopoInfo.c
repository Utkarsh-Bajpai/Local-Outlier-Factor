//
// Created by fvasluia on 5/20/20.
//

#include <immintrin.h>
#include <math.h>
#include "../include/AVX_utils.h"
#include "../include/AVXMetrics.h"
#include "../include/AVXTopoInfo.h"
#include "../../include/file_utils.h"
#include "../../include/utils.h"
#include "../../include/sort.h"
#include "../../include/tsc_x86.h"
#include "../../include/performance_measurement.h"

typedef long (* test_fun)(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                         double* neighborhood_distance_table_ptr, int k, int num_points, int split_dim, int dim);

# define WIDTH 1

long AVXDistanceComputeTopologyInfo(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                         double* k_dist_table_ptr, int k, int num_points, int split_dim,  int dim) {
    int max_idx = pow(lattice->resolution, split_dim);


    long flops = 0;

    for (int i = 0; i < num_points; ++i) {
        // keep the indices of the cell for a proper BFS
        int* indices = XmallocMatrixInt(1, split_dim); //same as Vector
        int multiplier = 1;
        int index = 0;

        for(int  li = 0; li < split_dim; ++li) {
            int dim_cell = ceil((points_matrix_ptr[i * dim + li] - lattice->min_range[li])/lattice->axis_step[li]) - 1;
            if(dim_cell < 0)
                dim_cell = 0;
            indices[li] = dim_cell;
            index += multiplier * dim_cell;
            multiplier *= lattice->resolution;
        }
        flops += 6 * split_dim;

        int candidate_list_sz = 0;
        int* candidate_list = NULL;

        candidate_list = (int*) malloc(5 * dim * 13 * sizeof(int)); // TODO get a proper initialization
        if (candidate_list == NULL) {
            fprintf(stderr, "Bad Alloc in candidate list for extended case");
        }

        // update with the set of neighboring indices in the neighboring cell of the current point
        int factor = 1;
        int idx_candidate = 0;

        for (int l = 0; l < lattice->cell_Arr[index].current_index; ++l) {
            if (lattice->cell_Arr[index].point_ids[l] != i) {
                candidate_list[idx_candidate++] = lattice->cell_Arr[index].point_ids[l];
            }
        }
        candidate_list_sz += lattice->cell_Arr[index].current_index - 1;


        int shift_index = 1;
        // now try to generalize on the neighboring cells
        int BFS = 0;
        while((candidate_list_sz < k) || (!BFS)) {
            BFS = 1;
            factor = 1;
            for (int l_dim = 0; l_dim < split_dim; ++l_dim) {
                if ((indices[l_dim] - shift_index) >= 0) {
                    int neigh_cel_dwn = index - shift_index * factor;
                    candidate_list_sz += lattice->cell_Arr[neigh_cel_dwn].current_index;
                    for (int l = 0; l < lattice->cell_Arr[neigh_cel_dwn].current_index; l++) {
                        candidate_list[idx_candidate++] = lattice->cell_Arr[neigh_cel_dwn].point_ids[l];
                    }
                }
                if ((indices[l_dim] + shift_index) < lattice->resolution){
                    int neigh_cel_up = index + shift_index * factor;
                    candidate_list_sz += lattice->cell_Arr[neigh_cel_up].current_index;
                    for (int l = 0; l < lattice->cell_Arr[neigh_cel_up].current_index; l++) {
                        candidate_list[idx_candidate++] = lattice->cell_Arr[neigh_cel_up].point_ids[l];
                    }
                }
                factor *= lattice->resolution;
                flops += 8;
            }
            ++shift_index;
            if(shift_index == lattice->resolution) {
                break;
            }
        }


        free(indices);


        RECORD* distances = (RECORD*) malloc(candidate_list_sz * sizeof(RECORD));
        if (distances == NULL) {
            fprintf(stderr, "NULL Allocation for RECORD array. Exiting..");
            return 0;
        }


        for (int li = 0; li < candidate_list_sz; ++li) {
            int current_point_idx = candidate_list[li];
            double dist =  AVXEuclideanDistance(points_matrix_ptr + i * dim,
                                                     points_matrix_ptr + current_point_idx * dim,
                                                     dim);
            RECORD r = {current_point_idx, dist};
            distances[li] = r;

            flops += 4 * dim + 1;
        }

        if (candidate_list_sz > k) {
            quickSort(distances, 0, candidate_list_sz - 1);
            flops += candidate_list_sz * log(candidate_list_sz);
        } else {
            free(distances);
            distances = (RECORD*) malloc(num_points * sizeof(RECORD));
            int neigh_idx = 0;
            for(int j = 0; j < num_points; ++j) {
                if( j != i) {
                    RECORD r = {j, UnrolledEuclideanDistance(
                            points_matrix_ptr + i * dim,
                            points_matrix_ptr + j * dim,
                            dim
                    )
                    };
                    distances[neigh_idx++] = r;
                }
            }
            flops += (num_points - 1) * (4 * dim + 1);
            quickSort(distances, 0, neigh_idx - 1);
            flops += (num_points - 1) * log(num_points - 1);
        }

        for (int p = 0; p < k; ++p) {
            k_dist_table_ptr[i * k + p] = distances[p].distance;
            neigh_index_table_ptr[i * k + p] = distances[p].point_idx;
        }

        free(candidate_list);
        free(distances);
    }
    return flops;
}

long AVXTopologyInfo(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                     double* neighborhood_distance_table_ptr, int k, int num_points, int split_dim, int dim) {

    long flops = 0;
    double res3 = (double) lattice->resolution * lattice->resolution * lattice -> resolution;
    double res2 = (double) res3/lattice->resolution;
    double res4 = res3 * lattice ->resolution;

    __m256d hor_incr = _mm256_set1_pd(res4);

    double *avx_indices = XmallocMatrixDouble(1, split_dim);
    __m256d max_idx = _mm256_set1_pd((double) (lattice->resolution));
    __m256d zeros = _mm256_setzero_pd();
    __m256d ones = _mm256_set1_pd(1.0);
    for (int i = 0; i < num_points; ++i) {
        double* candidate_cells = XmallocVectorDouble(256); //take care here

        int avx_index = AVXComputeCellIndexWithOutput(lattice, points_matrix_ptr + (i * dim), avx_indices, split_dim, dim);
        flops += 6 * dim;

        __m256d local_idx = _mm256_set1_pd((double) avx_index); //given the low nr of split dims -> go to dbl arithmetic
        int BFS = 0;

        __m256d shift = _mm256_set1_pd(1.0);
        int shift_index = 1;
        int nr_cells = 0;
        while (shift_index <= WIDTH || !BFS) {
            BFS = 1;
            __m256d factor = _mm256_set_pd(res3, res2, (double) lattice->resolution, 1.0);
            int j;
            for (j = 0; (j + 3) < split_dim; j += 4) {
                __m256d cell_idx = _mm256_loadu_pd(avx_indices + j);
                __m256d res_dwn = _mm256_sub_pd(cell_idx, shift);
                __m256d res_up = _mm256_add_pd(cell_idx, shift);

                __m256d mask_dwn = _mm256_cmp_pd(res_dwn, zeros, _CMP_GE_OQ);
                __m256d mask_up = _mm256_cmp_pd(res_up, max_idx, _CMP_LT_OQ);


                __m256d increment = _mm256_and_pd(factor, mask_up);
                __m256d decrement = _mm256_and_pd(factor, mask_dwn);

                __m256d cells_up = _mm256_add_pd(local_idx, increment);
                __m256d cells_dwn = _mm256_sub_pd(local_idx, decrement);

                _mm256_storeu_pd(candidate_cells + nr_cells, cells_dwn);
                _mm256_storeu_pd(candidate_cells + nr_cells + 4, cells_up);

                factor = _mm256_mul_pd(factor, hor_incr);
                nr_cells += 8;
            }
            flops += 9 * split_dim;

            shift = _mm256_add_pd(shift, ones);
            ++shift_index;
            if (shift_index == lattice->resolution - 1) {
                break;
            }
            flops += 4;
        }

        RECORD* candidates = (RECORD*) malloc(num_points * sizeof(RECORD));
        if(candidates == NULL) {
            fprintf(stderr, "NULL allocation for records..");
            exit(13);
        }

        int neigh_idx = 0;

        for(int lk = 0; lk < lattice->cell_Arr[avx_index].current_index; lk++) {
            int point_idx = lattice->cell_Arr[avx_index].point_ids[lk];
            if (point_idx != i) {
                RECORD r = {point_idx, AVXEuclideanDistance(
                        points_matrix_ptr + i * dim,
                        points_matrix_ptr + point_idx * dim,
                        dim)
                };
                candidates[neigh_idx++] = r;
            }
        }


        for(int lk = 0; lk < nr_cells; ++lk) {
            int cell_idx = (int)candidate_cells[lk];
            if(cell_idx != avx_index) {
                for(int lk1=0; lk1 < lattice->cell_Arr[cell_idx].current_index; ++lk1) {
                    int point_idx = lattice->cell_Arr[cell_idx].point_ids[lk1];
                    RECORD r = {point_idx, AVXEuclideanDistance(
                                points_matrix_ptr + i * dim,
                                points_matrix_ptr + point_idx * dim,
                                dim
                                )
                    };
                    candidates[neigh_idx++] = r;
                }
            }
        }
        flops += neigh_idx * (4 * dim + 1);

        if (neigh_idx > k) {
            quickSort(candidates, 0, neigh_idx - 1);
            flops += neigh_idx * log(neigh_idx);
        } else {
            free(candidates);
            candidates = (RECORD*) malloc(num_points * sizeof(RECORD));
            int neigh_idx = 0;
            for(int j = 0; j < num_points; ++j) {
                if( j != i) {
                    RECORD r = {j, UnrolledEuclideanDistance(
                            points_matrix_ptr + i * dim,
                            points_matrix_ptr + j * dim,
                            dim
                    )
                    };
                    candidates[neigh_idx++] = r;
                }
            }
            flops += (num_points - 1) * (4 * dim + 1);
            quickSort(candidates, 0, neigh_idx - 1);
            flops += (num_points - 1) * log(num_points - 1);
        }


        for (int li = 0; li < k; ++li) {
            neigh_index_table_ptr[i * k + li] = candidates[li].point_idx;
            neighborhood_distance_table_ptr[i * k + li] = candidates[li].distance;
        }

        free(candidates);
    }
    return flops;
}

void test_avx_topo_info(int num_pts, int k) {
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

    double* points = LoadTopologyInfo(input_file, meta_file, &lattice, min_range, max_range, resolution, &num_pts, &dim,
                                      &k, split_dim);
    int* k_neigh_index = XmallocMatrixInt(num_pts, k);
    double* k_dist_table = XmallocMatrixDouble(num_pts, k);

    int* avx_k_neigh_index = XmallocMatrixInt(num_pts, k);
    double* avx_k_dist_table = XmallocMatrixDouble(num_pts, k);

    ComputeTopologyInfo(&lattice, points, k_neigh_index, k_dist_table, k, num_pts, split_dim, dim);
    AVXTopologyInfo(&lattice, points, avx_k_neigh_index, avx_k_dist_table, k, num_pts, split_dim, dim);

    for(int i = 0; i < num_pts; ++i) {
        for(int j = 0; j < k; ++j ) {
            printf(" (%d, %lf) - (%d, %lf) ", k_neigh_index[i * k + j], k_dist_table[ i*k + j],
                   avx_k_neigh_index[i * k + j], avx_k_dist_table[ i*k + j]);
        }
        printf("\n");
    }
}

void avx_topology_info_driver(int num_pts, int k, int dim, int num_reps) {
    int resolution = 3;
    int split_dim = 8; //don't be crazu about it
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

    double* points = LoadTopologyInfo(input_file, meta_file, &lattice, min_range, max_range, resolution, &num_pts, &dim,
                                      &k, split_dim);
    int* k_neigh_index = XmallocMatrixInt(num_pts, k);
    double* k_dist_table = XmallocMatrixDouble(num_pts, k);

    test_fun* fun_array = (test_fun*) calloc(4, sizeof(test_fun));
    fun_array[0] = &ComputeTopologyInfo;
    fun_array[1] = &BASEComputeTopologyInfo;
    fun_array[2] = &AVXTopologyInfo;
    fun_array[3] = &AVXDistanceComputeTopologyInfo;

    char* names[4] = {"topo_unrolled", "base-topo", "topo_avx", "topo-avx-dist"};


    myInt64 start, end;
    double cycles1;

    for(int k = 0; k < 4; ++k) {
        double multiplier = 1;
        double numRuns = 10;

        do {
            numRuns = numRuns * multiplier;
            start = start_tsc();
            for (size_t i = 0; i < numRuns; i++) {
                (*fun_array[k])(&lattice, points, k_neigh_index, k_dist_table, k, num_pts, split_dim, dim);
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
                 flops = (*fun_array[k])(&lattice, points, k_neigh_index, k_dist_table, k, num_pts, split_dim, dim);
            }
            end = stop_tsc(start);

            cycles1 = ((double) end) / numRuns;
            cyclesPtr[j] = cycles1;
        }

        // flops = num_pts * num_pts/2 * (4 * dim - 1) + num_pts * (num_pts - 1) * log(num_pts - 1) + num_pts * (num_pts -1);
        qsort(cyclesPtr, num_reps, sizeof(double), compare_double);
        double cycles = cyclesPtr[((int) num_reps / 2) + 1];
        free(cyclesPtr);
        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        printf("%s - %lf - %d - %lf\n", names[k], cycles, flops, perf);
    }

}