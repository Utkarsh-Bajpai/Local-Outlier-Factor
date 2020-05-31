//
// Created by fvasluia on 5/8/20.
//
#include <stdlib.h>
#include <math.h>

#include "../include/ComputeTopologyInfo.h"
#include "../include/utils.h"
#include "../include/sort.h"
#include "../include/file_utils.h"
#include "../include/lof_baseline.h"

#include "../unrolled/include/metrics.h"
#include "../unrolled/include/ComputeKDistanceAll.h"
#include "../unrolled/include/Algorithm.h"
#include "../include/metrics.h"

// generally better to choose small prime numbers (avoids strided access)
# define RESOLUTION 3

int compare_record(const void* a, const void* b) {
    // this static cast has to be removed if a better sorting is found
    RECORD* A = (RECORD*) a;
    RECORD* B = (RECORD*) b;
    double aa = A->distance;
    double bb = B->distance;

    if (aa > bb)
        return 1;
    else if (aa < bb)
        return -1;
    else
        return 0;
}

void MergedBaseline();

long BASEComputeTopologyInfo(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                         double* k_dist_table_ptr, int k, int num_points, int split_dim,  int dim) {
    int max_idx = pow(lattice->resolution, split_dim);

    long flops = 0;

    for (int i = 0; i < num_points; ++i) {
        //int index = ComputeCellIndex(lattice, points_matrix_ptr + i * dim, split_dim, dim);

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

        candidate_list = (int*) malloc(5 * dim * 13 * sizeof(int));
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
            double dist =  EuclideanDistance(points_matrix_ptr + i * dim,
                                                     points_matrix_ptr + current_point_idx * dim,
                                                     dim);
            RECORD r = {current_point_idx, dist};
            distances[li] = r;

        }
        flops = candidate_list_sz * (4 * dim + 1);

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

long ComputeTopologyInfo(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                         double* k_dist_table_ptr, int k, int num_points, int split_dim,  int dim) {
    int max_idx = pow(lattice->resolution, split_dim);

    long flops = 0;


    for (int i = 0; i < num_points; ++i) {
        //int index = ComputeCellIndex(lattice, points_matrix_ptr + i * dim, split_dim, dim);

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

                flops += 8; // the computation of the shifted indices
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
            double dist =  UnrolledEuclideanDistance(points_matrix_ptr + i * dim,
                                                     points_matrix_ptr + current_point_idx * dim,
                                                     dim);
            RECORD r = {current_point_idx, dist};
            distances[li] = r;


        }
        flops += candidate_list_sz * (4 * dim + 1);

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

int gi = 0;
void generateAllNeighbors(int n, int* neigh_table, int* arr, int i, int shift_index)
{
    if (i == -1) {
        for(int li =0; li < n; li++) {
            neigh_table[gi * n + li] = arr[li];
        }
        ++gi;
        return;
    }

    arr[i] = 0;
    generateAllNeighbors(n, neigh_table, arr, i - 1, shift_index);

    arr[i] = 1;
    generateAllNeighbors(n, neigh_table, arr, i - 1, shift_index);
}

long ComputeTopologyInfoWithPermutations(MULTI_LATTICE* lattice, double* points_matrix_ptr, int* neigh_index_table_ptr,
                         double* k_dist_table_ptr, int k, int num_points, int split_dim,  int dim) {
    int max_idx = pow(lattice->resolution, split_dim);

    long flops = 0;


    for (int i = 0; i < num_points; ++i) {
        //int index = ComputeCellIndex(lattice, points_matrix_ptr + i * dim, split_dim, dim);

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

        candidate_list = (int*) malloc(32 * dim * 13 * sizeof(int)); // TODO get a proper initialization
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
        int* arr = XmallocMatrixInt(1, split_dim);
        int num_neigh = 1 << split_dim;
        int* neigh_table = XmallocMatrixInt(num_neigh, split_dim);

        while((candidate_list_sz < k) || (BFS == 0)) {
            gi = 0;
            generateAllNeighbors(split_dim, neigh_table, arr, split_dim - 1, shift_index);
            BFS = 1;

            for(int ni = 0; ni < num_neigh; ++ni) {
                int local_idx_dwn = index;
                int local_idx_up = index;
                factor = 1;
                int valid_up = 1, valid_dwn = 1;

                for(int nj = 0; nj < split_dim; ++nj) {
                    int local_displacement = neigh_table[ni * split_dim + nj];
                    //printf("%d ", local_displacement);
                    if((indices[nj] - local_displacement) >= 0) {
                        local_idx_dwn -= local_displacement * factor;
                    } else {
                        valid_dwn = 0;
                    }
                    if((indices[nj] + local_displacement) < lattice->resolution) {
                        local_idx_up += local_displacement * factor;
                    } else {
                        valid_up = 0;
                    }
                    factor *= lattice->resolution;
                }
                //printf("\n");

                flops += 8 * split_dim;

                if(valid_dwn && (local_idx_dwn != index)) {
                    candidate_list_sz += lattice->cell_Arr[local_idx_dwn].current_index;
                    for (int l = 0; l < lattice->cell_Arr[local_idx_dwn].current_index; l++) {
                        int neigh_id = lattice->cell_Arr[local_idx_dwn].point_ids[l];
                        if (neigh_id != i) {
                            //printf("DWN(%d from %d): %d - %d \n", ni, index, local_idx_dwn, neigh_id);
                            candidate_list[idx_candidate++] = neigh_id;
                        }
                    }
                }
                if(valid_up && (local_idx_up != index)) {
                    candidate_list_sz += lattice->cell_Arr[local_idx_up].current_index;
                    for (int l = 0; l < lattice->cell_Arr[local_idx_up].current_index; l++) {
                        int neigh_id = lattice->cell_Arr[local_idx_up].point_ids[l];
                        if (neigh_id != i) {
                            //printf("UP(%d from %d):%d - %d \n", ni, index,  local_idx_up, neigh_id);
                            candidate_list[idx_candidate++] = neigh_id;
                        }
                    }
                }

            }
            ++shift_index;
            if(shift_index == lattice->resolution) {
                break;
            }
        }


        free(indices);
        free(arr);
        free(neigh_table);


        RECORD* distances = (RECORD*) malloc(candidate_list_sz * sizeof(RECORD));
        if (distances == NULL) {
            fprintf(stderr, "NULL Allocation for RECORD array. Exiting..");
            return 0;
        }

        //printf("%d\n", i);
        for (int li = 0; li < candidate_list_sz; ++li) {
            int current_point_idx = candidate_list[li];
            //printf("\t%d \n", current_point_idx);
            double dist =  UnrolledEuclideanDistance(points_matrix_ptr + i * dim,
                                                     points_matrix_ptr + current_point_idx * dim,
                                                     dim);
            RECORD r = {current_point_idx, dist};
            distances[li] = r;


        }
        flops += candidate_list_sz * (4 * dim + 1);

        if (candidate_list_sz > k) {
            quickSort(distances, 0, candidate_list_sz - 1);
            for(int li = 0; li < candidate_list_sz; li++) {
                printf("(%d, %lf) ", distances[li].point_idx, distances[li].distance);
            }
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

int topology_info_driver(int num_pts, int k, int num_reps) {
    int resolution = 17;
    int dim = 16;
    int split_dim = 5;
    // just to test it
    char dir_full[50];
    sprintf(dir_full, "/n%d_k%d_dim%d/", num_pts, k, dim);
    char* input_file_name = concat(dir_full, "dataset.txt");
    char* meta_file_name = concat(dir_full, "metadata.txt");

    FILE* input_file = open_with_error_check(input_file_name, "r");
    FILE* meta_file = open_with_error_check(meta_file_name, "r");
    double* min_range;
    double* max_range;
    int* neigh_idx = XmallocMatrixInt(num_pts, k);
    double* neigh_dist = XmallocMatrixDouble(num_pts, k);

    int* perm_neigh_idx = XmallocMatrixInt(num_pts, k);
    double* perm_neigh_dist = XmallocMatrixDouble(num_pts, k);
    MULTI_LATTICE lattice;

    double* points = LoadTopologyInfo(input_file, meta_file, &lattice, min_range, max_range, resolution, &num_pts, &dim,
                                      &k, split_dim);
    ComputeTopologyInfo(&lattice, points, neigh_idx, neigh_dist, k, num_pts, split_dim, dim);
    ComputeTopologyInfoWithPermutations(&lattice, points, perm_neigh_idx, perm_neigh_dist, k, num_pts, split_dim, dim);

    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDoubleRandom(num_pts);

    ComputePairwiseDistances(dim, num_pts, points, UnrolledEuclideanDistance, distances_indexed_ptr);
    ComputeKDistanceAll(k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
    // ComputeKDistanceAllFaster

    printf("\n [");
    for (int i = 0; i < num_pts; ++i) {
        printf(" %lf,", k_distances_indexed_ptr[i * k + k - 1]);
    }
    printf("]");

    printf("\n [");
    for (int i = 0; i < num_pts; ++i) {
        printf(" %lf,", neigh_dist[i * k + k - 1]);
    }
    printf("]");

    printf("\n [");
    for (int i = 0; i < num_pts; ++i) {
        printf(" %lf,", perm_neigh_dist[i * k + k - 1]);
    }
    printf("]");
}

