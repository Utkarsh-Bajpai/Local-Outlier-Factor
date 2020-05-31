//
// Created by Utkarsh Bajpai on 08.05.20.
//

#include <stdlib.h>
#include <math.h>

#include "../include/Algorithm.h"
#include "../../include/lof_baseline.h"
#include "../../include/tsc_x86.h"
#include "../../include/tests.h"

typedef void (* my_fun)(int, int, const double*, const double*, int*);

#define CYCLES_REQUIRED 1e8
#define NUM_FUNCTIONS 4


void
ComputeKDistanceNeighborhood1(int num_pts, int k, const double* k_distances_indexed_ptr,
                              const double* distances_indexed_ptr, int* neighborhood_index_table_ptr) {
    /** Implement Definition 4 from the paper
    *
    * @param obj_idx: index of an object, with respect to which the distance is computed
    * @param n: number of points in the dataset
    *
    * @return: fill the elements of the table table neighborhood_index_table_ptr, corresponding to obj_idx
    *          s.t. the k entries to the i'th line correspond to the k nearest neighbors i.e the points with
     *          distance <= kdist
    * @note:
   */

    int idx, i;
    for (idx = 0; idx < num_pts; idx++) {
        double kdist = k_distances_indexed_ptr[idx];
        int k_so_far = 0;

        for (i = 0; i < idx; i++) {
            if (distances_indexed_ptr[num_pts * i + idx] <= kdist) {

                neighborhood_index_table_ptr[idx * k + k_so_far] = i;
                k_so_far++;
            }
        }

        for (i = idx + 1; i < num_pts; i++) {
            if (distances_indexed_ptr[idx * num_pts + i] <= kdist) {

                neighborhood_index_table_ptr[idx * k + k_so_far] = i;
                k_so_far++;
            }
        }

    }

}


void ComputeKDistanceNeighborhoodAllInnerLoop(int num_pts, int k, const double* k_distances_indexed_ptr,
                                              const double* distances_indexed_ptr, int* neighborhood_index_table_ptr) {
    /** Implement Definition 4 from the paper
    *
    * @param obj_idx: index of an object, with respect to which the distance is computed
    * @param n: number of points in the dataset
    *
    * @return: fill the elements of the table table neighborhood_index_table_ptr, corresponding to obj_idx
    *          s.t. the k entries to the i'th line correspond to the k nearest neighbors i.e the points with
     *          distance <= kdist
    * @note:
   */

    int idx, i;
    for (idx = 0; idx < num_pts; idx++) {
        double kdist = k_distances_indexed_ptr[idx];
        int k_so_far = 0;

        int idxk = idx * k;

        for (i = 0; (i + 4) < idx; i += 5) {
            if (distances_indexed_ptr[num_pts * i + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 1) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 2) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 3) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 4) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = (i + 4);
                k_so_far++;
            }
        }

        for (; i < idx; i++) {
            if (distances_indexed_ptr[num_pts * i + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        int idx_num_pts = idx * num_pts;

        for (i = idx + 1; (i + 4) < num_pts; i += 5) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 1] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 2] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 3] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 4] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 4;
                k_so_far++;
            }
        }

        for (; i < num_pts; i++) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

    }

}


double ComputeKDistanceNeighborhoodAllFaster(int num_pts, int k, const double* k_distances_indexed_ptr,
                                             const double* distances_indexed_ptr, int* neighborhood_index_table_ptr) {
    /** Implement Definition 4 from the paper
    *
    * @param obj_idx: index of an object, with respect to which the distance is computed
    * @param n: number of points in the dataset
    *
    * @return: fill the elements of the table table neighborhood_index_table_ptr, corresponding to obj_idx
    *          s.t. the k entries to the i'th line correspond to the k nearest neighbors i.e the points with
     *          distance <= kdist
    * @note:
   */

    int idx, i;
    for (idx = 0; idx < num_pts; idx += 5) {
        double kdist = k_distances_indexed_ptr[idx];
        int k_so_far = 0;

        int idxk = idx * k;

        for (i = 0; (i + 4) < idx; i += 5) {
            if (distances_indexed_ptr[num_pts * i + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 1) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 2) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 3) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 4) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = (i + 4);
                k_so_far++;
            }
        }

        for (; i < idx; i++) {
            if (distances_indexed_ptr[num_pts * i + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        int idx_num_pts = idx * num_pts;

        for (i = idx + 1; (i + 4) < num_pts; i += 5) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 1] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 2] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 3] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 4] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 4;
                k_so_far++;
            }
        }

        for (; i < num_pts; i++) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }


        //****OUTER LOOP UNROLL 2***
        kdist = k_distances_indexed_ptr[(idx + 1)];
        k_so_far = 0;

        idxk = (idx + 1) * k;

        for (i = 0; (i + 4) < (idx + 1); i += 5) {
            if (distances_indexed_ptr[num_pts * i + (idx + 1)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 1) + (idx + 1)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 2) + (idx + 1)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 3) + (idx + 1)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 4) + (idx + 1)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = (i + 4);
                k_so_far++;
            }
        }

        for (; i < (idx + 1); i++) {
            if (distances_indexed_ptr[num_pts * i + (idx + 1)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        idx_num_pts = (idx + 1) * num_pts;

        for (i = (idx + 1) + 1; (i + 4) < num_pts; i += 5) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 1] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 2] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 3] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 4] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 4;
                k_so_far++;
            }
        }

        for (; i < num_pts; i++) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        //****OUTER LOOP UNROLL 3***
        kdist = k_distances_indexed_ptr[(idx + 2)];
        k_so_far = 0;

        idxk = (idx + 2) * k;

        for (i = 0; (i + 4) < idx; i += 5) {
            if (distances_indexed_ptr[num_pts * i + (idx + 2)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 1) + (idx + 2)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 2) + (idx + 2)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 3) + (idx + 2)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 4) + (idx + 2)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = (i + 4);
                k_so_far++;
            }
        }

        for (; i < (idx + 2); i++) {
            if (distances_indexed_ptr[num_pts * i + (idx + 2)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        idx_num_pts = (idx + 2) * num_pts;

        for (i = (idx + 2) + 1; (i + 4) < num_pts; i += 5) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 1] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 2] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 3] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 4] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 4;
                k_so_far++;
            }
        }

        for (; i < num_pts; i++) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        //****OUTER LOOP UNROLL 4***
        kdist = k_distances_indexed_ptr[(idx + 3)];
        k_so_far = 0;

        idxk = (idx + 3) * k;

        for (i = 0; (i + 4) < (idx + 3); i += 5) {
            if (distances_indexed_ptr[num_pts * i + (idx + 3)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 1) + (idx + 3)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 2) + (idx + 3)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 3) + (idx + 3)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 4) + (idx + 3)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = (i + 4);
                k_so_far++;
            }
        }

        for (; i < (idx + 3); i++) {
            if (distances_indexed_ptr[num_pts * i + (idx + 3)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        idx_num_pts = (idx + 3) * num_pts;

        for (i = (idx + 3) + 1; (i + 4) < num_pts; i += 5) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 1] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 2] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 3] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 4] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 4;
                k_so_far++;
            }
        }

        for (; i < num_pts; i++) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        //****OUTER LOOP UNROLL 5***
        kdist = k_distances_indexed_ptr[(idx + 4)];
        k_so_far = 0;

        idxk = (idx + 4) * k;

        for (i = 0; (i + 4) < (idx + 4); i += 5) {
            if (distances_indexed_ptr[num_pts * i + (idx + 4)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 1) + (idx + 4)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 2) + (idx + 4)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 3) + (idx + 4)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 4) + (idx + 4)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = (i + 4);
                k_so_far++;
            }
        }

        for (; i < (idx + 4); i++) {
            if (distances_indexed_ptr[num_pts * i + (idx + 4)] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        idx_num_pts = (idx + 4) * num_pts;

        for (i = (idx + 4) + 1; (i + 4) < num_pts; i += 5) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 1] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 2] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 3] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 4] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 4;
                k_so_far++;
            }
        }

        for (; i < num_pts; i++) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }
    }


    //***LEFTOVER***
    for (; idx < num_pts; idx++) {
        double kdist = k_distances_indexed_ptr[idx];
        int k_so_far = 0;

        int idxk = idx * k;

        for (i = 0; (i + 4) < idx; i += 5) {
            if (distances_indexed_ptr[num_pts * i + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 1) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 2) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 3) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[num_pts * (i + 4) + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = (i + 4);
                k_so_far++;
            }
        }

        for (; i < idx; i++) {
            if (distances_indexed_ptr[num_pts * i + idx] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }

        int idx_num_pts = idx * num_pts;

        for (i = idx + 1; (i + 4) < num_pts; i += 5) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 1] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 1;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 2] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 2;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 3] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 3;
                k_so_far++;
            }

            if (distances_indexed_ptr[idx_num_pts + i + 4] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i + 4;
                k_so_far++;
            }
        }

        for (; i < num_pts; i++) {
            if (distances_indexed_ptr[idx_num_pts + i] <= kdist) {
                neighborhood_index_table_ptr[idxk + k_so_far] = i;
                k_so_far++;
            }
        }
    }

    return num_pts * (num_pts - 1.0);
}

/*
int Compute_K_Neighborhood_driver(int num_pts, int k, int dim, int num_reps){

}*/

int Compute_K_Neighborhood_driver(int num_pts, int k, int dim, int num_reps) {

    my_fun* fun_array = (my_fun*) calloc(NUM_FUNCTIONS, sizeof(my_fun));
    fun_array[0] = &ComputeKDistanceNeighborhoodAll;
    fun_array[1] = &ComputeKDistanceNeighborhood1;
    fun_array[2] = &ComputeKDistanceNeighborhoodAllInnerLoop;
    fun_array[3] = &ComputeKDistanceNeighborhoodAllFaster;

    char* fun_names[NUM_FUNCTIONS] = {"V0", "V1", "V2", "V3"};

    myInt64 start, end;
    double cycles;


//    FILE* results_file = open_with_error_check("../lrd_benchmark_k10_full.txt", "a");

    // INITIALIZE RANDOM INPUT

    double* distances_indexed_ptr = XmallocMatrixDoubleRandom(num_pts, num_pts);
    double* k_distances_indexed_ptr = XmallocVectorDoubleRandom(num_pts);

    int* neighborhood_index_table_ptr_true = XmallocMatrixInt(num_pts, k);

    ComputeKDistanceNeighborhood1(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr,
                                  neighborhood_index_table_ptr_true);

    double totalCycles = 0;

    for (int fun_index = 0; fun_index < NUM_FUNCTIONS; fun_index++) {
        //free(neighborhood_index_table_ptr);
        //neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);

        int* neighborhood_index_table_ptr = XmallocMatrixInt(num_pts, k);


        // VERIFICATION : ------------------------------------------------------------------------
        (*fun_array[fun_index])(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr,
                                neighborhood_index_table_ptr);

        int ver = test_neigh_ind(num_pts, k, neighborhood_index_table_ptr, neighborhood_index_table_ptr_true);

        if (ver != 1) {
            printf("RESULTS ARE DIFFERENT FROM BASELINE!\n");
//            exit(-1);
        }

        double numRuns = 10;

        start = start_tsc();
        for (size_t i = 0; i < numRuns; i++) {
            (*fun_array[fun_index])(num_pts, k, k_distances_indexed_ptr, distances_indexed_ptr,
                                    neighborhood_index_table_ptr);
        }
        end = stop_tsc(start);

        cycles = (double) end / numRuns;

        totalCycles += cycles;


        double flops = num_pts * (num_pts - 1);
        double perf = round((1000.0 * flops) / cycles) / 1000.0;
        printf("%s n:%d cycles:%lf perf:%lf \n\n", fun_names[fun_index], num_pts, cycles, perf);
//        fprintf(results_file, "%s, %d, %d, %lf, %lf\n", fun_names[fun_index], num_pts, k, cycles, perf);
    }

    printf("-------------\n");

    free(distances_indexed_ptr);
    free(k_distances_indexed_ptr);
    //free(neighborhood_index_table_ptr);
    //free(neighborhood_index_table_ptr_true);
//    fclose(results_file);
    return 0;
}
