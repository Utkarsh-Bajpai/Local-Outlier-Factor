//
// Created by fvasluia on 2/28/20.
//

#include <stdio.h>
#include <stdlib.h>
#include "../include/utils.h"
#include "include/lof.h"
#include "../include/file_utils.h"

int main() {

    int max_num_pts = (1 << 20);
    //used as ratio from max_num_pts
    float max_num_neigh_r = 0.25f;
    int max_num_dim = 10000;

    /*
    ComputeLocalReachabilityDensity_1

    FILE* results_file = open_with_error_check("../benchmark_results.txt", "w");
    fprintf(results_file, "N,K,DIM,F2,F3,F4,F5,F6,F7,TOTAL\n");
    FILE* exec_file = open_with_error_check("Execution_time_c2.txt", "w");
    fprintf(exec_file, "Nr points, dim, nr neigh, python execution time\n");

    for (int num_pts = 8; num_pts <= max_num_pts; num_pts = num_pts * 2) {
        int num_neigh = 2;
        int num_dim = 2;

        main_lrd();
    }

    fclose(results_file);
    */
    printf("Beta is Latin for 'still doesn’t work' ");

}

