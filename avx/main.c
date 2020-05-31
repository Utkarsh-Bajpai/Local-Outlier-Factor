//
// Created by fvasluia on 2/28/20.
//

#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"
#include "../include/file_utils.h"
#include "../include/Algorithm.h"
#include "../include/performance_measurement.h"
#include "../include/tests.h"

#include "include/ComputeLocalReachabilityDensity.h"
#include "include/ComputeReachabilityDistance.h"
#include "include/ComputeLocalReachabilityDensityMerged.h"
#include "include/ComputeLocalOutlierFactor.h"
#include "include/ComputeLocalReachabilityDensityMerged_Point.h"

#include "include/AVXMetrics.h"
#include "../include/lattice.h"
#include "include/AVXTopoInfo.h"
#include "include/ComputePairwiseDistanceMMMAvx.h"
#include "include/MMMAvx.h"

int main() {
    // AVX
    printf("The best thing about a boolean is even if you are wrong, you are only off by a bit\n");


    int num_pts_grid[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    int dim_grid[7] = {5, 10, 50, 100, 200, 300, 400};
    int blocks[5] = {50, 70, 100, 150};

    srand(5);
    int B0 = 40;
    int B1 = 20;
    int BK = 8;

    int nr_funs = 6;
    for (int fun_index = 3; fun_index < nr_funs; fun_index++) {
        for (int i = 0; i < 9; i++) {
            avx_mmm_driver(num_pts_grid[i], 40, B0, B1, BK, 14, fun_index);
        }

    }

//    for (int i = 7; i < 9; i++) {
//        avx_mmm_driver(num_pts_grid[i], 40, B0, B1, BK, 14, 3);
//    }

//    for (int fun_index = 0; fun_index < 5; fun_index++) {
//        for (int i = 0; i < 7; i++) {
//            avx_mmm_driver(1000, dim_grid[i], B0, B1, BK, 14, fun_index);
//        }
//    }



}
