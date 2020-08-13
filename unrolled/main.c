//
// Created by fvasluia on 2/28/20.
//

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../include/utils.h"
#include "../include/file_utils.h"
#include "../include/lof_baseline.h"
#include "../include/tests.h"
#include "../include/Algorithm.h"
#include "../include/final_plots.h"

#include "include/ComputeLocalReachabilityDensity.h"
#include "include/ComputeReachabilityDistance.h"
//#include "include/ComputeLocalReachabilityDensityMerged.h"
//#include "include/ComputeLocalOutlierFactor.h"
#include "include/KNN.h"
#include "include/ComputePairwiseDistancesMMMUnroll.h"
#include "include/ComputeKDistanceNeighborhoodUnrolled.h"
//#include "include/ComputeLocalReachabilityDensityMerged_Point.h"
#include "../avx/include/ComputeLocalOutlierFactorAVX.h"
#include "../avx/include/ComputePairwiseDistanceMMMAvx.h"


#include "../avx/include/AVXMetrics.h"
#include "../avx/include/ComputeLocalReachabilityDensityMergedAVX.h"
#include "../avx/include/ComputeLocalReachabilityDensityMerged_PointAVX.h"
#include "../avx/include/AVXTopoInfo.h"
#include "../avx/include/ComputeLocalReachabilityDensityMergedAVX.h"

// ----------------------------------------------------- > PARAMETERS FOR PLOTTING !
int NUM_SPLITS = 4;
int RESOLUTION = 3;
int B0_PLOTS = 40, B1_PLOTS = 20;

int num_pts_grid[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
int num_pts_grid_length = 9;
int num_pts_ref = 1000;

int k_grid[9] = {2, 4, 8, 16, 32, 64, 128, 256, 512};
int k_grid_length = 9;
int k_ref = 64;        // with small k there will be no improvement from AVX LRDF pipeline 2

int dim_grid[7] = {2, 10, 50, 100, 200, 300, 400};
int dim_grid_length = 7;
int dim_ref = 40;

int main() {


    performance_plot_algo_num_pts(k_ref, dim_ref,
                                  num_pts_grid,
                                  "measurement_Final.txt", "baseline setting", "a",
                                  B0_PLOTS, B1_PLOTS, // for blocked MMM
                                  NUM_SPLITS, RESOLUTION, // for lattice

            // FUNCTIONS FOR baseline

                                  AVXEuclideanDistance,
                                  ComputePairwiseDistances,
                                  ComputeKDistanceAll,
                                  ComputeKDistanceNeighborhoodAll,
                                  ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128,
                                  ComputeLocalOutlierFactor_1,

            // FUNCTIONS FOR knn_memory_struct (additional)
                                  KNN_fastest,
                                  ComputeLocalReachabilityDensityMerged_Point_AVX32_FASTEST,
                                  ComputeLocalReachabilityDensityMerged_Pipeline2,
                                  ComputeLocalOutlierFactor_2_AVX_Fastest,

            // FUNCTIONS FOR knn_blocked_mmm
                                  ComputePairwiseDistancesMMMAvx_Fastest,

            // FUNCTIONS FOR lattice
                                  AVXTopologyInfo);


}

