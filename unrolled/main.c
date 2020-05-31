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
#include "include/ComputeKDistanceNeighborhood.h"
//#include "include/ComputeLocalReachabilityDensityMerged_Point.h"
#include "../avx/include/ComputeLocalOutlierFactor.h"
#include "../avx/include/ComputePairwiseDistanceMMMAvx.h"


#include "../avx/include/AVXMetrics.h"
#include "../avx/include/ComputeLocalReachabilityDensityMerged.h"
#include "../avx/include/ComputeLocalReachabilityDensityMerged_Point.h"
#include "../avx/include/AVXTopoInfo.h"

// ----------------------------------------------------- > PARAMETERS FOR PLOTTING !
int NUM_SPLITS = 4;
int RESOLUTION = 3;
int B0_PLOTS = 40,  B1_PLOTS = 20;

int num_pts_grid[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
int num_pts_grid_length = 9;
int num_pts_ref = 1000;

int k_grid[9] = { 2, 4, 8, 16, 32, 64, 128, 256, 512};
int k_grid_length = 9;
int k_ref = 64;        // with small k there will be no improvement from AVX LRDF pipeline 2

int dim_grid[7] = {2, 10, 50, 100, 200, 300, 400};
int dim_grid_length = 7;
int dim_ref = 40;

int main() {


    performance_plot_algo_num_pts( k_ref,  dim_ref,
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
                                   AVXTopologyInfo );

    /*
    'double (int, int, const double *, my_metrics_fnc, double *)' (aka 'double (int, int, const double *, double (*)(const double *, const double *, int), double *)') to parameter of type
    'my_dist_fnc' (aka 'void (*)(int, int, const double *, double (*)(const double *, const double *, int), double *)') [-Wincompatible-pointer-types]
    */
    //Compute_K_Neighborhood_driver(num_pts, k, dim, 10);
    //verify_function( num_pts, pnt_idx, k );

    // Baseline

    // GIVES EXEC CODE 11 with num of points > 1000
    /*
    double cycles_baseline = algorithm_driver_baseline( num_pts, k, dim,
                                                        EuclideanDistance,
                                                        ComputePairwiseDistances,
                                                        ComputeKDistanceAll,
                                                        ComputeKDistanceNeighborhoodAll,
                                                        ComputeLocalReachabilityDensityMerged,
                                                        ComputeLocalOutlierFactor);


    printf("Average number of cycles for baseline: %lf\n", cycles_baseline);
     */

    //test_pipeline_2(100, 10, 0.001, 1 );
    // test_pipeline_2_ver2( dim, num_pts, k, 0.001, ComputeLocalReachabilityDensityMerged_Pipeline2);
    // Pipeline 2


    /*
    double cycles_pipeline2 = algorithm_driver_knn_memory_struct(num_pts, k, dim,
                                                                 EuclideanDistance,
                                                                 ComputePairwiseDistances,
                                                                 KNN_fastest,
                                                                 ComputeLocalReachabilityDensityMerged_Point,
                                                                 ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                                 ComputeLocalOutlierFactor_Pipeline2);

    printf("Average number of cycles for pipeline 2: %lf\n", cycles_pipeline2);

    // Pipeline 3

    int num_splits = 4, resolution = 3;
    assert(num_splits%4 == 0);
    double cycles_pipeline_3 = algorithm_driver_lattice(num_pts, k, dim,
                                                        num_splits, resolution,
                                                        ComputeTopologyInfo,
                                                        ComputeLocalReachabilityDensityMerged_Point,
                                                        ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                        ComputeLocalOutlierFactor_Pipeline2);

    printf("Average number of cycles for pipeline 3: %lf\n", cycles_pipeline_3);

    // Pipeline 4

    int B0 = 40, B1 = 20;


    double cycles_pipeline_4 = algorithm_driver_knn_blocked_mmm(num_pts, k, 10,
                                                                B0, B1,
                                                                ComputePairwiseDistancesMMM_baseline,
                                                                KNN_fastest,
                                                                ComputeLocalReachabilityDensityMerged_Point,
                                                                ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                                ComputeLocalOutlierFactor_Pipeline2);

    printf("Average number of cycles for pipeline KNN+Pairwise Distance MMM: %lf\n", cycles_pipeline_4);
    */

}

