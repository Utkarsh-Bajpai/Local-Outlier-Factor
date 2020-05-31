#include <stdio.h>


#include "include/Algorithm.h"
#include "include/lof_baseline.h"
#include "include/final_plots.h"

#include "unrolled/include/ComputePairwiseDistancesMMMUnroll.h"
#include "avx/include/ComputeLocalOutlierFactor.h"
//#include "memory/include/ComputePairwiseDistance.h"

#include "include/benchmarks.h"

int main() {
    printf("Hello, World!\n");

    benchmark_baseline(10);

//
//    // int num_pts = 100, pnt_idx = 54, k = 10, dim = 20;
//
//    /*
//    double cycles_pipeline2 = algorithm_driver_knn_memory_struct( num_pts, k, dim,
//                                                                  EuclideanDistance,
//                                                                  ComputePairwiseDistances,
//                                                                  KNN,
//                                                                  ComputeLocalReachabilityDensityMerged_Pipeline2,
//                                                                  ComputeLocalOutlierFactor_Pipeline2  //        1444934.184000; 1361051.732000
//                                                                  //ComputeLocalOutlierFactor_2_unroll_AVX_fixed //  1959217.295000
//                                                                  );
//
//    printf("Average number of cycles for pipeline 2: %lf\n", cycles_pipeline2);
//    */
//    performance_plot_algo_num_pts( k_ref,  dim_ref,
//                           "measurement_test.txt", "baseline setting", "w",
//                                    B0_PLOTS, B1_PLOTS, // for blocked MMM
//                                    NUM_SPLITS, RESOLUTION, // for lattice
//
//                                    // FUNCTIONS FOR baseline
//                                    EuclideanDistance,
//                                    ComputePairwiseDistances,
//                                    ComputeKDistanceAll,
//                                    ComputeKDistanceNeighborhoodAll,
//                                    ComputeLocalReachabilityDensityMerged,
//                                    ComputeLocalOutlierFactor,
//
//                                    // FUNCTIONS FOR knn_memory_struct (additional)
//                                    KNN_fastest,
//                                    ComputeLocalReachabilityDensityMerged_Point,
//                                    ComputeLocalReachabilityDensityMerged_Pipeline2,
//                                    ComputeLocalOutlierFactor_Pipeline2,
//
//                                    // FUNCTIONS FOR knn_blocked_mmm
//                                   ComputePairwiseDistancesMMM_baseline,
//
//                                    // FUNCTIONS FOR lattice
//                                    ComputeTopologyInfo );
//

    return 0;
}
