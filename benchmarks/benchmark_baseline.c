//
// Created by fvasluia on 5/23/20.
//

# include "../include/benchmarks.h"
# include "../include/utils.h"
# include "../include/Algorithm.h"
#include "../unrolled/include/ComputePairwiseDistancesMMMUnroll.h"

#define NUMSETS 9

void benchmark_baseline(int num_runs) {
    int dims[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    int n = 500;
    int k=8;
    int split_dim = 4;
    int resolution = 3;

     double* perf_topo_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
     double* perf_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mem_baseline = XmallocMatrixDouble(num_runs, NUMSETS);


    FILE* base_dump_file = fopen("./benchmarks/text_dump/baseline_plots_dim_new.txt", "w");

    printf("Topology structure baseline\n");

    for(int run = 0; run < num_runs; ++run) {
        for (int i = 0; i < NUMSETS; ++i) {
            double perf = algorithm_driver_lattice(
                    n,
                    k,
                    dims[i],
                    split_dim,
                    resolution,
                    BASEComputeTopologyInfo,
                    ComputeLocalReachabilityDensityMerged_Point,
                    ComputeLocalReachabilityDensityMerged_Pipeline2,
                    ComputeLocalOutlierFactor_Pipeline2
            );
            perf_topo_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", n , dims[i], k, perf);
        }
    }

    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "%s\n", "Topology baseline");
        fprintf(base_dump_file, "%d\n", k);

        for(int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", dims[i]);

            for(int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_topo_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    }
//     // ---------------------------------------------------------------------------------------------------------------
//    printf("baseline");
//    for(int run = 0; run < num_runs; ++run) {
//        for (int i = 0; i < NUMSETS; ++i) {
//            double perf = algorithm_driver_baseline(
//                    n,
//                    k,
//                    dims[i],
//                    EuclideanDistance,
//                    ComputePairwiseDistances,
//                    ComputeKDistanceAll,
//                    ComputeKDistanceNeighborhoodAll,
//                    ComputeLocalReachabilityDensityMerged,
//                    ComputeLocalOutlierFactor
//            );
//            perf_baseline[run * NUMSETS + i] = perf;
//            printf("%d - %d - %d - %lf\n", n , dims[i], k, perf);
//        }
//    }
//
//    base_dump_file = fopen("./benchmarks/text_dump/baseline_plots_dim.txt", "a");
//    if (base_dump_file != NULL) {
//        fprintf(base_dump_file, "%s\n", "Baseline");
//        fprintf(base_dump_file, "%d\n", k);
//
//        for(int i = 0; i < NUMSETS; ++i) {
//            fprintf(base_dump_file, "\n%d, ", dims[i]);
//
//            for(int j = 0; j < num_runs; j++) {
//                fprintf(base_dump_file, "%lf, ", perf_baseline[j * NUMSETS + i]);
//            }
//        }
//        fclose(base_dump_file);
//    }
//
//    // --------------------------------------------------------------------------------------------------------------
//
//    printf("MMM baseline \n");
//    for(int run = 0; run < num_runs; ++run) {
//        for (int i = 0; i < NUMSETS; ++i) {
//            int B0 = 40, B1 = 20;
//            double  perf = algorithm_driver_knn_blocked_mmm(n,  k, dims[i],
//                                                                        B0, B1,
//                                                                        ComputePairwiseDistancesMMM_baseline,
//                                                                        KNN_fastest,
//                                                                        ComputeLocalReachabilityDensityMerged_Point,
//                                                                        ComputeLocalReachabilityDensityMerged_Pipeline2,
//                                                                        ComputeLocalOutlierFactor_Pipeline2);
//            perf_mmm_baseline[run * NUMSETS + i] = perf;
//            printf("%d - %d - %d - %lf\n", n , dims[i], k, perf);
//        }
//    }
//
//    base_dump_file = fopen("./benchmarks/text_dump/baseline_plots_dim.txt", "a");
//    if (base_dump_file != NULL) {
//        fprintf(base_dump_file, "\n%s\n", "MMM Baseline");
//        fprintf(base_dump_file, "%d\n", k);
//
//        for(int i = 0; i < NUMSETS; ++i) {
//            fprintf(base_dump_file, "\n%d, ", dims[i]);
//
//            for(int j = 0; j < num_runs; j++) {
//                fprintf(base_dump_file, "%lf, ", perf_mmm_baseline[j * NUMSETS + i]);
//            }
//        }
//        fclose(base_dump_file);
//    }
//
//    // --------------------------------------------------------------------------------------------------------------
//
//    printf("Memory layout baseline \n");
//    for(int run = 0; run < num_runs; ++run) {
//        for (int i = 0; i < NUMSETS; ++i) {
//            double  perf = algorithm_driver_knn_memory_struct(n, k, dims[i],
//                                                              EuclideanDistance,
//                                                              ComputePairwiseDistances,
//                                                              KNN_fastest,
//                                                              ComputeLocalReachabilityDensityMerged_Point,
//                                                              ComputeLocalReachabilityDensityMerged_Pipeline2,
//                                                              ComputeLocalOutlierFactor_Pipeline2);
//            perf_mem_baseline[run * NUMSETS + i] = perf;
//            printf("%d - %d - %d - %lf\n", n, dims[i], k, perf);
//        }
//    }
//
//     base_dump_file = fopen("./benchmarks/text_dump/baseline_plots_dim.txt", "a");
//    if (base_dump_file != NULL) {
//        fprintf(base_dump_file, "\n%s\n", "Mem Layout");
//        fprintf(base_dump_file, "%d\n", k);
//
//        for(int i = 0; i < NUMSETS; ++i) {
//            fprintf(base_dump_file, "\n%d, ", dims[i]);
//
//            for(int j = 0; j < num_runs; j++) {
//                fprintf(base_dump_file, "%lf, ", perf_mem_baseline[j * NUMSETS + i]);
//            }
//        }
//        fclose(base_dump_file);
//    }

}