//
// Created by fvasluia on 5/23/20.
//
# include "../include/benchmarks.h"
# include "../include/utils.h"
# include "../include/Algorithm.h"

#define NUMSETS 9

void benchmark_topology(int num_runs) {
    int num_pts[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    int k=8;
    int dim = 50;
    int split_dim = 4;
    int resolution = 3;

    double* cycles_matr_topo_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* cycles_matr_baseline = XmallocMatrixDouble(num_runs, NUMSETS);

    printf("Topology structure\n");

    for(int run = 0; run < num_runs; ++run) {
        for (int i = 0; i < NUMSETS; ++i) {
            double cycles = algorithm_driver_lattice(
                    num_pts[i],
                    k,
                    dim,
                    split_dim,
                    resolution,
                    BASEComputeTopologyInfo,
                    ComputeLocalReachabilityDensityMerged_Pipeline2,
                    ComputeLocalOutlierFactor_Pipeline2
                );
            cycles_matr_topo_baseline[run * NUMSETS + i] = cycles;
            printf("%d - %d - %d - %lf\n", num_pts[i], dim, k, cycles);
        }
    }

    printf("Clean baseline");
    for(int run = 0; run < num_runs; ++run) {
        for (int i = 0; i < NUMSETS; ++i) {
            double cycles = algorithm_driver_baseline(
                    num_pts[i],
                    k,
                    dim,
                    EuclideanDistance,
                    ComputePairwiseDistances,
                    ComputeKDistanceAll,
                    ComputeKDistanceNeighborhoodAll,
                    ComputeLocalReachabilityDensityMerged,
                    ComputeLocalOutlierFactor
            );
            cycles_matr_baseline[run * NUMSETS + i] = cycles;
            printf("%d - %d - %d - %lf\n", num_pts[i], dim, k, cycles);
        }
    }


    // generate the files for plotting
    FILE* dump_file = fopen("./benchmarks/text_dump/test_topo_baseline.txt", "w");
    if (dump_file != NULL) {
        fprintf(dump_file, "%s\n", "topology structure baseline");
        fprintf(dump_file, "%d\n", k);

        for(int i = 0; i < NUMSETS; ++i) {
            fprintf(dump_file, "\n%d, ", num_pts[i]);

            for(int j = 0; j < num_runs; j++) {
                fprintf(dump_file, "%lf, ", cycles_matr_topo_baseline[j * NUMSETS + i]);
            }
        }
        fclose(dump_file);
    }


    FILE* base_dump_file = fopen("./benchmarks/text_dump/test_baseline.txt", "w");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "%s\n", "baseline");
        fprintf(base_dump_file, "%d\n", k);

        for(int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts[i]);

            for(int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", cycles_matr_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    }

}