//
// Created by fvasluia on 5/23/20.
//

#include "../include/utils.h"
#include "../include/Algorithm.h"
#include "../avx/include/AVXTopoInfo.h"
#include "../avx/include/ComputeLocalOutlierFactorAVX.h"
#include "../avx/include/ComputeLocalReachabilityDensityMerged_PointAVX.h"
#include "../avx/include/AVXMetrics.h"
#include "../avx/include/ComputeLocalReachabilityDensityMergedAVX.h"
#include "../avx/include/ComputePairwiseDistanceMMMAvx.h"

#include "../unrolled/include/ComputePairwiseDistancesMMMUnroll.h"
#include "../unrolled/include/ComputeLocalReachabilityDensityMerged_PointUnrolled.h"
#include "../unrolled/include/ComputeLocalOutlierFactorUnrolled.h"
#include "../unrolled/include/ComputeKDistanceNeighborhoodUnrolled.h"
#include "../unrolled/include/ComputeLocalReachabilityDensityMergedUnrolled.h"
#include "../unrolled/include/ComputeKDistanceAllUnrolled.h"

// PROBLEMS WITH  printf("Baseline MMM pairwise dist \n"); !!!
// ----------------------------------------------------- > DEFINE ALL FILENAMES

char *BASELINE_FILENAME = "../benchmarks/text_dump/baseline_plots_num_pts_new.txt";
char *UNROLLED_FILENAME = "../benchmarks/text_dump/unrolled_plots_num_pts_new.txt";
char *AVX_FILENAME = "../benchmarks/text_dump/avx_plots_num_pts_new.txt";
char *MMM_PAIRWISE_FILENAME = "../benchmarks/text_dump/mmm_num_pts_new.txt";

char *K_P2_BASELINE_FILENAME = "../benchmarks/text_dump/baseline_plots_k_new.txt";
char *K_P2_UNROLLED_FILENAME = "../benchmarks/text_dump/unrolled_plots_k_new.txt";
char *K_P2_AVX_FILENAME = "../benchmarks/text_dump/avx_plots_k_new.txt";

char *DIM_P1_BL_FILENAME = "../benchmarks/text_dump/bl_plots_dim_new.txt";
char *DIM_P1_UNR_FILENAME = "../benchmarks/text_dump/unr_plots_dim_new.txt";
char *DIM_P1_AVX_FILENAME = "../benchmarks/text_dump/avx_plots_dim_new.txt";

// ----------------------------------------------------- > PARAMETERS FOR PLOTTING (copied from unrolled/main.c)
// Define the constantst for the code
int NUM_SPLITS = 4;
int RESOLUTION = 3;
int B0_PLOTS = 40, B1_PLOTS = 20;

int num_pts_grid[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
int num_pts_ref = 1000;
#define NUMSETS 9
// MIN_I_NUM_PTS and MAX_I_NUM_PTS constants should allow us to go over a part of te grid
// and to get at least some results
// if we start from MIN_I_PTS != 0, the results are appended to the exsiting file

// RUN 1 (with num_runs_num_pts = 20)
#define MIN_I_NUM_PTS 0
#define MAX_I_NUM_PTS 2
// should be 4

// RUN 2 (with num_runs_num_pts = 10)
//#define MIN_I_NUM_PTS 4
//#define MAX_I_NUM_PTS 6

// RUN 3 (with num_runs_num_pts = 2) // add more runs
//#define MIN_I_NUM_PTS 6
//#define MAX_I_NUM_PTS 9

// IMPORTANT: largest value of k_grid should be less than num_pts_ref!
int k_grid[10] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1000};
#define K_GRID 10
int k_ref = 8;        // with small k there will be no improvement from AVX LRDF pipeline 2
int k = 8;            // MAKE SURE THEY ARE THE SAME !!!

int dim_grid[12] = {2, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000};
#define DIM_GRID 12
// should be 12
int dim_ref = 40;
int dims = 40;          // MAKE SURE THEY ARE THE SAME !!!

void benchmark_baseline(int num_runs) {

    // int dims = 40;
    int n[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    // int k = 8;
    int split_dim = 4;
    int resolution = 3;

    double* perf_topo_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_knn = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mem_baseline = XmallocMatrixDouble(num_runs, NUMSETS);

    printf("In the function\n");
    char* mode = MIN_I_NUM_PTS == 0 ? "w" : "a";
    // if we start not from the first i, there already exist the file so we just need to append to it
    FILE* base_dump_file = fopen(BASELINE_FILENAME, mode); // "../benchmarks/text_dump/baseline_plots_dim_new.txt"

    printf("\nTopology structure baseline\n");

    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
            double perf = algorithm_driver_lattice(
                    num_pts_grid[i],//n[i],
                    k,
                    dims,
                    NUM_SPLITS, // split_dim,
                    RESOLUTION, // resolution,
                    BASEComputeTopologyInfo,
                    ComputeLocalReachabilityDensityMerged_Point,
                    ComputeLocalReachabilityDensityMerged_Pipeline2,
                    ComputeLocalOutlierFactor_Pipeline2
            );
            perf_topo_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);

        }
    }

    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Topology baseline");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_topo_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }
    // ---------------------------------------------------------------------------------------------------------------
    printf("\nbaseline\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
            double perf = algorithm_driver_baseline(
                    num_pts_grid[i],
                    k,
                    dims,
                    EuclideanDistance,
                    ComputePairwiseDistances,
                    ComputeKDistanceAll,
                    ComputeKDistanceNeighborhoodAll,
                    ComputeLocalReachabilityDensityMerged,
                    ComputeLocalOutlierFactor
            );
            perf_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    base_dump_file = fopen( BASELINE_FILENAME, "a"); // "../benchmarks/text_dump/baseline_plots_dim.txt"
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Baseline");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }


    // --------------------------------------------------------------------------------------------------------------

    printf("KNN MMM pairwise dist \n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            int B0 = 40, B1 = 20;
            double perf = algorithm_driver_knn_mmm_pairwise_dist(num_pts_grid[i], k, dims,
                                                                 B0, B1,
                                                                 ComputePairwiseDistancesMMM_baseline,
                                                                 KNN_fastest,
                                                                 ComputeLocalReachabilityDensityMerged_Point,
                                                                 ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                                 ComputeLocalOutlierFactor_Pipeline2);
            perf_mmm_knn[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    base_dump_file = fopen( BASELINE_FILENAME, "a");
    //base_dump_file = fopen( "../benchmarks/text_dump/baseline_plots_dim.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "KNN MMM baseline");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mmm_knn[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // --------------------------------------------------------------------------------------------------------------


    printf("Memory layout baseline \n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            double perf = algorithm_driver_knn_memory_struct(num_pts_grid[i], k, dims,
                                                             EuclideanDistance,
                                                             ComputePairwiseDistances,
                                                             KNN_fastest,
                                                             ComputeLocalReachabilityDensityMerged_Point,
                                                             ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                             ComputeLocalOutlierFactor_Pipeline2);
            perf_mem_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    base_dump_file = fopen( BASELINE_FILENAME, "a");
    // base_dump_file = fopen("../benchmarks/text_dump/baseline_plots_dim.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Mem Layout baseline");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mem_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }
}


void benchmark_unrolled(int num_runs) {
    // int dims = 40;
    int n[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};;
    // int k = 8;
    int split_dim = 4;
    int resolution = 3;

    double* perf_topo_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_knn = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mem_baseline = XmallocMatrixDouble(num_runs, NUMSETS);

    char* mode = MIN_I_NUM_PTS == 0 ? "w" : "a";
    FILE* base_dump_file = fopen(UNROLLED_FILENAME, mode); // "../benchmarks/text_dump/unrolled_plots_dim_new.txt"

    printf("Topology structure unrolled\n");

    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            double perf = algorithm_driver_lattice(
                    num_pts_grid[i],
                    k,
                    dims,
                    NUM_SPLITS, // split_dim,
                    RESOLUTION, // resolution,
                    BASEComputeTopologyInfo,
                    ComputeLocalReachabilityDensityMergedPointUnroll_Fastest,
                    ComputeLocalReachabilityDensityMerged_Pipeline2,
                    ComputeLocalOutlierFactor_Pipeline2
            );
            perf_topo_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Topology unrolled");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        // for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_topo_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }
    // ---------------------------------------------------------------------------------------------------------------

    printf("baseline unrolled\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            double perf = algorithm_driver_baseline(
                    num_pts_grid[i],
                    k,
                    dims,
                    UnrolledEuclideanDistance,
                    ComputePairwiseDistances,
                    ComputeKDistanceAllUnroll_Fastest,
                    ComputeKDistanceNeighborhoodAllUnrolled_Fastest,
                    ComputeLocalReachabilityDensityMergedUnroll_Fastest,
                    ComputeLocalOutlierFactorUnroll_Fastest
            );
            perf_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    base_dump_file = fopen(UNROLLED_FILENAME, "a");
    //base_dump_file = fopen("../benchmarks/text_dump/unrolled_plots_dim_new.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "\nBaseline unrolled");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // --------------------------------------------------------------------------------------------------------------

    printf("KNN MMM pairwise dist unrolled\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            int B0 = 40, B1 = 20;
            double perf = algorithm_driver_knn_mmm_pairwise_dist(num_pts_grid[i], k, dims,
                                                                 B0, B1,
                                                                 ComputePairwiseDistancesMMMUnroll_Fastest,
                                                                 KNN_fastest,
                                                                 ComputeLocalReachabilityDensityMergedPointUnroll_Fastest,
                                                                 ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                                 ComputeLocalOutlierFactor_Pipeline2);
            perf_mmm_knn[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    base_dump_file = fopen(UNROLLED_FILENAME, "a");
    // base_dump_file = fopen("../benchmarks/text_dump/unrolled_plots_dim_new.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "\nKNN MMM unrolled");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mmm_knn[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }


    // --------------------------------------------------------------------------------------------------------------


    printf("Memory layout unrolled\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            double perf = algorithm_driver_knn_memory_struct(num_pts_grid[i], k, dims,
                                                             UnrolledEuclideanDistance,
                                                             ComputePairwiseDistances,
                                                             KNN_fastest,
                                                             ComputeLocalReachabilityDensityMergedPointUnroll_Fastest,
                                                             ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                             ComputeLocalOutlierFactor_Pipeline2);
            perf_mem_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    base_dump_file = fopen(UNROLLED_FILENAME, "a");
    // base_dump_file = fopen("../benchmarks/text_dump/unrolled_plots_dim_new.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "\nMem Layout unrolled");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mem_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

}

void benchmark_avx(int num_runs) {
    // int dims = 40;
    int n[9] = {100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000};
    // int k = 8;
    int split_dim = 4;
    int resolution = 3;

    double* perf_topo_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_knn = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mem_baseline = XmallocMatrixDouble(num_runs, NUMSETS);


    char* mode = MIN_I_NUM_PTS == 0 ? "w" : "a";
    FILE* base_dump_file = fopen(AVX_FILENAME, mode);

    printf("Topology structure avx\n");

    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            double perf = algorithm_driver_lattice(
                    num_pts_grid[i],
                    k,
                    dims,
                    NUM_SPLITS, // split_dim,
                    RESOLUTION, // resolution,
                    AVXTopologyInfo,
                    ComputeLocalReachabilityDensityMerged_Point_AVX32_FASTEST,
                    ComputeLocalReachabilityDensityMerged_Pipeline2,
                    ComputeLocalOutlierFactor_2_AVX_Fastest
            );
            perf_topo_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Topology avx");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_topo_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }
    // ---------------------------------------------------------------------------------------------------------------

    printf("baseline avx");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            double perf = algorithm_driver_baseline(
                    num_pts_grid[i],
                    k,
                    dims,
                    AVXEuclideanDistance,
                    ComputePairwiseDistances,
                    ComputeKDistanceAllUnroll_Fastest,
                    ComputeKDistanceNeighborhoodAllUnrolled_Fastest,
                    ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128,
                    //need avx for the baseline LOF which was not finished by utkarsh
                    ComputeLocalOutlierFactorUnroll_Fastest
            );
            perf_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }


    base_dump_file = fopen(AVX_FILENAME, "a"); // "../benchmarks/text_dump/avx_plots_dim_new.txt"
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Baseline avx");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // --------------------------------------------------------------------------------------------------------------

    printf("KNN MMM pairwise dist \n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            int B0 = 40, B1 = 20;
            double perf = algorithm_driver_knn_mmm_pairwise_dist(num_pts_grid[i], k, dims,
                                                                 B0, B1,
                                                                 ComputePairwiseDistancesMMMAvx_Fastest,
                                                                 KNN_fastest,
                                                                 ComputeLocalReachabilityDensityMerged_Point_AVX32_FASTEST,
                                                                 ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                                 ComputeLocalOutlierFactor_2_AVX_Fastest);
            perf_mmm_knn[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    base_dump_file = fopen(AVX_FILENAME, "a");
    // base_dump_file = fopen("../benchmarks/text_dump/avx_plots_dim_new.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "KNN MMM avx");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mmm_knn[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // --------------------------------------------------------------------------------------------------------------


    printf("Memory layout avx \n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            double perf = algorithm_driver_knn_memory_struct(num_pts_grid[i], k, dims,
                                                             AVXEuclideanDistance,
                                                             ComputePairwiseDistances,
                                                             KNN_fastest,
                                                             ComputeLocalReachabilityDensityMerged_Point_AVX32_FASTEST,
                                                             ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                             ComputeLocalOutlierFactor_2_AVX_Fastest);
            perf_mem_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dims, k, perf);
        }
    }

    base_dump_file = fopen(AVX_FILENAME, "a");
    // base_dump_file = fopen("../benchmarks/text_dump/avx_plots_dim_new.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Mem Layout avx");
        fprintf(base_dump_file, "%d %d", k, dims);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
        //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mem_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

}

void benchmark_baseline_mmm_pairwise_distance(int num_runs){


    double* perf_mmm_pairwise_baseline = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_pairwise_unrolled = XmallocMatrixDouble(num_runs, NUMSETS);
    double* perf_mmm_pairwise_avx = XmallocMatrixDouble(num_runs, NUMSETS);

    char* mode = MIN_I_NUM_PTS == 0 ? "w" : "a";
    FILE* base_dump_file = fopen(MMM_PAIRWISE_FILENAME, mode);


    printf("Baseline MMM pairwise dist \n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < NUMSETS; ++i) {
            int B0 = 40, B1 = 20;
            double perf = algorithm_driver_baseline_mmm_pairwise_distance(num_pts_grid[i], k_ref, dim_ref,
                                                                          B0, B1,
                                                                          ComputePairwiseDistancesMMM_baseline,
                                                                          ComputeKDistanceAll,
                                                                          ComputeKDistanceNeighborhoodAll,
                                                                          ComputeLocalReachabilityDensityMerged,
                                                                          ComputeLocalOutlierFactor);
            perf_mmm_pairwise_baseline[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], k_ref, dim_ref, perf);
        }
    }

    //base_dump_file = fopen("../benchmarks/text_dump/baseline_plots_dim.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Baseline MMM");
        fprintf(base_dump_file, "%d %d", k_ref, dim_ref);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
            //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mmm_pairwise_baseline[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }
    // -------------------------------------------------------------------------->

    printf("Baseline MMM pairwise dist unrolled \n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
            // for (int i = 0; i < NUMSETS; ++i) {
            int B0 = 40, B1 = 20;
            double perf = algorithm_driver_baseline_mmm_pairwise_distance(num_pts_grid[i], k_ref, dim_ref,
                                                                          B0, B1,
                                                                          ComputePairwiseDistancesMMMUnroll_Fastest,
                                                                          ComputeKDistanceAllUnroll_Fastest,
                                                                          ComputeKDistanceNeighborhoodAllUnrolled_Fastest,
                                                                          ComputeLocalReachabilityDensityMergedUnroll_Fastest,
                                                                          ComputeLocalOutlierFactorUnroll_Fastest);
            perf_mmm_pairwise_unrolled[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], k_ref, dim_ref, perf);
        }
    }

    base_dump_file = fopen(MMM_PAIRWISE_FILENAME, 'a');
    // base_dump_file = fopen("../benchmarks/text_dump/unrolled_plots_dim_new.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Baseline MMM unrolled");
        fprintf(base_dump_file, "%d %d",  k_ref, dim_ref );

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
            // for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mmm_pairwise_unrolled[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // -------------------------------------------------------------------------->

    printf("Baseline MMM pairwise dist avx\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
            //for (int i = 0; i < NUMSETS; ++i) {
            int B0 = 40, B1 = 20;
            double perf = algorithm_driver_baseline_mmm_pairwise_distance(num_pts_grid[i], k_ref, dim_ref,
                                                                          B0, B1,
                                                                          ComputePairwiseDistancesMMMAvx_Fastest,
                                                                          ComputeKDistanceAllUnroll_Fastest,
                                                                          ComputeKDistanceNeighborhoodAllUnrolled_Fastest,
                                                                          ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128,
                                                                          ComputeLocalOutlierFactor);
            perf_mmm_pairwise_avx[run * NUMSETS + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_grid[i], dim_ref, k_ref, perf);
        }
    }

    base_dump_file = fopen(MMM_PAIRWISE_FILENAME, "a");
    // base_dump_file = fopen("../benchmarks/text_dump/avx_plots_dim_new.txt", "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "Baseline MMM avx");
        fprintf(base_dump_file, "%d %d", dim_ref, k_ref);

        for (int i = MIN_I_NUM_PTS; i < MAX_I_NUM_PTS; ++i) { //
            //for (int i = 0; i < NUMSETS; ++i) {
            fprintf(base_dump_file, "\n%d, ", num_pts_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_mmm_pairwise_avx[j * NUMSETS + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

}

// -------------------------------------------------------------------------------------------------------------------->
//
// second part of the pipeline comparison by k
//
// -------------------------------------------------------------------------------------------------------------------->


void benchmark_second_part_baseline(int num_runs){

    double* perf_baseline = XmallocMatrixDouble(num_runs, K_GRID);
    double* perf_memory_pipeline = XmallocMatrixDouble(num_runs, K_GRID);

    FILE* base_dump_file = fopen(K_P2_BASELINE_FILENAME, "w"); // "../benchmarks/text_dump/second_part_k_new_baseline.txt"

    // ---------------------------------------------------------------------------------------------------------------
    printf("second part pipeline original baseline");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < K_GRID; ++i) {
            double perf = algorithm_driver_second_part_original(num_pts_ref, k_grid[i], dim_ref,
                                                                ComputeLocalReachabilityDensityMerged,
                                                                ComputeLocalOutlierFactor);

            perf_baseline[run * K_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, dim_ref, k_grid[i], perf);
        }
    }

    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "second part pipeline original baseline");
        fprintf(base_dump_file, "%d", num_pts_ref);

        for (int i = 0; i < K_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", k_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline[j * K_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // --------------------------------------------------------------------------------------------------------------

    printf("second part pipeline 2 baseline");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < K_GRID; ++i) {

            double perf =  algorithm_driver_second_part_memory_optimization(num_pts_ref, k_grid[i], dim_ref,
                                                                            ComputeLocalReachabilityDensityMerged_Point,
                                                                            ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                                            ComputeLocalOutlierFactor_Pipeline2);

            perf_memory_pipeline[run * K_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, dim_ref, k_grid[i], perf);
        }
    }

    base_dump_file = fopen(K_P2_BASELINE_FILENAME, "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "second part pipeline 2 baseline");
        fprintf(base_dump_file, "%d", num_pts_ref);

        for (int i = 0; i < K_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", k_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_memory_pipeline[j * K_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }
}

void benchmark_second_part_unrolled(int num_runs){

    double* perf_baseline = XmallocMatrixDouble(num_runs, K_GRID);
    double* perf_memory_pipeline = XmallocMatrixDouble(num_runs, K_GRID);

    FILE* base_dump_file = fopen(K_P2_UNROLLED_FILENAME, "w"); // "../benchmarks/text_dump/second_part_k_new_unrolled.txt"

    // ---------------------------------------------------------------------------------------------------------------
    printf("second part pipeline original unrolled");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < K_GRID; ++i) {
            double perf = algorithm_driver_second_part_original(num_pts_ref, k_grid[i], dim_ref,
                                                                ComputeLocalReachabilityDensityMergedUnroll_Fastest,
                                                                ComputeLocalOutlierFactorUnroll_Fastest);

            perf_baseline[run * K_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, dim_ref, k_grid[i], perf);
        }
    }

    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "second part pipeline original unrolled");
        fprintf(base_dump_file, "%d", num_pts_ref);

        for (int i = 0; i < K_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", k_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline[j * K_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // --------------------------------------------------------------------------------------------------------------

    printf("second part pipeline 2 unrolled");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < K_GRID; ++i) {

            double perf =  algorithm_driver_second_part_memory_optimization(num_pts_ref, k_grid[i], dim_ref,
                                                            ComputeLocalReachabilityDensityMergedPointUnroll_Fastest,
                                                            ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                            ComputeLocalOutlierFactor_Pipeline2);

            perf_memory_pipeline[run * K_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, dim_ref, k_grid[i], perf);
        }
    }

    base_dump_file = fopen(K_P2_UNROLLED_FILENAME, "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "second part pipeline 2 unrolled");
        fprintf(base_dump_file, "%d", num_pts_ref);

        for (int i = 0; i < K_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", k_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_memory_pipeline[j * K_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }
}

void benchmark_second_part_avx(int num_runs){

    double* perf_baseline = XmallocMatrixDouble(num_runs, K_GRID);
    double* perf_memory_pipeline = XmallocMatrixDouble(num_runs, K_GRID);

    FILE* base_dump_file = fopen(K_P2_AVX_FILENAME, "w"); // "../benchmarks/text_dump/second_part_k_new_avx.txt"

    // ---------------------------------------------------------------------------------------------------------------
    printf("second part pipeline original avx");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < K_GRID; ++i) {
            double perf = algorithm_driver_second_part_original(num_pts_ref, k_grid[i], dim_ref,
                                                                ComputeLocalReachabilityDensityMerged_OUTER4_INNER4_AVX_128,
                                                                ComputeLocalOutlierFactorUnroll_Fastest);

            perf_baseline[run * K_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, dim_ref, k_grid[i], perf);
        }
    }

    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "second part pipeline original avx");
        fprintf(base_dump_file, "%d", num_pts_ref);

        for (int i = 0; i < K_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", k_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline[j * K_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // --------------------------------------------------------------------------------------------------------------

    printf("second part pipeline 2 avx");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < K_GRID; ++i) {

            //if(k_grid[i] <= num_pts_ref) {
            double perf = algorithm_driver_second_part_memory_optimization(num_pts_ref, k_grid[i], dim_ref,
                                                                           ComputeLocalReachabilityDensityMerged_Point_AVX32_FASTEST,
                                                                           ComputeLocalReachabilityDensityMerged_Pipeline2,
                                                                           ComputeLocalOutlierFactor_2_AVX_Fastest);

            perf_memory_pipeline[run * K_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, dim_ref, k_grid[i], perf);
            // }
        }
    }

    base_dump_file = fopen(K_P2_AVX_FILENAME, "a");
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "second part pipeline 2 avx");
        fprintf(base_dump_file, "%d", num_pts_ref);

        for (int i = 0; i < K_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", k_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_memory_pipeline[j * K_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }
}

// -------------------------------------------------------------------------------------------------------------------->

// first part of the pipeline comparison by dim

// -------------------------------------------------------------------------------------------------------------------->
// DIM_P1_BL_FILENAME
// DIM_P1_UNR_FILENAME
// DIM_P1_AVX_FILENAME

void benchmark_first_part_baseline(int num_runs) {

    double *perf_baseline_baseline = XmallocMatrixDouble(num_runs, DIM_GRID);
    // currently not since does not work properly
    // double *perf_mmm_pairwise_distance = XmallocMatrixDouble(num_runs,DIM_GRID);
    double *perf_knn_mmm_pairwise_dist = XmallocMatrixDouble(num_runs, DIM_GRID);
    double *perf_lattice = XmallocMatrixDouble(num_runs, DIM_GRID);

    FILE *base_dump_file;

    // 1 ) ---------------------------------------------------------------------------------------------------------------
    printf("First part baseline\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < DIM_GRID; ++i) {
            double perf = algorithm_driver_first_part_baseline(num_pts_ref, k_ref,
                                                                dim_grid[i],
                                                                EuclideanDistance,
                                                                ComputePairwiseDistances,
                                                                ComputeKDistanceAll,
                                                                ComputeKDistanceNeighborhoodAll);

            perf_baseline_baseline[run * DIM_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, k_ref, dim_grid[i], perf);
        }
    }

    base_dump_file = fopen(DIM_P1_BL_FILENAME,"w");     // first open mode should be "w"
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "First part baseline");
        fprintf(base_dump_file, "%d %d", num_pts_ref, k_ref);

        for (int i = 0; i < DIM_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", dim_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline_baseline[j * DIM_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }


    // 2 ) ---------------------------------------------------------------------------------------------------------------

    printf("First part knn_mmm_pairwise_dist\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < DIM_GRID; ++i) {

            int B0 = 40, B1 = 20;
            double perf =algorithm_driver_first_part_knn_mmm_pairwise_dist(num_pts_ref, k, dim_grid[i],
                                                                 B0, B1,
                                                                 ComputePairwiseDistancesMMMAvx_Fastest,
                                                                 KNN_fastest);

            perf_knn_mmm_pairwise_dist[run * DIM_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, k_ref, dim_grid[i], perf);
        }
    }


    base_dump_file = fopen(DIM_P1_BL_FILENAME,"a");     // first open mode should be "w"
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "First part knn_mmm_pairwise_dist");
        fprintf(base_dump_file, "%d %d", num_pts_ref, k_ref);

        for (int i = 0; i < DIM_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", dim_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_knn_mmm_pairwise_dist[j * DIM_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

}


void benchmark_first_part_unrolled(int num_runs) {

    double *perf_baseline_baseline = XmallocMatrixDouble(num_runs, DIM_GRID);
    double *perf_knn_mmm_pairwise_dist = XmallocMatrixDouble(num_runs, DIM_GRID);

    // currently not since does not work properly
    // double *perf_mmm_pairwise_distance = XmallocMatrixDouble(num_runs,DIM_GRID);
    // double *perf_lattice = XmallocMatrixDouble(num_runs, DIM_GRID);

    FILE *base_dump_file;

    // 1 ) ---------------------------------------------------------------------------------------------------------------
    printf("First part baseline unrolled\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < DIM_GRID; ++i) {
            double perf = algorithm_driver_first_part_baseline(num_pts_ref, k_ref,
                                                               dim_grid[i],
                                                               UnrolledEuclideanDistance,
                                                               ComputePairwiseDistances,
                                                               ComputeKDistanceAllUnroll_Fastest,
                                                               ComputeKDistanceNeighborhoodAllUnrolled_Fastest);

            perf_baseline_baseline[run * DIM_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, k_ref, dim_grid[i], perf);
        }
    }

    base_dump_file = fopen(DIM_P1_UNR_FILENAME,"w");     // first open mode should be "w"
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "First part baseline unrolled");
        fprintf(base_dump_file, "%d %d", num_pts_ref, k_ref);

        for (int i = 0; i < DIM_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", dim_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline_baseline[j * DIM_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // 2 ) ---------------------------------------------------------------------------------------------------------------

    printf("First part knn_mmm_pairwise_dist unrolled\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < DIM_GRID; ++i) {

            int B0 = 40, B1 = 20;
            double perf =algorithm_driver_first_part_knn_mmm_pairwise_dist(num_pts_ref, k, dim_grid[i],
                                                                           B0, B1,
                                                                           ComputePairwiseDistancesMMMUnroll_Fastest,
                                                                           KNN_fastest);

            perf_knn_mmm_pairwise_dist[run * DIM_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, k_ref, dim_grid[i], perf);
        }
    }


    base_dump_file = fopen(DIM_P1_UNR_FILENAME,"a");     // first open mode should be "w"
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "First part knn_mmm_pairwise_dist unrolled");
        fprintf(base_dump_file, "%d %d", num_pts_ref, k_ref);

        for (int i = 0; i < DIM_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", dim_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_knn_mmm_pairwise_dist[j * DIM_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

}


void benchmark_first_part_avx(int num_runs) {

    double *perf_baseline_baseline = XmallocMatrixDouble(num_runs, DIM_GRID);
    double *perf_knn_mmm_pairwise_dist = XmallocMatrixDouble(num_runs, DIM_GRID);

    // currently not since does not work properly
    // double *perf_mmm_pairwise_distance = XmallocMatrixDouble(num_runs,DIM_GRID);
    // double *perf_lattice = XmallocMatrixDouble(num_runs, DIM_GRID);

    FILE *base_dump_file;

    // 1 ) ---------------------------------------------------------------------------------------------------------------
    printf("First part baseline avx\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < DIM_GRID; ++i) {
            double perf = algorithm_driver_first_part_baseline(num_pts_ref, k_ref,
                                                               dim_grid[i],

                                                               AVXEuclideanDistance,
                                                               ComputePairwiseDistances,
                                                               ComputeKDistanceAllUnroll_Fastest,
                                                               ComputeKDistanceNeighborhoodAllUnrolled_Fastest);

            perf_baseline_baseline[run * DIM_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, k_ref, dim_grid[i], perf);
        }
    }

    base_dump_file = fopen(DIM_P1_AVX_FILENAME,"w");     // first open mode should be "w"
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "First part baseline avx");
        fprintf(base_dump_file, "%d %d", num_pts_ref, k_ref);

        for (int i = 0; i < DIM_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", dim_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_baseline_baseline[j * DIM_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

    // 2 ) ---------------------------------------------------------------------------------------------------------------

    printf("First part knn_mmm_pairwise_dist avx\n");
    for (int run = 0; run < num_runs; ++run) {
        printf("Run %d\n", run);
        for (int i = 0; i < DIM_GRID; ++i) {

            int B0 = 40, B1 = 20;
            double perf =algorithm_driver_first_part_knn_mmm_pairwise_dist(num_pts_ref, k, dim_grid[i],
                                                                           B0, B1,
                                                                           ComputePairwiseDistancesMMMAvx_Fastest,
                                                                           KNN_fastest);

            perf_knn_mmm_pairwise_dist[run * DIM_GRID + i] = perf;
            printf("%d - %d - %d - %lf\n", num_pts_ref, k_ref, dim_grid[i], perf);
        }
    }

    base_dump_file = fopen(DIM_P1_AVX_FILENAME,"a");     // first open mode should be "w"
    if (base_dump_file != NULL) {
        fprintf(base_dump_file, "\n\n%s\n", "First part knn_mmm_pairwise_dist avx");
        fprintf(base_dump_file, "%d %d", num_pts_ref, k_ref);

        for (int i = 0; i < DIM_GRID; ++i) {
            fprintf(base_dump_file, "\n%d, ", dim_grid[i]);

            for (int j = 0; j < num_runs; j++) {
                fprintf(base_dump_file, "%lf, ", perf_knn_mmm_pairwise_dist[j * DIM_GRID + i]);
            }
        }
        fclose(base_dump_file);
    } else {
        printf("\n\n WARNING: FILE IS NOT OPEN !\n\n");
    }

}