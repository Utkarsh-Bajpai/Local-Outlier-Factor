/**
* Plotting drivers
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../include/lof_baseline.h"
#include "../unrolled/include/KNN.h"
#include "../avx/include/MMMAvx.h"
#include "../include/ComputeTopologyInfo.h"
#include "../include/Algorithm.h"
#include "../include/final_plots.h"


// ---------------------------------------------------------------------------------------------->
// Plot 3 flops per cycle (4 pipelines x 3: baseline + unrolled + avx)
#define NUM_SINGLE_RUNS = 10
//  Single static assignment (SSA) code: !
void performance_plot_algo_num_pts(int k_ref, int dim_ref,
                                   int* num_pts_grid,
                                   const char* file_name,
                                   const char* function_name, const char* mode,
                                   int B0, int B1, // for blocked MMM
                                   int num_splits, int resolution, // for lattice
                                  // ubajpai@student.ethz.ch, rpasca@student.ethz.ch, fvasluianu@student.ethz.ch
                                  // Bajpai Utkarsh;
        //Pasca Razvan;
        //Theodoridis Theodoros
        //Vasluianu Florin

        // FUNCTIONS FOR baseline
                                   my_metrics_fnc metrics_fnc,
                                   my_dist_fnc dist_fnc,
                                   my_kdistall_fnc kdist_fnc,
                                   my_neigh_fnc neigh_fnc,
                                   my_lrdm_fnc lrdm_fnc,
                                   my_lof_fnc lof_fnc,

        // FUNCTIONS FOR knn_memory_struct (additional)
                                   my_knn_fnc knn_fnc,
                                   my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                   my_lrdm2_fnc lrdm2_fnc,
                                   my_lof2_fnc lof2_fnc,

        // FUNCTIONS FOR knn_blocked_mmm
                                   my_mmm_dist_fnc dist_block_fnc,

        // FUNCTIONS FOR lattice
                                   my_topolofy_fnc topolofy_fnc) {
    /**
     * Creates information for 4 lines: baseline, knn_memory_struct, knn_blocked_mmm and lattice
     *
     * @param file_name
     * @param function_name
     *
     */

    unsigned int rep;

    //for (num_pts = 4; num_pts < max_num_pts; num_pts += step)
    int num_pts_current;

    int i_start = 0, i_fin = 5;
    for (int i = i_start; i < i_fin; i++) {

        FILE* results_file = fopen(file_name, mode);
        // SHOULD BE LESS THAN NUM PTS !!! - add assertion

        if (results_file == NULL) {
            fprintf(stderr, "Error at opening %s", results_file);
            printf("Check if the file was generated or the path in open_with_error_check\n");
        }

        fprintf(results_file, "%s\n", function_name);
        fprintf(results_file, "%d\n", k_ref);
        fprintf(results_file, "\n");


        num_pts_current = num_pts_grid[i];
        if (num_pts_current >= k_ref) {    // number of neighbrs can't be lower than the number of points

            // MEASUREMENTS FOR baseline
            fprintf(results_file, "\nBaseline\n");
            fprintf(results_file, "%d, ", num_pts_current);
            printf("%d\n", num_pts_current);

            for (rep = 0; rep < 10; rep++) {

                double res = algorithm_driver_baseline(num_pts_current, k_ref, dim_ref,
                        // functions
                                                       metrics_fnc,
                                                       dist_fnc,
                                                       kdist_fnc,
                                                       neigh_fnc,
                                                       lrdm_fnc,
                                                       lof_fnc);

                fprintf(results_file, "%f, ", res);
                printf("%f, ", res);
            }

            printf("\nBaseline benchmark done!\n");
            printf("-------------------------\n");

            // MEASUREMENTS FOR knn_memory_struct
            /*
            fprintf(results_file, "\nknn_memory_struct\n");
            fprintf(results_file, "%d, ", num_pts_current);
            for (rep = 0; rep < 10; rep++) {

                double res = algorithm_driver_knn_memory_struct(num_pts_current, k_ref, dim_ref,
                        // functions
                                                                metrics_fnc,
                                                                dist_fnc,
                                                                knn_fnc,
                                                                lrdm2_pnt_fnc,
                                                                lrdm2_fnc,
                                                                lof2_fnc);

                fprintf(results_file, "%f, ", res);
            }

            printf("KNN with memory lof benchmark done!\n");
            printf("-------------------------\n");
            */

            // MEASUREMENTS FOR knn_blocked_mmm
            fprintf(results_file, "\nknn_blocked_mmm\n");
            fprintf(results_file, "%d, ", num_pts_current);
            for (rep = 0; rep < 10; rep++) {

                double res = algorithm_driver_knn_blocked_mmm(num_pts_current, k_ref, dim_ref,
                                                              B0, B1,
                        // functions
                                                              dist_block_fnc,
                                                              knn_fnc,
                                                              lrdm2_pnt_fnc,
                                                              lrdm2_fnc,
                                                              lof2_fnc);

                fprintf(results_file, "%f, ", res);
                printf("%f, ", res);
            }

            printf("\nKNN with MMM pairwise and lof memory structure benchmark done!\n");
            printf("-------------------------\n");


            // MEASUREMENTS FOR lattice
            fprintf(results_file, "\nlattice\n");
            fprintf(results_file, "%d, ", num_pts_current);

            for (rep = 0; rep < 10; rep++) {

                double res = algorithm_driver_lattice(num_pts_current, k_ref, dim_ref,
                                                      num_splits, resolution,
                        // functions
                                                      topolofy_fnc,
                                                      lrdm2_pnt_fnc,
                                                      lrdm2_fnc,
                                                      lof2_fnc);

                fprintf(results_file, "%f, ", res);
                printf("%f, ", res);
            }

            printf("\nLattice heuristic with lof memory structure benchmark done!\n");
            printf("-------------------------\n");


        }
        fprintf(results_file, "\n");
        fclose(results_file);
    } // iterate over dimensions



}

void driver_runtime_performance_plot_algo_num_pts(int k_ref, int dim_ref,
                                                  int* num_pts_grid,
                                                  int num_rep_measurements,
                                                  const char* file_name,
                                                  const char* function_name, const char* mode,
                                                  int B0, int B1, // for blocked MMM
                                                  int num_splits, int resolution, // for lattice

        // FUNCTIONS FOR baseline
                                                  my_metrics_fnc metrics_fnc,
                                                  my_dist_fnc dist_fnc,
                                                  my_kdistall_fnc kdist_fnc,
                                                  my_neigh_fnc neigh_fnc,
                                                  my_lrdm_fnc lrdm_fnc,
                                                  my_lof_fnc lof_fnc,

        // FUNCTIONS FOR knn_memory_struct (additional)
                                                  my_knn_fnc knn_fnc,
                                                  my_lrdm2_pnt_fnc lrdm2_pnt_fnc,
                                                  my_lrdm2_fnc lrdm2_fnc,
                                                  my_lof2_fnc lof2_fnc,

        // FUNCTIONS FOR knn_blocked_mmm
                                                  my_mmm_dist_fnc dist_block_fnc,

        // FUNCTIONS FOR lattice
                                                  my_topolofy_fnc topolofy_fnc) {
    /**
     * run the function for all the pipelines we want to compare
     */

    // return 0;

    FILE* results_file = fopen(file_name, mode);
    // SHOULD BE LESS THAN NUM PTS !!! - add assertion

    if (results_file == NULL) {
        fprintf(stderr, "Error at opening %s", results_file);
        printf("Check if the file was generated or the path in open_with_error_check\n");
    }

    unsigned int rep;

    fprintf(results_file, "%s\n", function_name);
    fprintf(results_file, "%d\n", k_ref);
    fprintf(results_file, "\n");

    //for (num_pts = 4; num_pts < max_num_pts; num_pts += step)
    int num_pts_current;
    double res;

    for (int i = 6; i < 9; i++) { // iterate over entire point grid

        num_pts_current = num_pts_grid[i];
        printf("NUMBER OF POINTS %d\n", num_pts_current);
        if (num_pts_current >= k_ref) {    // number of neighbrs can't be lower than the number of points

            // MEASUREMENTS FOR baseline
            fprintf(results_file, "\nBaseline\n");
            fprintf(results_file, "%d, ", num_pts_current);

            // START MEASUREMENTS IN

            clock_t tic = clock();
            for (rep = 0; rep < num_rep_measurements; rep++) {

                algorithm_driver_baseline(num_pts_current, k_ref, dim_ref,
                        // functions
                                          metrics_fnc,
                                          dist_fnc,
                                          kdist_fnc,
                                          neigh_fnc,
                                          lrdm_fnc,
                                          lof_fnc);

                //printf("%f, ", res);
                //fprintf(results_file, "%f, ", res);
            }
            clock_t toc = clock();
            res = ( 1000.0 * (double)(toc - tic) / CLOCKS_PER_SEC) / num_rep_measurements;
            printf("%f, ", res);
            fprintf(results_file, "%f, ", res);

            fclose(results_file);
            results_file = fopen(file_name, "a");


            //printf("Baseline benchmark done!\n");
            //printf("-------------------------\n");

            // MEASUREMENTS FOR knn_memory_struct
            fprintf(results_file, "\nknn_memory_struct\n");
            fprintf(results_file, "%d, ", num_pts_current);

            tic = clock();
            for (rep = 0; rep < num_rep_measurements; rep++) {


                double res = algorithm_driver_baseline_mmm_pairwise_distance(num_pts_current, k_ref, dim_ref,
                                                                             B0, B1,
                                                                             dist_block_fnc,
                                                                             kdist_fnc,
                                                                             neigh_fnc,
                                                                             lrdm_fnc,
                                                                             lof_fnc);

            }
            toc = clock();
            res = ( 1000.0 * (double)(toc - tic) / CLOCKS_PER_SEC) / num_rep_measurements;
            printf("%f, ", res);
            fprintf(results_file, "%f, ", res);

            fclose(results_file);
            results_file = fopen(file_name, "a");

            //printf("KNN with memory lof benchmark done!\n");
            //printf("-------------------------\n");

            // MEASUREMENTS FOR knn_blocked_mmm
            fprintf(results_file, "\nknn_blocked_mmm\n");
            fprintf(results_file, "%d, ", num_pts_current);

            tic = clock();
            for (rep = 0; rep < num_rep_measurements; rep++) {

                algorithm_driver_knn_mmm_pairwise_dist(num_pts_current, k_ref, dim_ref,
                                                       B0, B1,
                                                       dist_block_fnc,
                                                       knn_fnc,
                                                       lrdm2_pnt_fnc,
                                                       lrdm2_fnc,
                                                       lof2_fnc);

            }
            toc = clock();
            res = ( 1000.0 * (double)(toc - tic) / CLOCKS_PER_SEC) / num_rep_measurements;
            printf("%f, ", res);
            fprintf(results_file, "%f, ", res);

            fclose(results_file);
            results_file = fopen(file_name, "a");
            //printf("\n\nKNN with MMM pairwise and lof memory structure benchmark done!\n");
            //printf("-------------------------\n\n");


            // MEASUREMENTS FOR lattice
            fprintf(results_file, "\nlattice\n");
            fprintf(results_file, "%d, ", num_pts_current);

            tic = clock();
            for (rep = 0; rep < num_rep_measurements; rep++) {

                algorithm_driver_lattice(num_pts_current, k_ref, dim_ref,
                                         num_splits, resolution,
                        // functions
                                         topolofy_fnc,
                                         lrdm2_pnt_fnc,
                                         lrdm2_fnc,
                                         lof2_fnc);
            }
            toc = clock();
            res = ( 1000.0 * (double)(toc - tic) / CLOCKS_PER_SEC) / num_rep_measurements;
            printf("%f, ", res);
            fprintf(results_file, "%f, ", res);

            //printf("Lattice heuristic with lof memory structure benchmark done\n!");
            //printf("-------------------------\n");
            fclose(results_file);
            results_file = fopen(file_name, "a");

        }
        printf("\n");
        fprintf(results_file, "\n");
    } // iterate over dimensions
    fclose(results_file);
}
