#include <stdio.h>


#include "include/final_benchmarks.h"

int main() {

    // MEASUREMENTS FOR DIFFERENT K
    //this code part is very fast (max 2-3 min on my machine)

    int num_runs_k = 20;
    benchmark_second_part_baseline(num_runs_k);
    benchmark_second_part_unrolled(num_runs_k);
    benchmark_second_part_avx(num_runs_k);
    printf("Finished k measurements\n");


    // MEASUREMENTS FOR DIMS
    int num_runs_dim = 10;
    benchmark_first_part_baseline(num_runs_dim);
    benchmark_first_part_unrolled(num_runs_dim);
    benchmark_first_part_avx(num_runs_dim);

    return 0;
}
