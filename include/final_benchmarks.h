//
// Created by fvasluia on 5/23/20.
//

#ifndef FASTLOF_FINAL_BENCHMARKS_H
#define FASTLOF_FINAL_BENCHMARKS_H

void benchmark_topology(int num_runs);

void benchmark_baseline(int num_runs);

void benchmark_unrolled(int num_runs);

void benchmark_avx(int num_runs);


void benchmark_baseline_mmm_pairwise_distance(int num_runs);
// measuring the performance improvement for different values of k

void benchmark_second_part_baseline(int num_runs);
void benchmark_second_part_unrolled(int num_runs);
void benchmark_second_part_avx(int num_runs);

// measuring performance for different values of dim
void benchmark_first_part_baseline(int num_runs);
void benchmark_first_part_unrolled(int num_runs);
void benchmark_first_part_avx(int num_runs);


#endif //FASTLOF_FINAL_BENCHMARKS_H
