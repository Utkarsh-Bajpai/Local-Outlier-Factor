cmake_minimum_required(VERSION 3.15)
project(FastLOF C)

set(CMAKE_C_STANDARD 11)


#CHANGED FROM  CMAKE_CXX_FLAGS
set(CMAKE_C_FLAGS "-O0 -fno-tree-vectorize -mavx2 -mfma")

## for the baseline
#add_executable(baseline
#        baseline/main.c
#        #        baseline/include/lof.h
#        include/metrics.h
#        src/metrics.c
#        include/utils.h
#        include/lattice.h
#        include/file_utils.h
#        baseline/main.c
#        #        baseline/src/lof.c
#        src/utils.c
#        src/lattice.c
#        src/file_utils.c
#        include/tests.h
#        src/tests.c
#        include/lof_baseline.h
#        src/lof_baseline.c
#        include/performance_measurement.h
#        src/performance_measurement.c)
#
##for the unrolled version
#add_executable(unrolled
#
#        avx/include/MMMAvx.h
#        avx/performance_improvement/MMMAvx.c
#
#        include/final_plots.h
#        src/final_plots.c
#
#        unrolled/include/KNN.h
#        unrolled/performance_improvement/KNN.c
#
#        avx/include/ComputePairwiseDistanceMMMAvx.h
#        avx/performance_improvement/ComputePairwiseDistanceMMMAvx.c
#
#        #        avx/include/ComputeLocalOutlierFactor.h
#        #        avx/performance_improvement/ComputeLocalOutlierFactor.c
#        avx/include/AVX_utils.h
#        avx/src/AVX_utils.c
#
#
#        #unrolled/include/Algorithm.h
#        include/metrics.h
#        src/metrics.c
#        include/Algorithm.h
#        src/Algorithm.c
#        include/sort.h
#        include/utils.h
#        include/file_utils.h
#        include/tests.h
#        include/tsc_x86.h
#        include/lattice.h
#        include/ComputeTopologyInfo.h
#        unrolled/include/ComputeLocalReachabilityDensity.h
#        unrolled/include/ComputeReachabilityDistance.h
#        unrolled/include/ComputeKDistanceAll.h
#        unrolled/include/ComputeKDistanceNeighborhood.h
#        unrolled/include/ComputeKDistanceObject.h
#        unrolled/include/ComputePairwiseDistancesMMMUnroll.h
#        unrolled/include/ComputeLocalOutlierFactorUnrolled.h
#        unrolled/include/KNN.h
#        unrolled/main.c
#        src/sort.c
#        src/lattice.c
#        unrolled/performance_improvement/ComputeLocalReachabilityDensity.c
#        unrolled/performance_improvement/ComputeReachabilityDistance.c
#        # unrolled/performance_improvement/ComputeLocalOutlierFactor.c
#        unrolled/performance_improvement/ComputeKDistanceAll.c
#        unrolled/performance_improvement/ComputeKDistanceNeighborhood.c
#        unrolled/performance_improvement/ComputeKDistanceObject.c
#        unrolled/performance_improvement/FasterMetrics.c
#        unrolled/performance_improvement/ComputeLocalReachabilityDensityMerged.c
#        src/ComputeTopologyInfo.c
#        unrolled/performance_improvement/KNN.c
#        unrolled/performance_improvement/ComputePairwiseDistancesMMMUnroll.c
#        src/tests.c
#        src/lattice.c
#        src/utils.c
#        src/file_utils.c
#        # unrolled/src/Algorithm.c
#        unrolled/performance_improvement/KNN.c
#        include/lof_baseline.h
#        src/lof_baseline.c
#        include/performance_measurement.h
#        src/performance_measurement.c
#        unrolled/performance_improvement/MMM.c
#        unrolled/include/MMM.h
#        #unrolled/include/ComputeLocalReachabilityDensityMerged_Point.h
#        #unrolled/performance_improvement/ComputeLocalReachabilityDensityMerged_Point.c
#
#
#        avx/include/AVXMetrics.h
#        avx/performance_improvement/AVX_metrics.c
#        avx/include/AVXTopoInfo.h
#        avx/performance_improvement/AVXTopoInfo.c
#        avx/include/ComputeLocalOutlierFactorUnrolled.h
#        avx/performance_improvement/ComputeLocalOutlierFactorUnrolled.c
#        avx/include/ComputeLocalReachabilityDensityMerged.h
#        avx/performance_improvement/ComputeLocalReachabilityDensityMerged.c
#        avx/include/ComputeLocalReachabilityDensityMerged_PointUnrolled.h
#        avx/performance_improvement/ComputeLocalReachabilityDensityMerged_PointUnrolled.c
#
#        avx/performance_improvement/AVX_lattice.c
#        avx/include/AVX_utils.h
#        )
#
#add_executable(avx
#        avx/main.c
#        include/Algorithm.h
#        src/Algorithm.c
#        include/metrics.h
#        src/metrics.c
#        include/lof_baseline.h
#        src/lof_baseline.c
#        include/file_utils.h
#        include/tests.h
#        include/sort.h
#        include/tsc_x86.h
#        include/utils.h
#        include/lattice.h
#        # for unrolled euclidean
#        unrolled/performance_improvement/FasterMetrics.c
#        avx/include/AVX_utils.h
#        avx/include/AVXMetrics.h
#        avx/performance_improvement/AVX_metrics.c
#        avx/performance_improvement/AVX_lattice.c
#        #        avx/include/ComputeTopologyInfo.h
#        avx/include/ComputePairwiseDistanceMMMAvx.h
#        #        avx/include/ComputeLocalReachabilityDensityMerged.h
#        #avx/include/ComputeKDistanceAll.h
#        #avx/include/ComputeKDistanceNeighborhood.h
#        #avx/include/ComputeKDistanceObject.h
#        #        avx/include/ComputeLocalReachabilityDensity.h
#        #avx/include/ComputeReachabilityDistance.h
#        #avx/include/ComputeLocalOutlierFactorUnrolled.h
#        avx/include/AVXTopoInfo.h
#        #       avx/include/metrics.h
#        src/tests.c
#        src/utils.c
#        src/file_utils.c
#        src/lattice.c
#        #        avx/src/sort.c
#        avx/performance_improvement/AVX_lattice.c
#        src/sort.c
#        # avx/performance_improvement/ComputeLocalReachabilityDensityMerged.c
#        #avx/performance_improvement/ComputeKDistanceAll.c
#        # avx/performance_improvement/ComputeKDistanceNeighborhood.c
#        #avx/performance_improvement/ComputeKDistanceObject.c
#        #        avx/performance_improvement/ComputeLocalOutlierFactor.c
#        #        avx/performance_improvement/ComputeReachabilityDistance.c
#        avx/performance_improvement/ComputePairwiseDistanceMMMAvx.c
#        avx/performance_improvement/FasterMetrics.c
#        #        avx/performance_improvement/ComputeTopologyInfo.c
#        include/performance_measurement.h
#        src/performance_measurement.c
#        src/metrics.c
#        avx/performance_improvement/MMMAvx.c
#        avx/include/MMMAvx.h
#        unrolled/performance_improvement/MMM.c
#        unrolled/include/MMM.h
#        #       avx/performance_improvement/ComputeTopologyInfo.c
#        avx/performance_improvement/AVX_lattice.c
#        avx/performance_improvement/AVX_metrics.c
#        avx/performance_improvement/AVXTopoInfo.c
#        include/performance_measurement.h
#        src/performance_measurement.c
#        src/ComputeTopologyInfo.c
#        avx/performance_improvement/AVXTopoInfo.c
#        include/performance_measurement.h
#        src/performance_measurement.c
#        avx/include/AVX_utils.h
#        avx/src/AVX_utils.c
#        #avx/include/ComputeLocalReachabilityDensityMerged_PointUnrolled.h
#        #avx/performance_improvement/ComputeLocalReachabilityDensityMerged_PointUnrolled.c
#        unrolled/include/ComputeLocalReachabilityDensityMerged_PointUnrolled.h
#
#        #avx/performance_improvement/ComputeLocalOutlierFactorUnrolled.c
#        #avx/include/ComputeLocalOutlierFactorUnrolled.h
#
#        avx/include/AVXTopoInfo.h
#        avx/performance_improvement/AVX_lattice.c
#        )
#
#add_executable(memory
#        memory/main.c
#        #include/Algorithm.h
#        #src/Algorithm.c
#        include/lof_baseline.h
#        src/lof_baseline.c
#        include/performance_measurement.h
#        src/performance_measurement.c
#        include/file_utils.h
#        include/tests.h
#        include/sort.h
#        include/tsc_x86.h
#        include/utils.h
#        include/lattice.h
#        include/metrics.h
#        src/tests.c
#        src/utils.c
#        src/file_utils.c
#        src/lattice.c
#        src/metrics.c
#        memory/include/ComputeLocalOutlierFactor.h
#        memory/include/ComputeKDistanceObject.h
#        memory/include/ComputeKDistanceNeighborhood.h
#        memory/include/ComputeKDistanceAll.h
#        memory/include/ComputeLocalReachabilityDensityMerged.h
#        memory/include/ComputeReachabilityDistance.h
#        memory/include/ComputeLocalReachabilityDensity.h
#        memory/performance_improvement/ComputeLocalOutlierFactor.c
#        memory/performance_improvement/ComputeKDistanceObject.c
#        memory/performance_improvement/ComputeKDistanceNeighborhood.c
#        memory/performance_improvement/ComputeKDistanceAll.c
#        memory/performance_improvement/ComputeLocalReachabilityDensityMerged.c
#        memory/performance_improvement/ComputeReachabilityDistance.c
#        memory/performance_improvement/ComputePairwiseDistanceMMMAnastasia.c
#        memory/include/ComputePairwiseDistanceMMMAnastasia.h
#        memory/include/ComputeLocalReachabilityDensityMerged_Pipeline2.h
#        memory/performance_improvement/ComputeLocalReachabilityDensityMerged_Pipeline2.c)

add_executable(final_benchmark
        main.c

        include/final_benchmarks.h
        benchmarks/final_benchmarks.c

        include/Algorithm.h
        src/Algorithm.c
        include/lof_baseline.h
        src/lof_baseline.c
        include/metrics.h
        src/metrics.c
        include/utils.h
        src/utils.c
        include/file_utils.h
        src/file_utils.c
        include/lattice.h
        src/lattice.c
        include/sort.h
        src/sort.c
        include/tests.h
        src/tests.c
        include/ComputeTopologyInfo.h
        src/ComputeTopologyInfo.c
        include/performance_measurement.h
        src/performance_measurement.c

        memory/include/ComputePairwiseDistanceMMMAnastasia.h
        memory/performance_improvement/ComputePairwiseDistanceMMMAnastasia.c

        unrolled/include/ComputeKDistanceAllUnrolled.h
        unrolled/performance_improvement/ComputeKDistanceAllUnrolled.c
        unrolled/include/ComputeKDistanceNeighborhoodUnrolled.h
        unrolled/performance_improvement/ComputeKDistanceNeighborhoodUnrolled.c
        unrolled/include/ComputeKDistanceObjectUnrolled.h
        unrolled/performance_improvement/ComputeKDistanceObjectUnrolled.c
        unrolled/include/ComputeLocalOutlierFactorUnrolled.h
        unrolled/performance_improvement/ComputeLocalOutlierFactorUnrolled.c
        unrolled/include/ComputeLocalReachabilityDensityMerged_PointUnrolled.h
        unrolled/performance_improvement/ComputeLocalReachabilityDensityMerged_PointUnrolled.c
        unrolled/include/ComputeLocalReachabilityDensityMergedUnrolled.h
        unrolled/performance_improvement/ComputeLocalReachabilityDensityMergedUnrolled.c
        unrolled/include/ComputePairwiseDistancesMMMUnroll.h
        unrolled/performance_improvement/ComputePairwiseDistancesMMMUnroll.c
        unrolled/include/KNN.h
        unrolled/performance_improvement/KNN.c
        unrolled/include/MMM.h
        unrolled/performance_improvement/MMM.c
        unrolled/performance_improvement/FasterMetrics.c

        avx/include/AVX_utils.h
        avx/src/AVX_utils.c
        avx/performance_improvement/AVX_lattice.c
        avx/include/AVXMetrics.h
        avx/performance_improvement/AVX_metrics.c
        avx/include/AVXTopoInfo.h
        avx/performance_improvement/AVXTopoInfo.c
        avx/include/ComputeLocalOutlierFactorAVX.h
        avx/performance_improvement/ComputeLocalOutlierFactorAVX.c
        avx/include/ComputeLocalReachabilityDensityMerged_PointAVX.h
        avx/performance_improvement/ComputeLocalReachabilityDensityMerged_PointAVX.c
        avx/include/ComputeLocalReachabilityDensityMergedAVX.h
        avx/performance_improvement/ComputeLocalReachabilityDensityMergedAVX.c
        avx/include/ComputePairwiseDistanceMMMAvx.h
        avx/performance_improvement/ComputePairwiseDistanceMMMAvx.c
        avx/include/MMMAvx.h
        avx/performance_improvement/MMMAvx.c)

#target_link_libraries(baseline m)
#target_link_libraries(unrolled m)
#target_link_libraries(avx m)
#target_link_libraries(memory m)
target_link_libraries(final_benchmark m)
