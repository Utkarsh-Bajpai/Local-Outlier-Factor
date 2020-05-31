//
// Created by Sycheva  Anastasia on 15.05.20.
//

#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"
#include "../include/file_utils.h"
#include "../include/Algorithm.h"
#include "../include/performance_measurement.h"

#include "include/ComputePairwiseDistance.h"
#include "include/ComputeLocalReachabilityDensityMerged_Pipeline2.h"


int main() {
    // MEMORY
    // TO BE CHANGED!
    printf("not only is my short term memory is horrible, but so is my short term memory\n");

    // ASY ---------------------------------------------------------->
    //cache_block_selection( 2000, 10, 10 );
    //printf("Start computation\n");
    //cache_block_selection( 5000, 10, 100 );

    int num_pts = 50000, k = 200;
    verify_lrdm2_function( num_pts, k, ComputeLocalReachabilityDensityMerged_Pipeline2_Recursive );

    printf("Original function\n");
    performance_measure_fast_for_lrdm_pipeline2( num_pts, k, 100,
                                                 ComputeLocalReachabilityDensityMerged_Pipeline2);

    printf("Recursive function\n");
    performance_measure_fast_for_lrdm_pipeline2( num_pts, k, 100,
                                                 ComputeLocalReachabilityDensityMerged_Pipeline2_Recursive);


    /*
    printf("\n\nTEST MINI MMM  ********************************\n");
    test_mini_mmm_lower_triangular_driver( mini_mmm_lower_triangular,  mmm_baseline);
    //test_mini_mmm_lower_triangular_driver( mini_mmm_full,  baseline_full);

    // STILL FAILS SOME
    printf("\n\n\nTEST MICRO MMM  ********************************\n");
    test_micro_mmm_triangular_driver();
     */

}


