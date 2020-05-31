#include <stdio.h>
#include <immintrin.h>
#include "stdlib.h"

#include "../../include/Algorithm.h"
#include "../../include/utils.h"
#include "../../include/tests.h"
#include "../../include/tsc_x86.h"
#include "../../include/lof_baseline.h"
#include "../../include/performance_measurement.h"

/**
* OBSERVATIONS
*            1) LRD_SCORE_TABLE_PTR is accessed multiple times
 *           2) Reversing the order seems to be suboptimal since
 *           would need to store the intermediate results of computations -> increase memory footprint anyway
 *
 * POTENTIAL IMPROVEMENT
 *           1) Change the order in which the points are processed, i.e. nieghbors are processed
 *           one after the other
 *
*/
//------------------------------------------------------------------------------------ <Memory improvements>


// ------------------------------------------------------------------------------------ </Memory improvements>

#define NUM_FUNCTIONS_LOF_MEM 1
int lof_driver_memory(int k) {
    /**
     * TODO: 1. add simple version, without box plots -> take from Razvan
    */


    my_lof_fnc* fun_array_lof = (my_lof_fnc*) calloc(NUM_FUNCTIONS_LOF_MEM, sizeof(my_lof_fnc));
    fun_array_lof[0] = &ComputeLocalOutlierFactor;
    // fun_array_lof[1] = &ComputeLocalOutlierFactor_Avx_1;

    char* fun_names[NUM_FUNCTIONS_LOF_MEM] = {"baseline"};

    return 1;
}



