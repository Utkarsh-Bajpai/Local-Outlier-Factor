//
// Created by Utkarsh Bajpai on 08.05.20.
//

#include <stdlib.h>

#include "../../include/lof_baseline.h"
#include "../../include/utils.h"
#include "math.h"
#include "../include/ComputeKDistanceObjectUnrolled.h"
#include "../include/ComputeKDistanceAllUnrolled.h"
#include "../include/Algorithm.h"

// -------------------------------------------------------------------------------------------------------------------------------------------

double
ComputeKDistanceAllUnroll_Fastest(int k, int num_pts, const double* distances_indexed_ptr, double* k_distances_indexed_ptr) {
    /** Implement Definition 3 from the paper i.e. for each point it's distance to the farthest k'th neighbor
   * @param obj_idx: index of the object
   *
   * @return: fill k_distances_indexed_ptr
  */
    int idx;
    for (idx = 0; idx + 5 < num_pts; idx += 5) {
        ComputeKDistanceObject(idx, k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
        ComputeKDistanceObject(idx + 1, k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
        ComputeKDistanceObject(idx + 2, k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
        ComputeKDistanceObject(idx + 3, k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
        ComputeKDistanceObject(idx + 4, k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);

    }
    if (idx != num_pts) {
        for (; idx < num_pts; idx++) {
            ComputeKDistanceObjectFaster(idx, k, num_pts, distances_indexed_ptr, k_distances_indexed_ptr);
        }
    }
    return num_pts * (num_pts - 1.0) * log(num_pts);
} // ComputeKDistanceAll
