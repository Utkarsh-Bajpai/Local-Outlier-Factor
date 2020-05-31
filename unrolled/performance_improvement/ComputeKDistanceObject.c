//
// Created by Utkarsh Bajpai on 08.05.20.
//

# include <stdlib.h>
# include "../include/Algorithm.h"


void ComputeKDistanceObjectFaster(int obj_idx, int k, int num_pts, const double* distances_indexed_ptr,
                                  double* k_distances_indexed_ptr) {
    /**
     * Implement Definition 3 from the paper
     * @param obj_idx: index of the object
     * @return: fill k_distances_indexed_ptr
    **/

    int i;
    double* dist_to_obj = XmallocVectorDouble(num_pts - 1); // collect distances to obj_idx

    for (i = 0; (i + 5) < obj_idx; i += 5) {
        // distances_indexed_ptr[i * n  + j] = dist(p_i, p_j) for i < j
        dist_to_obj[i] = distances_indexed_ptr[i * num_pts + obj_idx]; // check this later !!!
        dist_to_obj[i + 1] = distances_indexed_ptr[(i + 1) * num_pts + obj_idx];
        dist_to_obj[i + 2] = distances_indexed_ptr[(i + 2) * num_pts + obj_idx];
        dist_to_obj[i + 3] = distances_indexed_ptr[(i + 3) * num_pts + obj_idx];
        dist_to_obj[i + 4] = distances_indexed_ptr[(i + 4) * num_pts + obj_idx];
    }
    if (i != obj_idx) {
        for (; i < obj_idx; i++) {
            // distances_indexed_ptr[i * n  + j] = dist(p_i, p_j) for i < j
            dist_to_obj[i] = distances_indexed_ptr[i * num_pts + obj_idx]; // check this later !!!
        }
    }

    int obj_pts = obj_idx * num_pts;

    for (i = obj_idx + 1; (i + 5) < num_pts; i += 5) {
        dist_to_obj[i - 1] = distances_indexed_ptr[obj_pts + i]; // check this later !!!
        dist_to_obj[(i + 1) - 1] = distances_indexed_ptr[obj_pts + i + 1];
        dist_to_obj[(i + 2) - 1] = distances_indexed_ptr[obj_pts + i + 2];
        dist_to_obj[(i + 3) - 1] = distances_indexed_ptr[obj_pts + i + 3];
        dist_to_obj[(i + 4) - 1] = distances_indexed_ptr[obj_pts + i + 4];
    }
    if (i != num_pts) {
        for (; i < num_pts; i++) {
            dist_to_obj[i - 1] = distances_indexed_ptr[obj_pts + i];
        }
    }

    qsort(dist_to_obj, num_pts - 1, sizeof(double), compare_double);
    // (1) for at least k objects o' dist(obj_idx, o') <= dist(obj_idx, o)
    // (2) for at most k-1 objects o' dist(obj_idx, o') < dist(obj_idx, o)
    // SPECIAL CASE: dist_to_obj[k] == dist_to_obj[k-1] => (2) does not hold => careful !
    while (dist_to_obj[k] == dist_to_obj[k - 1] && k > 1) {
        k = k - 1;
    }
    k_distances_indexed_ptr[obj_idx] = dist_to_obj[k - 1];
}