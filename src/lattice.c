//
// Created by fvasluia on 5/3/20.
//
#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include "../include/lattice.h"
#include "../include/utils.h"

#define CELL_SIZE 13

MULTI_LATTICE BuildLattice(int num_dim, int split_dim,  int resolution, double* min_range, double* max_range ){
    MULTI_LATTICE lattice;
    lattice.num_dimensions = split_dim;
    lattice.resolution = resolution;
    lattice.axis_step = XmallocVectorDouble(split_dim);
    lattice.min_range = min_range;
    for(int i=0; i < split_dim; ++i) {
        lattice.axis_step[i] = (max_range[i] - min_range[i])/resolution;
    }
    int max_cells = pow(resolution, split_dim);
    lattice.cell_Arr = (LATTICE_CELL *)malloc(max_cells * sizeof(LATTICE_CELL));
    for(int i = 0; i < max_cells; ++i) {
        LATTICE_CELL cell;
        cell.current_size = CELL_SIZE;
        cell.current_index = 0,
        cell.point_ids = (int*)malloc(CELL_SIZE * sizeof(int));
        lattice.cell_Arr[i] = cell;
    }

    return lattice;
}
int ComputeCellIndex(MULTI_LATTICE* lattice, double* point, int split_dims, int num_dims){
    int multiplier = 1;
    int index = 0;
    for(int i=0; i < split_dims; ++i) {
        int dim_cell = ceil((point[i] - lattice->min_range[i])/lattice->axis_step[i]) - 1;
        if(dim_cell <0) dim_cell = 0;
        // printf("%d \n", dim_cell);
        index += multiplier * dim_cell;
        multiplier *= lattice->resolution;
    }
    return index;
}

void InsertElement(MULTI_LATTICE* lattice, double* point, int point_idx, int split_dims,  int num_dims) {
    int index = ComputeCellIndex(lattice, point, split_dims, num_dims);
    if(lattice->cell_Arr[index].current_index < lattice->cell_Arr[index].current_size) {
        lattice->cell_Arr[index].point_ids[lattice->cell_Arr[index].current_index++] = point_idx;
    } else {
        // rehashing by expanding the size of the current cell
        int* new_arr = (int*)malloc(2 * lattice->cell_Arr[index].current_size * sizeof(int));
        for(int i = 0; i < lattice->cell_Arr[index].current_index; ++i) {
            new_arr[i] = lattice->cell_Arr[index].point_ids[i];
        }
        // free the old pointer to avoid leaks
        free(lattice->cell_Arr[index].point_ids);
        // now link the new double size pointer
        lattice->cell_Arr[index].point_ids = new_arr;
        lattice->cell_Arr[index].current_size *= 2;
        lattice->cell_Arr[index].point_ids[lattice->cell_Arr[index].current_index++] = point_idx;
    }

}