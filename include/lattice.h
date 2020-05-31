//
// Created by fvasluia on 5/3/20.
//

#ifndef FASTLOF_LATTICE_H
#define FASTLOF_LATTICE_H


typedef struct t_cell {
    int current_size;
    int current_index;
    int* point_ids;
}LATTICE_CELL;

typedef struct vec_cell {
    int num_dimensions;
    double* axis_step;
    double* min_range;
    int resolution;
    LATTICE_CELL* cell_Arr;
}MULTI_LATTICE;

MULTI_LATTICE BuildLattice(int num_dim, int split_dim,  int resolution, double* min_range, double* max_range);
void InsertElement(MULTI_LATTICE* lattice, double* point, int point_idx, int split_dim, int num_dims);
int ComputeCellIndex(MULTI_LATTICE* lattice, double* point, int split_dims, int num_dims);

// vectorized implementation in the avx dir
int AVXComputeCellIndex(MULTI_LATTICE* lattice, double* point, int split_dims, int num_dims);
int AVXComputeCellIndexWithOutput(MULTI_LATTICE* lattice, double* point, double* cell_locations, int split_dims, int num_dims);
void test_idx_comp(int num_pts, int k); //testbench
// ---------------------------------------------
#endif //FASTLOF_LATTICE_H
