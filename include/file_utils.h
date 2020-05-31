//
// Created by pasca on 03.05.2020.
//
#include <stdio.h>
#include "../include/lattice.h"

#ifndef FASTLOF_FILE_UTILS_H
#define FASTLOF_FILE_UTILS_H

double* Load2DDataFromFile(FILE* placeholder_file, int* nr_points, int* nr_neighs);

double* LoadResultsFromFile(FILE* placeholder_file, int num_points);

void parseDataLine(char* line, double* output_ptr);

// use the meta file to load the information about the range along each axis
double* LoadGenericDataFromFile(FILE* placeholder_file, FILE* placeholder_meta_file, double* min_range_ptr,
                                double* max_range_ptr, int* nr_points, int* dim, int* nr_neighs);

// build the multi dimensional lattice as well
double* LoadTopologyInfo(FILE* placeholder_file, FILE* placeholder_meta_file, MULTI_LATTICE* lattice,
                         double* min_range_ptr, double* max_range_ptr, int resolution, int* nr_points,
                         int* dim, int* nr_neighs, int num_splits);

int* LoadNeighIndResultsFromFile(FILE* pFile, int num_points, int k);

double* LoadNeighDistResultsFromFile(FILE* pFile, int num_points, int k);

double* LoadLRDResultsFromFile(FILE* pFile, int num_points);

FILE* open_with_error_check(const char* file_name, const char* mode);


#endif //FASTLOF_FILE_UTILS_H
