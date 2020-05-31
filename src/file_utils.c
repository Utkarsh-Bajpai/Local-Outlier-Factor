//
// Created by pasca on 03.05.2020.
//

#include <stdlib.h>
#include "../include/file_utils.h"
#include "../include/utils.h"
#include "string.h"


#include <stdlib.h>
#include "../include/utils.h"
#include "string.h"
#include "../include/lattice.h"


double* Load2DDataFromFile(FILE* placeholder_file, int* nr_points, int* nr_neighs) {
    char line[50];

    if (placeholder_file != NULL) {
        fgets(line, sizeof(line), placeholder_file);
        sscanf(line, "%d,%d\n", nr_points, nr_neighs);

        double* points_set_ptr = XmallocMatrixDouble(*nr_points, 2);

        // get rid of the file header
        fgets(line, sizeof(line), placeholder_file);

        int id;
        double x, y;
        for (int i = 0; i < *nr_points; ++i) {
            fgets(line, sizeof(line), placeholder_file);
            sscanf(line, "%d,%lf,%lf\n", &id, &x, &y);
            points_set_ptr[2 * i] = x;
            points_set_ptr[2 * i + 1] = y;
        }
        return points_set_ptr;
    } else {
        return NULL;
    }
}

double* LoadResultsFromFile(FILE* placeholder_file, int nr_points) {
    char line[25];
    if (placeholder_file != NULL) {
//        fgets(line, sizeof(line), placeholder_file);
//        sscanf(line, "%d\n", &nr_points);
//
//        fgets(line, sizeof(line), placeholder_file);
        double* score_point_set_ptr = XmallocVectorDouble(nr_points);
        int id;
        double score;
        for (int i = 0; i < nr_points; ++i) {
            fgets(line, sizeof(line), placeholder_file);
            sscanf(line, "%d,%lf\n", &id, &score);
            score_point_set_ptr[i] = score;
        }

        // just in case..to be sure the file is successfully parsed
//        printf("Last LOF %lf\n", score_point_set_ptr[nr_points - 1]);
        return score_point_set_ptr;
    } else {
        return NULL;
    }

}

void parseDataLine(char* line, double* output_ptr) {
    int dim_line = strlen(line);
    int idx = 0;
    int out_idx = 0;
    char num_buffer[30];
    int idx_buff = 0;
    while (idx < dim_line) {
        if (((line[idx] <= '9') && (line[idx] >= '0')) || (line[idx] == '.') || (line[idx] == '-')) {
            num_buffer[idx_buff++] = line[idx];
        } else {
            if (line[idx] == ',' || line[idx] == '\n') {
                num_buffer[idx_buff] = '\0';
                double nr = atof(num_buffer);
                output_ptr[out_idx++] = nr;
                idx_buff = 0;
            }
        }
        ++idx;
    }

}

// take care that the pointers for the range are to be given as NULL, they are used just for reference output
double* LoadGenericDataFromFile(FILE* placeholder_file, FILE* placeholder_meta_file, double* min_range_ptr,
                                double* max_range_ptr, int* nr_points, int* dim, int* nr_neighs) {

    char line[100];

    if (placeholder_meta_file != NULL) {

        if (placeholder_file != NULL) {
            // get rid of the header
            fgets(line, sizeof(line), placeholder_meta_file);
            // read the given sizes
            fgets(line, sizeof(line), placeholder_meta_file);
            sscanf(line, "%d,%d,%d\n", nr_points, dim, nr_neighs);

            // allocate memory for the data regarding the range
            min_range_ptr = XmallocVectorDouble(*dim);
            max_range_ptr = XmallocVectorDouble(*dim);

            char* long_line_ptr = NULL;
            size_t long_line_sz = 0;
            // get rid of the header
            fgets(line, sizeof(line), placeholder_meta_file);
            // read the min_range information
            getline(&long_line_ptr, &long_line_sz, placeholder_meta_file);
            // parse the min_range information
            parseDataLine(long_line_ptr, min_range_ptr);

            // get rid of the header
            fgets(line, sizeof(line), placeholder_meta_file);
            // read the max_range_information
            getline(&long_line_ptr, &long_line_sz, placeholder_meta_file);
            // parse the max range information
            parseDataLine(long_line_ptr, max_range_ptr);

            double* points_set_ptr = XmallocMatrixDouble(*nr_points, *dim);

            for (int i = 0; i < *nr_points; ++i) {
                getline(&long_line_ptr, &long_line_sz, placeholder_file);
                parseDataLine(long_line_ptr,
                              points_set_ptr + i * (*dim)); // put some brackets to those with pythonic dreams about **
            }
            free(long_line_ptr);
            return points_set_ptr;
        } else {
            fprintf(stderr, "Null pointer given as placeholder for the data file. Exiting...");
            return NULL;
        }

    } else {
        fprintf(stderr, "NULL pointer given as the placeholder for the meta file. Exiting...");
        return NULL;
    }
}

// same method as before, but this builds the lattice for the localization info
double* LoadTopologyInfo(FILE* placeholder_file, FILE* placeholder_meta_file, MULTI_LATTICE* lattice,
                         double* min_range_ptr, double* max_range_ptr, int resolution, int* nr_points,
                         int* dim, int* nr_neighs, int num_splits) {

    char line[100];

    if (placeholder_meta_file != NULL) {

        if (placeholder_file != NULL) {
            // get rid of the header
            fgets(line, sizeof(line), placeholder_meta_file);
            // read the given sizes
            fgets(line, sizeof(line), placeholder_meta_file);
            sscanf(line, "%d,%d,%d\n", nr_points, dim, nr_neighs);

            // allocate memory for the data regarding the range
            min_range_ptr = XmallocVectorDouble(*dim);
            max_range_ptr = XmallocVectorDouble(*dim);

            char* long_line_ptr = NULL;
            size_t long_line_sz = 0;
            // get rid of the header
            fgets(line, sizeof(line), placeholder_meta_file);
            // read the min_range information
            getline(&long_line_ptr, &long_line_sz, placeholder_meta_file);
            // parse the min_range information
            parseDataLine(long_line_ptr, min_range_ptr);

            // get rid of the header
            fgets(line, sizeof(line), placeholder_meta_file);
            // read the max_range_information
            getline(&long_line_ptr, &long_line_sz, placeholder_meta_file);
            // parse the max range information
            parseDataLine(long_line_ptr, max_range_ptr);
            *lattice = BuildLattice(*dim, num_splits, resolution, min_range_ptr, max_range_ptr);

            double* points_set_ptr = XmallocMatrixDouble(*nr_points, *dim);
            int l_nr_pts = *nr_points; int l_dim = *dim;
            for (int i = 0; i < l_nr_pts; ++i) {
                getline(&long_line_ptr, &long_line_sz, placeholder_file);
                parseDataLine(long_line_ptr,
                              points_set_ptr + i * l_dim);

                InsertElement(lattice, points_set_ptr + i * l_dim, i, num_splits, l_dim);
            }
            free(long_line_ptr);
            return points_set_ptr;
        } else {
            fprintf(stderr, "Null pointer given as placeholder for the data file. Exiting...");
            return NULL;
        }

    } else {
        fprintf(stderr, "NULL pointer given as the placeholder for the meta file. Exiting...");
        return NULL;
    }
}

int* LoadNeighIndResultsFromFile(FILE* pFile, int num_points, int k) {
    int id;
    int* neigh_ind_results_ptr = XmallocMatrixInt(num_points, k);
    for (int i = 0; i < num_points; i++) {
        fscanf(pFile, "%d", &id);
        for (int j = 0; j < k; j++) {
            fscanf(pFile, ",%d", &neigh_ind_results_ptr[i * k + j]);
        }
    }
//    printf("Last Neigh Ind %d\n", neigh_ind_results_ptr[num_points * k - 1]);
    return neigh_ind_results_ptr;

}

double* LoadNeighDistResultsFromFile(FILE* pFile, int num_points, int k) {
    int id;
    double* neigh_dist_results_ptr = XmallocMatrixDouble(num_points, k);
    for (int i = 0; i < num_points; i++) {
        fscanf(pFile, "%d", &id);
        for (int j = 0; j < k; j++) {
            fscanf(pFile, ",%lf", &neigh_dist_results_ptr[i * k + j]);
        }
    }
//    printf("Last Neigh Dist %lf\n", neigh_dist_results_ptr[num_points * k - 1]);
    return neigh_dist_results_ptr;

}

double* LoadLRDResultsFromFile(FILE* pFile, int num_points) {
    double* lrd_true = XmallocVectorDouble(num_points);
    int id;
    for (int i = 0; i < num_points; i++) {
        fscanf(pFile, "%d,%lf\n", &id, &lrd_true[i]);
    }
    return lrd_true;
}

FILE* open_with_error_check(const char* file_name, const char* mode) {

    char full_file[50] = "data/";
    strcat(full_file, file_name);


    FILE* input_file = fopen(full_file, mode);
    if (input_file == NULL) {
        fprintf(stderr, "Error at opening %s", full_file);
        printf("Check if the file was generated or the path in open_with_error_check\n");
        exit(-1);
    }
    return input_file;
}