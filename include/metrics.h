//
// Created by fvasluia on 2/28/20.
// List of other popular distance metrics:
// https://scikit-learn.org/stable/search.html?q=distance
//

#ifndef FASTLOF_METRICS_H
#define FASTLOF_METRICS_H

// typedef double (* my_metrics_fnc) (const double *, const double *, int);

double ComputeFlopsCosineSimilarity(int dim);
double ComputeFlopsEuclidianDistance(int dim);

double EuclideanDistance(const double *v1_ptr, const double *v2_ptr, int n_dim); // I don't know what is the reason for deleting this
double UnrolledEuclideanDistance(const double *v1_ptr, const double *v2_ptr, int n_dim);
double CosineSimilarityFaster1(const double *v1_ptr, const double *v2_ptr, int dim);

double performance_measure_hot_for_metrics(int dim, double (* fct)(const double* , const double* , int), 
                                                    double (* fct_correct)(const double* , const double* , int), 
                                                    int to_verify, int verobse, int compare_with_baseline);

void performance_plot_metrics_to_file( int max_dim, int step,
                                       const char* name, const char* mode,
                                       double (* fct)(const double* , const double* , int),
                                       double (* fct_correct)(const double* , const double* , int) );

int exec_metrics();
#endif //FASTLOF_METRICS_H
