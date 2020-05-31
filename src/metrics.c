//
// Created by fvasluia on 2/28/20.
// List of other popular distance metrics:
// https://scikit-learn.org/stable/search.html?q=distance
//

#include "math.h"
#include "../include/metrics.h"

double ComputeFlopsEuclidianDistance(int dim) {
    return 4 * dim + 1;
}
double EuclideanDistance(const double *v1_ptr, const double *v2_ptr, int n_dim) {
    double dist = 0;
    for (int i = 0; i < n_dim; ++i) {
        dist += (v1_ptr[i] - v2_ptr[i]) * (v1_ptr[i] - v2_ptr[i]);
    }
    return sqrt(dist);
}


double ComputeFlopsCosineSimilarity(int dim) {
    return 6 * dim + 5;
}
double CosineSimilarity(const double *v1_ptr, const double *v2_ptr, int n_dim) {
    /**
    * implementation of cosine similarity
    * see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html#scipy.spatial.distance.cosine
    */

    double dist, denom, numerator = 0;
    double x_2 = 1;
    double y_2 = 1;

    for (int i = 0; i < n_dim; ++i) {
        numerator += v1_ptr[i] * v2_ptr[i];
        x_2 *= v1_ptr[i];
        y_2 *= v2_ptr[i];
    }
    x_2 = sqrt(x_2);
    y_2 = sqrt(y_2);
    denom = x_2 * y_2;

    dist = 1 - numerator/denom;

    return dist;

}


