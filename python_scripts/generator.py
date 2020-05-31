import sys
from time import sleep

import numpy, os
import time

from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor

"""
Following https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py 

USAGE: 

// dataset with 100 datapoints, 4 clusters and 5 neighpors
// Q: what combinations should we try? 

n = 100, 500, 1000, 2000
$ python python_scripts/generator.py 4 100 5 2
$ python python_scripts/generator.py 4 500 5 2
$ python python_scripts/generator.py 4 1000 5 2
$ python python_scripts/generator.py 4 2000 5 2
$ python python_scripts/generator.py 4 3000 5 2


TEST:
$ python python_scripts/generator.py 4 3000 5 4
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def make_dataset(nr_clusters, cluster_stds, nr_samples, nr_features):
    """
    :param nr_clusters: Nr clusters for the dataset
    :param cluster_stds: Stds for the clusters, given as array [std1,std2,...,std3] -> No Covariance
    :param nr_samples: Nr of points to generate in the dataset
    :param nr_features: Nr of features for a point, should stick to 2 currently
    :return: dataset with nr_samples points drawn from n Gaussian Distributions with given parameters

    NOTE: important variables for performance plots:
        1. nr_neighbors, nr_samples, nr_features
        2. currentlly keep nr_features = 2

    TODO: add optional argument that indicates whether to produce intermediate datasets
    """
    dataset = make_blobs(centers=nr_clusters, cluster_std=cluster_stds, n_samples=nr_samples, n_features=nr_features)[0]
    return dataset.astype(numpy.double)


if __name__ == '__main__':
    numpy.random.RandomState(42)

    if len(sys.argv) != 5:
        print("Give argument for nr_clusters, nr_samples, K, dim")
        exit(-1)
    else:
        nr_clusters = int(sys.argv[1])
        nr_samples = int(sys.argv[2])
        nr_neighbors = int(sys.argv[3])
        nr_features = int(sys.argv[4])
        # file_name = sys.argv[4]128
        cluster_stds = numpy.random.uniform(low=0, high=10, size=[nr_clusters, nr_features]).astype(numpy.float64)

    dataset = make_dataset(nr_clusters, cluster_stds, nr_samples, nr_features)
    assert dataset.shape == (nr_samples, nr_features), "dimension mismatch"

    clf = LocalOutlierFactor(n_neighbors=nr_neighbors, algorithm="brute", metric="euclidean")
    start_time = time.clock()
    for i in range(0, 5):
        clf.fit_predict(dataset)
    exec_time = time.clock() - start_time
    print("--- %s seconds ---" % (exec_time / 5))

    with open("Execution_time.txt", "a") as f:
        f.write("{},{},{},{}\n".format(nr_samples, nr_features, nr_neighbors, exec_time / 5))

    neigh_dist, neigh_ind = clf.kneighbors()

    assert os.getcwd().split("/")[-1] == "team025", "wrighting to the wrong directory, run from team025 root"
    dir_name = "./data/n{}_k{}_dim{}".format(int(nr_samples), int(nr_neighbors), int(nr_features))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print("Writing to directory: {}".format(dir_name))
    with open("{}/metadata.txt".format(dir_name), "w+") as f:
        f.write("Nr points, dim, nr neigh\n")
        f.write("{},{},{}\n".format(nr_samples, nr_features, nr_neighbors))
        f.write("Min per dimension\n")
        s = ', '.join(str(dim) for dim in numpy.min(dataset, axis=0))
        f.write(s)
        f.write("\nMax per dimension\n")
        s = ', '.join(str(dim) for dim in numpy.max(dataset, axis=0))
        f.write(s)
        f.flush()

    with open("{}/dataset.txt".format(dir_name), "w+") as f:
        for point in dataset:
            # NOTE: CURRENTLY ONLY SUPPORT DIM = 2
            s = ', '.join(str(coord) for coord in point)
            f.write(s + "\n")
        f.flush()

    with open("{}/neigh_dist_results.txt".format(dir_name), "w+") as f:
        for i, dist_array in enumerate(neigh_dist):
            f.write("{}".format(i))
            for dist in dist_array:
                f.write(",{}".format(dist))
            f.write("\n")
        f.flush()

    with open("{}/lrd_results.txt".format(dir_name), "w+") as f:
        for i, val in enumerate(clf._lrd):
            f.write("{},{}\n".format(i, val))
        f.flush()

    with open("{}/neigh_ind_results.txt".format(dir_name), "w+") as f:
        for i, neigh_array in enumerate(neigh_ind):
            f.write("{}".format(i))
            for neigh in neigh_array:
                f.write(",{}".format(neigh))
            f.write("\n")
        f.flush()

    with open("{}/lof_results.txt".format(dir_name), "w+") as f:
        for id, score in enumerate(-clf.negative_outlier_factor_.astype(numpy.float64)):
            f.write("{},{}\n".format(id, score))
        f.flush()

    sleep(1)
