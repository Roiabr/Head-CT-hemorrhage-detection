from pyemd.emd import emd_samples
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wasserstein_distance
import numpy as np
from pyemd import emd
import Draw


def knn(trainX, trainY, testX, testY, images, index, numNeigh=3):
    # n_jobs means number of parallel jobs to run. -1 meansusing all processors
    model = KNeighborsClassifier(n_neighbors=numNeigh, n_jobs=-1)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY)

    print("---------------------KNN--------------------------")

   # Draw.drawPredict(model, testX, testY, images, index)

    print("Knn: raw pixel accuracy: {:.2f}%".format(acc * 100))


# Earth mover:
def EMD(x, y):
    first_histogram, bin_edges_first = np.histogram(x, bins = np.arange(250))
    second_histogram, bin_edges_second = np.histogram(y, bins=np.arange(250))
    return wasserstein_distance(first_histogram, second_histogram)


def knnEMD(trainX, trainY, testX, testY, images, index, numNeigh=3):
    # n_jobs means number of parallel jobs to run. -1 meansusing all processors
    model = KNeighborsClassifier(n_neighbors=numNeigh, algorithm='ball_tree',
                                 metric=EMD, metric_params=None, n_jobs=-1)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY)

    print("---------------------KNN with Earth Mover--------------------------")

   # Draw.drawPredict(model, testX, testY, images, index)

    print("KnnEMD: raw pixel accuracy: {:.2f}%".format(acc * 100))

