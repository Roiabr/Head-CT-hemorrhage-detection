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
    acc = model.score(testX, testY) * 100
    print("---------------------KNN--------------------------")
    # Draw.drawPredict(model, testX, testY, images, index)
    print("Knn: raw pixel accuracy: {:.2f}%".format(acc))
    return acc



# Earth mover:
def EMD(x, y):
    return wasserstein_distance(x, y)


def knnEMD(trainX, trainY, testX, testY, images, index, numNeigh=3):
    # n_jobs means number of parallel jobs to run. -1 meansusing all processors
    model = KNeighborsClassifier(n_neighbors=numNeigh, algorithm='ball_tree',
                                 metric=EMD, metric_params=None, n_jobs=-1)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY) * 100
    print("---------------------KNN with Earth Mover--------------------------")
    # Draw.drawPredict(model, testX, testY, images, index)
    print("KnnEMD: raw pixel accuracy: {:.2f}%".format(acc))
    return acc

