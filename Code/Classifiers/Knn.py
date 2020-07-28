from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wasserstein_distance
import Code.Draw


def knn(trainX, trainY, testX, testY, images, index, numNeigh=3):
    # n_jobs means number of parallel jobs to run. -1 meansusing all processors
    model = KNeighborsClassifier(n_neighbors=numNeigh, n_jobs=-1)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY) * 100
    Code.Draw.drawPredict(model, testX, testY, images, index)
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
    Code.Draw.drawPredict(model, testX, testY, images, index)
    return acc
