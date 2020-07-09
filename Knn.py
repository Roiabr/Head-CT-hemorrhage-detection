from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pyemd.emd import emd



# k-NN classifier for image classification
def knn(trainX, trainY, testX, testY, numNeigh = 3):
    # n_jobs means number of parallel jobs to run. -1 meansusing all processors
    model = KNeighborsClassifier(n_neighbors=numNeigh, n_jobs=-1)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY)
    print("raw pixel accuracy: {:.2f}%".format(acc * 100))

def knn_emd(trainX, trainY, testX, testY, numNeigh = 3):
    test_results = knn_earthMover(trainX, trainY, testX, testY)
    return accuracy(test_results, testY)


def knn_earthMover(trainX, trainY, testX, testY, numNeigh = 3):
    # label each new point by the k nearest neighbor
    test_results = np.empty([testX.shape[0]])
    for point in range(testX.shape[0]):
        k_nearest_dist = np.full((numNeigh), np.inf) # array of distances to the nearest neighbors
        k_labels = np.empty([numNeigh]) # The labels of the nearest neighbors
        for index in range(trainX.shape[0]):
            # compute the difference between the test-point to the base-point by lp distance:
            dif = emd(testX[point], trainX[index])
            # see if it is closer than a neighbor already exists:
            if dif < np.amax(k_nearest_dist):
                # put the shorter distance into in the max distance place
                current_max_index = np.argmax(k_nearest_dist)
                k_nearest_dist[current_max_index] = dif
                k_labels[current_max_index] = trainY[index]
        # determine the label of the new point as voted by most neighbors
        vote = np.sum(k_labels)
        if vote < 0:
            test_results[point] = -1
        else:
            test_results[point] = 1
    return test_results


def error(test_pred, test_true_labels):
    error_test = 0.0
    number_of_points = test_true_labels.shape[0]
    for point in range(number_of_points):
        error_test += int(test_pred[point] * test_true_labels[point] < 0)
    return error_test/number_of_points

def accuracy(test_pred, test_true_labels):
    return 100 - error(test_pred, test_true_labels)