from Cnn import cnnModel
import Adaboost
import Random_Forest
import Svm
from DecisionTreeClassifier import decisionTreeClassifier
from Knn import knn, knnEMD
import Extract as extract
import numpy as np


def splitTestTrain(X, Y):
    trainSize = (int)(0.8 * X.shape[0])
    Y = np.reshape(Y, (Y.shape[0], 1))
    indexes = np.arange(200)
    indexes = np.reshape(indexes, (200, 1))
    # to concatenate the data features with the labels
    # labels now is data[:, -1]
    data = np.concatenate((X, Y, indexes), axis=1)
    np.random.shuffle(data)
    trainX = data[: trainSize, :-2]
    trainY = data[: trainSize, -2]
    testX = data[trainSize:, :-2]
    testY = data[trainSize:, -2]
    imagesTest = data[trainSize:, -1]
    return trainX, trainY, testX, testY, imagesTest


if __name__ == '__main__':
    pathX = "head_ct/*.png"
    pathY = 'labels.csv'
    X, Y, images = extract.extract_features(pathX, pathY)
    trainX, trainY, testX, testY, testIm = splitTestTrain(X, Y)
    knnEMD(trainX, trainY, testX, testY,images, testIm, numNeigh=2)
    knn(trainX, trainY, testX, testY,images, testIm, numNeigh=2)
    Svm.svm(trainX, trainY, testX, testY,images, testIm)
    Random_Forest.random_forest(trainX, trainY, testX, testY,images, testIm)
    decisionTreeClassifier(trainX, trainY, testX, testY, images, testIm)
    Adaboost.adaBoost(trainX, trainY, testX, testY, images, testIm)
    cnnModel(320, 320, pathX, pathY)
