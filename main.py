from Cnn import cnnModel
import Adaboost
import Random_Forest
from Svm import svmlinear, svmpoly, svmrbf, svmsigmoid
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
    method_to_extract_features = 'simple'
    X, Y, images = extract.extract_features(pathX, pathY, method_to_extract_features)
    trainX, trainY, testX, testY, testIm = splitTestTrain(X, Y)

    print('Begins testing the models...')
    print('Extract features method:', method_to_extract_features)

    results = np.zeros((9,1))
    nb_iteration = 10
    for epoch in range(nb_iteration):
        print('epoch number:', epoch)
        results[0] += knnEMD(trainX, trainY, testX, testY,images, testIm, numNeigh=2)
        results[1] += knn(trainX, trainY, testX, testY,images, testIm, numNeigh=2)
        results[2] += svmlinear(trainX, trainY, testX, testY,images, testIm)
        results[3] += svmpoly(trainX, trainY, testX, testY,images, testIm)
        results[4] += svmrbf(trainX, trainY, testX, testY,images, testIm)
        results[5] += svmsigmoid(trainX, trainY, testX, testY,images, testIm)
        results[6] += Random_Forest.random_forest(trainX, trainY, testX, testY,images, testIm)
        results[7] += decisionTreeClassifier(trainX, trainY, testX, testY, images, testIm)
        results[8] += Adaboost.adaBoost(trainX, trainY, testX, testY, images, testIm)

    results = np.average(results, axis=1) / nb_iteration
    print()
    print()
    print('==========================================================')
    print('The average results for standard machine learning models:')
    print('knn-EMD: {:.2f}%'.format(results[0]))
    print('knn: {:.2f}%'.format(results[1]))
    print('svm-linear: {:.2f}%'.format(results[2]))
    print('svm-poly: {:.2f}%'.format(results[3]))
    print('scm-RBF: {:.2f}%'.format(results[4]))
    print('svm-sigmoid: {:.2f}%'.format(results[5]))
    print('random forest: {:.2f}%'.format(results[6]))
    print('decision tree: {:.2f}%'.format(results[7]))
    print('adaBoost: {:.2f}%'.format(results[8]))

    #cnnModel(320, 320, pathX, pathY)
