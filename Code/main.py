from Cnn import cnnModel
from Code.Classifiers.Adaboost import adaBoost
from Code.Classifiers.Random_Forest import random_forest
from Code.Classifiers.Svm import svmlinear, svmpoly, svmrbf, svmsigmoid
from Code.Classifiers.DecisionTreeClassifier import decisionTreeClassifier
from Code.Classifiers.Knn import knn, knnEMD
import Code.Extract as ex
import numpy as np
import glob
import cv2
import pandas as pd

"""
Shuffle the data and split it into train & test sets.
Parameters:
    X: numpy matrix, representing the images as vectors - each row is the image features.
    Y: numpy vector of the labels.
Returns:
    trainX
    trainY
    testX
    testY 
    imagesTest - images of test set at original size - for the Draw method.
"""
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

"""
Main experiment: 
"""
if __name__ == '__main__':

    # 1) Load images and labels
    pathX = "../Dataset/head_ct/*.png"
    pathY = '../Dataset/labels.csv'

    files = sorted(glob.glob(pathX))
    labels_df = pd.read_csv(pathY)
    labels = np.array(labels_df[' hemorrhage'].tolist())
    images = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in files])

    # Run on variety of image size options:
    for s in range(20, 150, 10):
        # 2) Extract the features with one of four methods: 'SIMPLE', 'HISTOGRAM', 'HUMOMENTS' and 'PAC'.
        # see 'Extract' doc.
        method_to_extract_features = ex.Method.HUMOMENTS
        X = ex.extract_features(images, method=method_to_extract_features, size=(s, s))
        print('Extract features method:', method_to_extract_features, ", image size:", s)

        # 3) Split data into train & test sets, including shuffle of the data
        trainX, trainY, testX, testY, testIm = splitTestTrain(X, labels)

        # 4) Train the models
        print('Begins testing the models...')

        results = np.zeros(9)
        nb_iteration = 10

        for epoch in range(nb_iteration):
            results[0] += knnEMD(trainX, trainY, testX, testY,images, testIm, numNeigh=2)
            results[1] += knn(trainX, trainY, testX, testY,images, testIm, numNeigh=2)
            results[2] += svmlinear(trainX, trainY, testX, testY,images, testIm)
            results[3] += svmpoly(trainX, trainY, testX, testY,images, testIm)
            results[4] += svmrbf(trainX, trainY, testX, testY,images, testIm)
            results[5] += svmsigmoid(trainX, trainY, testX, testY,images, testIm)
            results[6] += random_forest(trainX, trainY, testX, testY,images, testIm)
            results[7] += decisionTreeClassifier(trainX, trainY, testX, testY, images, testIm)
            results[8] += adaBoost(trainX, trainY, testX, testY, images, testIm)

        results = np.divide(results, nb_iteration)
        print('=============================')
        print('The average results for standard machine learning models:')
        print('knn-EMD: {:.2f}%'.format(results[0]))
        print('knn: {:.2f}%'.format(results[1]))
        print('svm-linear: {:.2f}%'.format(results[2]))
        print('svm-poly: {:.2f}%'.format(results[3]))
        print('svm-RBF: {:.2f}%'.format(results[4]))
        print('svm-sigmoid: {:.2f}%'.format(results[5]))
        print('random forest: {:.2f}%'.format(results[6]))
        print('decision tree: {:.2f}%'.format(results[7]))
        print('adaBoost: {:.2f}%'.format(results[8]))
        print('=============================')
        print()
        print()

    cnnModel(s, s, pathX, pathY)

    
