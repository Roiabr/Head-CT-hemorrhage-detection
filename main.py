# from Cnn import cnnModel
import Random_Forest
import Svm
from DecisionTreeClassifier import decisionTreeClassifier
from Knn import knn
import Extract as extract


def splitTestTrain(X, Y, images):
    trainSize = int(0.8 * X.shape[0])
    trainX = X[: trainSize]
    trainIm = images[: trainSize]
    trainY = Y[: trainSize]
    testX = X[trainSize:]
    testIm = images[trainSize:]
    testY = Y[trainSize:]
    return trainX, trainY, testX, testY, trainIm, testIm


if __name__ == '__main__':
    X, Y, images = extract.extract_features()

    trainX, trainY, testX, testY, trainIm, testIm = splitTestTrain(X, Y, images)

    knn(trainX, trainY, testX, testY, testIm, numNeigh=2)
    Svm.svm(trainX, trainY, testX, testY, testIm)
    Random_Forest.random_forest(trainX, trainY, testX, testY, testIm)
    decisionTreeClassifier(trainX, trainY, testX, testY, testIm)

    # cnnModel(128, 128)
