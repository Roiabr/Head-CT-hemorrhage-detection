import Knn
import Extract as extract


def splitTestTrain(X, Y):
    trainSize = (int)(0.8 * X.shape[0])
    trainX = X[: trainSize]
    trainY = Y[: trainSize]
    testX = X[trainSize:]
    testY = Y[trainSize:]
    return trainX, trainY, testX, testY


if _name_ == '_main_':
    X, Y = extract.extract_features()
    trainX, trainY, testX, testY = splitTestTrain(X, Y)
    Knn.knn(trainX, trainY, testX, testY, numNeigh = 2)
    Knn.knn_emd(trainX, trainY, testX, testY)