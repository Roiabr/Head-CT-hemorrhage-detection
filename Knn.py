from sklearn.neighbors import KNeighborsClassifier


# k-NN classifier for image classification
def knn(trainX, trainY, testX, testY, numNeigh = 3):
    # n_jobs means number of parallel jobs to run. -1 meansusing all processors
    model = KNeighborsClassifier(n_neighbors=numNeigh, n_jobs=-1)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY)
    print("raw pixel accuracy: {:.2f}%".format(acc * 100))