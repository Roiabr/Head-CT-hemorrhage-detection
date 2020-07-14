from sklearn.neighbors import KNeighborsClassifier
import Draw


def knn(trainX, trainY, testX, testY, images, index, numNeigh=3):
    # n_jobs means number of parallel jobs to run. -1 meansusing all processors
    model = KNeighborsClassifier(n_neighbors=numNeigh, n_jobs=-1)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY)

    print("---------------------KNN--------------------------")

    Draw.drawPredict(model, testX, testY, images, index)

    print("Knn: raw pixel accuracy: {:.2f}%".format(acc * 100))
