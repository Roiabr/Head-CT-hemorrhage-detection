from Knn import knn
import Extract

if __name__ == '__main__':
    path = 'C:/Users/owner/Desktop/dataset.txt'
    trainX, trainY, testX, testY = read_data(path)
    knn(trainX, trainY, testX, testY, numNeigh = 5)