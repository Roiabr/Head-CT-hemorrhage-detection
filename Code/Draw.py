import random
import matplotlib.pyplot as plt


def draw(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
        if labels[i] == 1:
            plt.title("\nLabel:{}".format("Hemorrhage"))
        else:
            plt.title("\nLabel:{}".format("No Hemorrhage"))
    # show the plot
    plt.show()


def drawPredict(model, testX, testY, images, index):
    modelName = str(model)
    modelName = modelName.split("(")[0]
    rand = random.randint(0, 39)
    inde = int(index[rand])
    plt.imshow(images[inde])
    if testY[rand] == 1:
        plt.title("\nLabel:{}".format("Hemorrhage"))
    else:
        plt.title("\nLabel:{}".format("No Hemorrhage"))
    plt.show()
    predict = "Hemorrhage" if model.predict([testX[rand]]) == 1 else "No Hemorrhage"
    label = "Hemorrhage" if testY[rand] == 1 else "No Hemorrhage"
    print("The model", modelName, " predict:", predict, "the correct label:", label)

if __name__ == '__main__':
    model = "KNeighborsClassifier(n_jobs=-1, n_neighbors=2)"
    modelName = model.split("(")[0]
    print(modelName)
