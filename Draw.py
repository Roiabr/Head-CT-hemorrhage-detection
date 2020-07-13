#Let's take a look at the images
import matplotlib.pyplot as plt
import numpy as np


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
