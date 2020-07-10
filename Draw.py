#Let's take a look at the images
import matplotlib.pyplot as plt
import numpy as np


def draw(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))
        plt.title("\nLabel:{}".format(labels[i]))
    # show the plot
    plt.show()
