import pandas as pd
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras._impl.keras.layers import Activation
from tensorflow.python.layers.convolutional import Conv2D
from tensorflow.python.layers.core import Flatten, Dense, Dropout
from tensorflow.python.layers.pooling import MaxPooling2D


def cnnModel(img_width, img_height):
    files = sorted(glob.glob("head_ct/*.png"))
    labels_df = pd.read_csv('labels.csv')
    Y = np.array(labels_df[' hemorrhage'].tolist())
    # images = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in files])
    images = np.empty((len(files), img_width, img_height))

    for i, _file in enumerate(files):
        images[i, :, :] = cv2.resize(cv2.imread(_file, 0), (img_width, img_height))

    train_images, test_images, train_labels, test_labels = train_test_split(images, Y, test_size=0.2,
                                                                            random_state=1)
    val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5,
                                                                        random_state=1)

    SIIM_custom_model = None
    input_shape = (img_width, img_height, 1)
    SIIM_custom_model = Sequential()

    SIIM_custom_model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    SIIM_custom_model.add(Activation('relu'))

    SIIM_custom_model.add(MaxPooling2D(pool_size=(2, 2)))

    SIIM_custom_model.add(Conv2D(32, (3, 3)))
    SIIM_custom_model.add(Activation('relu'))
    SIIM_custom_model.add(MaxPooling2D(pool_size=(2, 2)))

    SIIM_custom_model.add(Conv2D(64, (3, 3)))
    SIIM_custom_model.add(Activation('relu'))
    SIIM_custom_model.add(MaxPooling2D(pool_size=(2, 2)))

    # Finally, we will add two dense layers, or 'Fully Connected Layers'.
    # These layers are classical neural nets, without convolutions.

    SIIM_custom_model.add(Flatten())
    SIIM_custom_model.add(Dense(64))
    SIIM_custom_model.add(Activation('relu'))

    # Dropout is an overfitting reduction technique.

    SIIM_custom_model.add(Dropout(0.5))

    # Now, we will set the output o the network.
    # The Dense function has the argument "1" because the net output is the hematoma x non-hematoma classification

    SIIM_custom_model.add(Dense(1))

    # The output is either 0 or 1 and this can be obtained with a sigmoid function.

    SIIM_custom_model.add(Activation('sigmoid'))

    # Let's compile the network.

    SIIM_custom_model.compile(loss='binary_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy'])
