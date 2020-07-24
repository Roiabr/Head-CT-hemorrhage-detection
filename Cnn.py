import pandas as pd
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from numpy.random import seed
seed(1337)
from tensorflow.random import set_seed
set_seed(1337)
import warnings
warnings.filterwarnings('ignore')

"""
Loading and resizing 2D images.
Parameters:
    pathX: path to the images folder
    pathY: path to the labels csv file
"""
def load_samples_as_images(pathX, pathY,img_width, img_height):
    files = sorted(glob.glob(pathX))
    labels_df = pd.read_csv(pathY)
    Y = np.array(labels_df[' hemorrhage'].tolist())
    images = np.empty((len(files), img_width, img_height))

    for i, _file in enumerate(files):
        images[i, :, :] = cv2.resize(cv2.imread(_file, 0), (img_width, img_height))

    return images, Y


"""
Train and us cnn model
Parameters:
    img_width: new size for the image width
    img_height: new size for the image height
    pathX: path to the images folder
    pathY: path to the labels csv file
"""
def cnnModel(img_width, img_height, pathX, pathY):
    # load the images and the labels:
    images, Y = load_samples_as_images(pathX, pathY, img_width, img_height)

    # split the dataset into train (80%), validation (10%) and test (10%) sets.
    train_images, test_images, train_labels, test_labels = train_test_split(images, Y, test_size=0.2,
                                                                            random_state=1)
    val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5,
                                                                        random_state=1)

    # ----- Build the model: -----

    # The first layer in the model is convolution layer-
    # hence we need to provide the keyword argument "input_shape"
    # "input_shape" = (image width, image height, number of channels)
    input_shape = (img_width, img_height, 1)

    # Sequential = pre-built keras model where you can just add the layers
    model = Sequential()

    # ----- First convolution layer: -----
    #   number of filters (the dimensionality of the output space) = 32
    #   kernel size (specifying the height and width of the 2D convolution window) = (3,3)
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))

    # activation layer: (Relu)
    model.add(Activation('relu'))

    # max-pooling layer:
    #   shrink the size of the first conv layer's output in 75% to be:
    #   from dimension of: (img_width - kernel_size + 1) , (img_height - kernel_size + 1)
    #   to dimension:   (img_width - kernel_size + 1)/2 , (img_height - kernel_size + 1)/2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ----- Second convolution layer: -----
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ----- Third convolution layer: -----
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Finally, adding dense layers as they are used to predict the labels.

    # flatten layer: expands a three-dimensional vector into a one-dimensional vector
    model.add(Flatten())

    # ----- Fourth layer: dense-----
    model.add(Dense(64))
    model.add(Activation('relu'))

    # Dropout is an overfitting reduction technique.
    model.add(Dropout(0.5))

    # ----- Fifth and last output  layer: -----
    # The Dense function has the argument "1" because the net output is the hematoma x non-hematoma classification
    model.add(Dense(1))

    # The output is either 0 or 1 and this can be obtained with a sigmoid function.
    model.add(Activation('sigmoid'))

    # print the model summary:
    model.summary()


    # compile the network:
    #   loss function binary cross entropy > to predict 0 or 1
    #
    model.compile(loss='binary_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy'])

    nb_train_samples = len(train_images)
    nb_validation_samples = len(val_images)
    epochs = 100
    batch_size = 10

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.0,
        zoom_range=0.1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for validation:
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow(
        train_images[..., np.newaxis],
        train_labels,
        batch_size=batch_size)

    validation_generator = val_datagen.flow(
        val_images[..., np.newaxis],
        val_labels,
        batch_size=batch_size)


    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    print("Accuracy: " + str(model.evaluate(test_images[..., np.newaxis] / 255., test_labels)[1] * 100) + "%")