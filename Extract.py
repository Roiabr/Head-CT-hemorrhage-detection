"""
This class includes the methods to extract features from image.
The features data must be a vector-shaped for the models we used, and
yet preserve the important properties in the image.
The methods converts 2D image to 1D vector:
    image_to_vector - convert by resizing and then flatten the image.
    image_to_histogram_vector - convert to image histogram
"""
import pandas as pd
import numpy as np
import glob
import Draw
import cv2

def image_to_vector(image, size):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC).flatten()

def image_to_histogram_vector(image):
    histogram, bin_edges = np.histogram(image, bins=np.arange(250))
    histogram = np.reshape(histogram, (1, 249))
    return histogram

"""
Main method
Parameters:
    pathX : path to the images folder
    pathY: path to the labels csv file
    method: the extract-feature method to use.
            'simple' =  image_to_vector
            'histogram' = image_to_histogram_vector 
"""
def extract_features(pathX, pathY, method = 'simple'):
    # Load images and labels
    files = sorted(glob.glob(pathX))
    labels_df = pd.read_csv(pathY)
    Y = np.array(labels_df[' hemorrhage'].tolist())
    images = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in files])

    # Draw.draw(images, Y)

    X = []
    if method == 'simple':
        # Resize the image to lower the features dimension
        size = (320, 320)
        flatten_size = size[0] * size[1]
        X = np.empty(shape=(0, flatten_size))
        for i, image in enumerate(images):
            X = np.vstack([image_to_vector(image, size), X])

    elif method == 'histogram':
        X = np.empty(shape=(0, 249))
        for i, image in enumerate(images):
            X = np.vstack([image_to_histogram_vector(image), X])

    return X, Y, images
