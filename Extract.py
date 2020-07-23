"""
This class includes the methods to extract features from image.
The features data must be a vector-shaped for the models we used, and
yet preserve the important properties in the image.
The methods converts 2D image to 1D vector:
    image_to_vector - convert by resizing and then flatten the image.
    image_to_histogram_vector - convert to image histogram
"""
import numpy as np
import Draw
import cv2
from enum import Enum

"""
enum class: Represent the method to extract features from the input images 
"""
class Method(Enum):
    SIMPLE = 1
    HISTOGRAM = 2


"""
First method for extracting the features from the input image,
by reducing its dimensions and turning it into a vector.
"""
def image_to_vector(image, size):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC).flatten()



"""
Second image for extracting the features from the input image,
by finding the histogram of each image.
"""
def image_to_histogram_vector(image):
    histogram, bin_edges = np.histogram(image, bins=np.arange(250))
    histogram = np.reshape(histogram, (1, 249))
    return histogram


"""
Main method to extract features
Parameters:
    images : images at original size as numpy arrays.
    method: Method(Enum), method to use while extract features.
            'SIMPLE' =  image_to_vector
            'HISTOGRAM' = image_to_histogram_vector 
Returns:
    succinct_x: images in succinct format
"""
def extract_features(images, method=Method.SIMPLE):
    # Draw.draw(images, labels)

    succinct_x = []
    if method == Method.SIMPLE:
        # Resize the image to lower the features dimension
        size = (320, 320)
        flatten_size = size[0] * size[1]
        succinct_x = np.empty(shape=(0, flatten_size))
        for i, image in enumerate(images):
            succinct_x = np.vstack([image_to_vector(image, size), succinct_x])

    elif method == Method.HISTOGRAM:
        succinct_x = np.empty(shape=(0, 249))
        for i, image in enumerate(images):
            succinct_x = np.vstack([image_to_histogram_vector(image), succinct_x])

    return succinct_x
