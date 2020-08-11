"""
This class includes the methods to extract features from image.
The features data must be a vector-shaped for the models we used, and
yet preserve the important properties in the image.
The methods converts 2D image to 1D vector:
    image_to_vector - convert by resizing and then flatten the image.
    image_to_histogram_vector - convert to image histogram.
    fd_hu_moments - convert to 7-dimension vector describing the shapes in the image.
    pca_reduction - usnig Principal Component Analysis for dimensionality reduction.
    cany_edge - convert by finding the image edges and then flatten the image.
"""
import numpy as np
import Draw
import cv2
from enum import Enum
from sklearn.decomposition import PCA

"""
enum class: Represent the method to extract features from the input images 
"""
class Method(Enum):
    SIMPLE = 1
    HISTOGRAM = 2
    HUMOMENTS = 3
    PCA = 4
    EDGES = 5

"""
First method for extracting the features from the input image,
by reducing its dimensions and turning it into a vector.
"""
def image_to_vector(image, size):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC).flatten()



"""
Second method for extracting the features from the input image,
by finding the histogram of each image.
"""
def image_to_histogram_vector(image):
    histogram, bin_edges = np.histogram(image, bins=np.arange(257))
    histogram = np.reshape(histogram, (1, 256))
    return histogram

"""
Third method:
Calculates seven Hu invariants of the image.
"""
def fd_hu_moments(image):
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

"""
Fourth method:
PCA (Principal Component Analysis) for or dimensionality reduction
"""
def pca_reduction(images):
    # Make an instance of the Model
    pca = PCA(n_components=.95)
    pca.fit(images)
    succinct_x = pca.transform(images)
    return succinct_x

"""
Fifth method:
Using Canny-edge detector for contours in the image
"""
def cany_edge(image, size):
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)
    edges = cv2.Canny(image, 100, 200)
    edges = edges.flatten()
    return edges



"""
Main method to extract features
Parameters:
    images : images at original size as numpy arrays.
    method: Method(Enum), method to use while extract features.
            'SIMPLE' =  image_to_vector
            'HISTOGRAM' = image_to_histogram_vector 
            'HUMOMENTS' = fd_hu_moments
            'PCA' = pca_reduction
            'EDGES' = cany_edge
Returns:
    succinct_x: images in succinct format
"""
def extract_features(images, method=Method.SIMPLE, size= (32, 32)):
    # Draw.draw(images, labels)

    succinct_x = []
    if method == Method.SIMPLE:
        # Resize the image to lower the features dimension
        flatten_size = size[0] * size[1]
        succinct_x = np.empty(shape=(0, flatten_size))
        for image in images:
            succinct_x = np.vstack([image_to_vector(image, size), succinct_x])

    elif method == Method.HISTOGRAM:
        succinct_x = np.empty(shape=(0, 256))
        for image in images:
            succinct_x = np.vstack([image_to_histogram_vector(image), succinct_x])

    elif method == Method.HUMOMENTS:
        succinct_x = np.empty(shape=(0,7))
        for image in images:
            succinct_x = np.vstack([fd_hu_moments(image), succinct_x])

    elif method == Method.PCA:
        # first: create a matrix of all images flatten to a vector
        resized = [image_to_vector(image,size) for image in images] # return a list of np.array
        resized_images = np.stack(resized, axis=0)
        succinct_x = pca_reduction(resized_images)

    elif method == Method.EDGES:
        flatten_size = size[0] * size[1]
        succinct_x = np.empty(shape=(0, flatten_size))
        for image in images:
            succinct_x = np.vstack([cany_edge(image, size), succinct_x])

    return succinct_x
