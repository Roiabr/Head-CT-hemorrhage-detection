import pandas as pd
import numpy as np
import glob
import Draw
import cv2


def image_to_feature_vector(image, size):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC).flatten()


def extract_features(pathX, pathY):
    files = sorted(glob.glob(pathX))
    labels_df = pd.read_csv(pathY)
    Y = np.array(labels_df[' hemorrhage'].tolist())
    images = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in files])

   # Draw.draw(images, Y)

    size = (320, 320)
    flatten_size = size[0] * size[1]
    X = np.empty(shape=(0, flatten_size))

    for i, image in enumerate(images):
        X = np.vstack([image_to_feature_vector(image, size), X])

    return X, Y, images
