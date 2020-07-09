import pandas as pd
import numpy as np
import glob
import cv2


def image_to_feature_vector(image, size=(32, 32)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_features():
    files = sorted(glob.glob("head_ct/*.png"))
    labels_df = pd.read_csv('labels.csv')
    Y = np.array(labels_df[' hemorrhage'].tolist())

    images = np.array([cv2.imread(path) for path in files])
    X = []

    for image in images:
        X.append(image_to_feature_vector(image))

    return X, Y
