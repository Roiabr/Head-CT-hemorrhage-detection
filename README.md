
## Machine Learnig - Final Project

![dataset-cover](https://user-images.githubusercontent.com/44756354/88648428-a697df00-d0cf-11ea-979e-7010b14c7fa9.jpg)
# Head CT hemorrhage detection

### Introduction:
In this project, we asked for machine learning final project to choose a dataset

and to implement a machine learning classifiers.

### Requirements
* Python 3.6+
* NumPy (pip install numpy)
* Pandas (pip install pandas)
* OpenCv (pip install opencv-python)
* Scikit-learn (pip install scikit-learn)
* SciPy (pip install scipy)
* MatplotLib (pip install matplotlib)
* Tensorflow (pip install tensorflow)
* Keras (pip install keras)
### About the project:
We choose to work on the [Head CT-hemorrhage](https://www.kaggle.com/felipekitamura/head-ct-hemorrhage/?select=head_ct) dataset
and we decided  to implement those classifiers:
* KNN - with the euclidean distance                      
* KNN - with earth mover distance            
* SVM - with linear, polynomial, RBF, sigmoid kernel.  
* ADABOOST
* DECISION TREE
* RANDOM FOREST
* CNN - Convolutional neural network

## Exemple:
![both](https://user-images.githubusercontent.com/44756354/88721518-3faa1280-d12f-11ea-8ddb-9ed9ba3aa8d2.png)


## Implementation:
- **Feature Extraction**: 

  First we extract the features by two approch:

  1) Use OpenCV to resize the picture to a smaller size and then push the picture to a one dimensions array with the pixels of the picture.
  
  2) Color Histogram - Color Histogram is a representation of the distribution of colors in an image. For digital images, a color histogram represents the number of pixels that                         have colors in each of a fixed list of color ranges, that span the image's color space, the set of all possible colors.
  After that we shuffle and split the data to trainX, trainY, testX, testY and send it to the classifiers
  
- **Classification**:
 
    We send the data to several classifiers and each classifier did the same thing: train the data(fit),get the score of the machine and the predict of the machine on one of the         picture in the dataset.
 
 

## Conclusion


## Contributor

* [Roi Abramovitch](https://www.linkedin.com/in/roi-abramovitch-04b62821/)

* [Chen Asraf](https://www.linkedin.com/in/chen-asaraf/)
