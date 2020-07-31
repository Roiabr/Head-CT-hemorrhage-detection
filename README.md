
## Machine Learning - Final Project

![dataset-cover](https://user-images.githubusercontent.com/44756354/88648428-a697df00-d0cf-11ea-979e-7010b14c7fa9.jpg)
# Head CT hemorrhage detection

### Introduction:
In this project, we used various machine learning algorithms to classify images.
We worked with [Head CT-hemorrhage](https://www.kaggle.com/felipekitamura/head-ct-hemorrhage/?select=head_ct) dataset, that contains 100 normal head CT slices and 100 other with hemorrhage.

### Requirements
* Python 3.6+
* NumPy (pip install numpy)
* Pandas (pip install pandas)
* OpenCv (pip install opencv-python)
* Scikit-learn (pip install scikit-learn)
* SciPy (pip install scipy)
* glob (pip install glob)
* MatplotLib (pip install matplotlib)
* Tensorflow (pip install tensorflow)
* Keras (pip install keras)
### About the project:
We chose to extract the features from the images in 2 ways: 
* Image histogram
* Image resize + flatten
We examined the results of the following classifiers:
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

  First, we extract the features by two approaches:

  1) Simple - Use OpenCV to resize the picture to a smaller size and then push the picture to a one dimensions array with the pixels of the picture.
  
  2) Histogram - Color Histogram is a representation of the distribution of colors in an image. For digital images, a color histogram represents the number of pixels that                         have colors in each of a fixed list of color ranges, that span the image's color space, the set of all possible colors.
  After that, we shuffle and split the data to trainX, trainY, testX, testY and send it to the classifiers
  
- **Classification**:
    
    We send the data to the classifiers and each classifier did the same thing: train the data(fit), get the score of the machine, and the predict of the machine on one of the       pictures in the dataset.
    
   For CNN we use the Simple approach to extract the features and we resize the picture in size of 320X320, and then we shuffle and split the data.                                  After that, we built a model with 5 convolution layer, and then we train the model.
    
## Accuracy (Performance):

   ![Accuracy](https://user-images.githubusercontent.com/44756354/88837833-58730080-d1e1-11ea-8ea1-e31953694850.png)
   ![AccuracyCNN](https://user-images.githubusercontent.com/44756354/88838146-cd463a80-d1e1-11ea-89a0-956b01683913.png)

## Conclusion

   As we can see, the Simple approach got better result then Histogram, as we got 90% accuracy in knn and 85% accuracy in random forest.
   
   But as we thought Convolutional neural network is the best approach to image classification, and indeed we got 100% accuracy when we used it to classify the images.


## Contributor

* [Roi Abramovitch](https://www.linkedin.com/in/roi-abramovitch-04b62821/)

* [Chen Asraf](https://www.linkedin.com/in/chen-asaraf/)
