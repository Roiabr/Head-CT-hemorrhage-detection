from sklearn.svm import SVC

import Draw

def svmlinear(X_train, y_train, X_test,y_test,images, testIm):
    svclassifier1 = SVC(kernel='linear')
    svclassifier1.fit(X_train, y_train)
    acc = svclassifier1.score(X_test, y_test) * 100
    # Draw.drawPredict(svclassifier1, X_test, y_test, images, index)
    return acc

def svmpoly(X_train, y_train, X_test,y_test,images, testIm):
    svclassifier2 = SVC(kernel='poly')
    svclassifier2.fit(X_train, y_train)
    acc = svclassifier2.score(X_test, y_test)  * 100
    # Draw.drawPredict(svclassifier2, X_test, y_test, images, index)
    return acc

def svmrbf(X_train, y_train, X_test,y_test,images, testIm):
    svclassifier3 = SVC(kernel='rbf')
    svclassifier3.fit(X_train, y_train)
    acc = svclassifier3.score(X_test, y_test) * 100
    # Draw.drawPredict(svclassifier3, X_test, y_test, images, index)
    return acc

def svmsigmoid(X_train, y_train, X_test,y_test,images, testIm):
    svclassifier4 = SVC(kernel='sigmoid')
    svclassifier4.fit(X_train, y_train)
    acc = svclassifier4.score(X_test, y_test) * 100
    # Draw.drawPredict(svclassifier4, X_test, y_test, images, index)
    return acc








