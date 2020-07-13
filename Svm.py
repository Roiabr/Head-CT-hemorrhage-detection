from sklearn.svm import SVC

import Draw


def svm(X_train, y_train, X_test,y_test,images):
    svclassifier1 = SVC(kernel='linear')
    svclassifier1.fit(X_train, y_train)
    acc = svclassifier1.score(X_test, y_test)

    print("---------------------SVM-LINEAR--------------------------")

    Draw.drawPredict(svclassifier1, X_test, y_test, images)

    print("svm-linear: raw pixel accuracy: {:.2f}%".format(acc * 100))

    svclassifier2 = SVC(kernel='poly')
    svclassifier2.fit(X_train, y_train)
    acc = svclassifier2.score(X_test, y_test)

    print("---------------------SVM-POLY--------------------------")

    Draw.drawPredict(svclassifier2, X_test, y_test, images)

    print("svm-poly: raw pixel accuracy: {:.2f}%".format(acc * 100))

    svclassifier3 = SVC(kernel='rbf')
    svclassifier3.fit(X_train, y_train)
    acc = svclassifier3.score(X_test, y_test)

    print("---------------------SVM-RBF--------------------------")

    Draw.drawPredict(svclassifier3, X_test, y_test, images)

    print("svm-rbf: raw pixel accuracy: {:.2f}%".format(acc * 100))

    svclassifier4 = SVC(kernel='sigmoid')
    svclassifier4.fit(X_train, y_train)
    acc = svclassifier4.score(X_test, y_test)

    print("---------------------SVM-SIGMOID--------------------------")

    Draw.drawPredict(svclassifier4, X_test, y_test, images)

    print("svm-sigmoid: raw pixel accuracy: {:.2f}%".format(acc * 100))
