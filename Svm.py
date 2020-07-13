from sklearn.svm import SVC


def svm(X_train, y_train, X_test,y_test):
    svclassifier1 = SVC(kernel='linear')
    svclassifier1.fit(X_train, y_train)
    acc = svclassifier1.score(X_test, y_test)
    print("raw pixel accuracy: {:.2f}%".format(acc * 100))

    svclassifier2 = SVC(kernel='rbf')
    svclassifier2.fit(X_train, y_train)
    acc = svclassifier2.score(X_test, y_test)
    print("raw pixel accuracy: {:.2f}%".format(acc * 100))

    svclassifier3 = SVC(kernel='sigmoid')
    svclassifier3.fit(X_train, y_train)
    acc = svclassifier3.score(X_test, y_test)
    print("raw pixel accuracy: {:.2f}%".format(acc * 100))
