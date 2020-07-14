from sklearn.ensemble import RandomForestClassifier
import Draw


def random_forest(X_train, y_train, X_test, y_test, images, index):
    # Instantiate model with 1000 decision trees
    forest_clf = RandomForestClassifier()
    # Train the model on training data
    forest_clf.fit(X_train, y_train)
    acc = forest_clf.score(X_test, y_test)

    print("---------------------Random-Forest--------------------------")

    Draw.drawPredict(forest_clf, X_test, y_test, images, index)

    print("Random-Forest: raw pixel accuracy: {:.2f}%".format(acc * 100))
