# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def random_forest(X_train, y_train, X_test, y_test):
    # Instantiate model with 1000 decision trees
    forest_clf = RandomForestClassifier(n_estimators=1000, max_depth=1000, random_state=42)
    # Train the model on training data
    forest_clf.fit(X_train, y_train)
    acc = forest_clf.score(X_test, y_test)
    print("raw pixel accuracy: {:.2f}%".format(acc * 100))
