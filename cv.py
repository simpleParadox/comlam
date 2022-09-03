"""
@author: Rohan Saha
This script takes some samples from all the 10 runs and keeps them in the training set. The rest
of the samples are put in the test set. But the samples for a single stimulus are averaged in the test set.
Therefore, each stimulus has only one sample in the test set.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import glob


def get_train_test_split(X, y):
    # Preprocess data here and obtain train test splits.
    return X_train, X_test, y_train, y_test


def nested_cross_validation(iterations, participant):

    # First load the raw data.

    raw_path = ""


    X, y = load_raw_data(participant)  # This is a custom function

    iter_scores = []

    # Now do the monte carlo nested cv.
    for iter in range(iterations):

        # Get different training and test sets everytime. This is the outer cv loop.
        # Since the training and test sets are not exclusive for each iteration - it's monte carlo cv.
        X_train, X_test, y_train, y_test = get_train_test_split(X, y)  # This is a custom function.

        alphas = alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]
        model = Ridge(solver='svd')
        # GridSearchCV internally does cross-validation. This is the inner cv loop.
        clf = GridSearchCV(model, alphas, n_jobs=-1, cv=5, scoring='neg_mean_squared_error')

        # Train the classifier with GridSearchCV
        clf.fit(X_train, y_train)

        # Obtain predictions.
        preds = clf.predict(X_test)

        # Evaluate predictions.
        score = evaluate_predictions(preds, y_test)  # This is a custom function.

        iter_scores.append(score)

    # Return averaged score.
    return np.mean(iter_scores)
