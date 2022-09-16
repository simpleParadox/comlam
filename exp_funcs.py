import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing sklearn packages for ML.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

def sentiment_classification(X, y, loocv=False, iters=50):

    """
    Do a 3-way sentiment classification to classify positive, negative, and neutral sentiment from the fMRI data.
    The data can be the beta weights or the raw realigned data. This function doesn't do the preprocessing.
    This function can do a LOOCV or classic train_test_split based on the function argument.
    :param X: Input feature data.
    :param y: Labels.
    :param loocv: Whether to do loocv or standard train_test_split. Default=False (do train_test_split of 20%)
    :param iters: train_test_split iters for non-loocv iterations.
    :return:
    """
    print("Doing sentiment classification.")
    if loocv:
        loo_outer = LeaveOneOut()
        accuracies = []
        for train_index, test_index in loo_outer.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            model = LogisticRegression()
            loo_inner = LeaveOneOut()
            clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=loo_inner)  # Using leave one out for hyperparameter tuning.
            clf.fit(X_train, y_train)
            accuracies.append(clf.score(X_test, y_test))

        return np.mean(accuracies)

    else:
        fold_accuracies = []
        for iter in range(iters):
            print("Iteration: ", iter)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            model = LogisticRegression()
            clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=None)  # Setting cv=None uses default 5 fold cross-validation.
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            print("Predictions: ", clf.predict(X_test))
            print("Actual: ", y_test)
            print("Accuracy: ", accuracy)
            fold_accuracies.append(accuracy)

        return np.mean(fold_accuracies)


def congruency_classification(X, y, iters):
    """
    Train a classification model on fMRI data for congruent and incongruent
    stimuli and do a prediction on the same. Using staraified shuffle split.
    :param X: Input data
    :param y: output labels
    :param iters: cross-validation iters
    :return:
    """
    print("Doing congruency classification.")
    fold_accuracies = []
    sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2, random_state=42)
    iter_count = 0
    for train_index , test_index in sss.split(X, y):
        print("Iteration: ", iter_count)
        iter_count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        param_grid = {'C':  [0.001, 0.01, 0.1, 1, 10, 100]}
        model = LogisticRegression()
        clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=None)  # Setting cv=None uses default 5 fold cross-validation.
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)