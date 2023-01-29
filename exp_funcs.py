import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing sklearn packages for ML.
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from calculate_nc import load_betas_and_stims, get_train_and_test_from_nc


from functions import extended_2v2, leave_one_out, get_dim_corr


def decoding_analysis(X=None, y=None, participant=None, iters=50, loocv=False, permuted=False, current_seed=-1, use_nc=False, brain_type='wholeBrain',
                             remove_neutral=False):
    fold_accuracies = []
    iter_count = 0
    if permuted:
        sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2)
    else:
        sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2, random_state=42)
    if use_nc:
        X, y = load_betas_and_stims(participant, brain_type, exp_type='sentiment', remove_neutral=remove_neutral)
        print("y: ", y)
        print("len(y): ", len(y))
        for i in range(iters):
            X_train, X_test, y_train, y_test = get_train_and_test_from_nc(X, y, seed=i)
            print("Iter count: ", iter_count)
            iter_count += 1
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            param_grid = {'alpha':  [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            model = RidgeCV()
            clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=None)  # Setting cv=None uses default 5 fold cross-validation.
            clf.fit(X_train, y_train)
            # accuracy = clf.score(X_test, y_test)
            # Use 2vs2 accuracy here.

        if permuted:
            return fold_accuracies
        return np.mean(fold_accuracies)
    else:
        print("Not using noise ceiling")
        preds_list = []
        y_test_list = []
        if loocv:
            train_indices, test_indices = leave_one_out(y)
            print("Length of test indices: ", len(test_indices))
        for train_index, test_index in zip(train_indices, test_indices):
            print("Iter count: ", iter_count, flush=True)
            iter_count += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print("X_train: ", X_train.shape)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Scale and do PCA on the target embeddings.
            scaler_target = StandardScaler()
            y_train = scaler_target.fit_transform(y_train)
            y_test = scaler_target.transform(y_test)

            pca = PCA(n_components=20, random_state=42)
            y_train = pca.fit_transform(y_train)
            y_test = pca.transform(y_test)

            # Now train the model.
            alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            clf = RidgeCV(alphas=alphas, gcv_mode='svd', scoring='neg_mean_squared_error', alpha_per_target=True)
            # clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=LeaveOneOut())  # Setting cv=None uses default 5 fold cross-validation.
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            preds_list.append(pred)
            y_test_list.append(y_test)
            # print("Predictions: ", clf.predict(X_test))
            # print("Actual: ", y_test)
            # print("Accuracy: ", accuracy)
            # Use 2vs2 accuracy here.
        accuracy, cosine_diff = extended_2v2(preds_list, y_test_list)
        print("Accuracy: ", accuracy, flush=True)
        fold_accuracies.append(accuracy)
        if permuted:
            return fold_accuracies
        return np.mean(fold_accuracies)



def encoding_analysis(X=None, y=None, participant=None, iters=50, loocv=False, permuted=False, current_seed=-1, use_nc=False, brain_type='wholeBrain', remove_neutral=False, do_corr=False):
    fold_accuracies = []
    iter_count = 0
    correlation_values = None
    if permuted:
        sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2)
    else:
        sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2, random_state=42)
    if use_nc:
        X, y = load_betas_and_stims(participant, brain_type, exp_type='sentiment', remove_neutral=remove_neutral)
        print("y: ", y)
        print("len(y): ", len(y))
        for i in range(iters):
            X_train, X_test, y_train, y_test = get_train_and_test_from_nc(X, y, seed=i)
            print("Iter count: ", iter_count)
            iter_count += 1
            scaler = StandardScaler()
            # NOTE: Only standardizing one of the things. In case of encoding, it's the input.
            y_train = scaler.fit_transform(y_train)
            y_test = scaler.transform(y_test)

            param_grid = {'alpha':  [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            model = RidgeCV()
            clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=None)  # Setting cv=None uses default 5 fold cross-validation.
            clf.fit(y_train, X_train)
            # accuracy = clf.score(X_test, y_test)
            # Use 2vs2 accuracy here.

        if permuted:
            return fold_accuracies
        return np.mean(fold_accuracies)
    else:
        print("Not using noise ceiling")
        preds_list = []
        y_test_list = []
        if loocv:
            train_indices, test_indices = leave_one_out(y)
            print("Length of test indices: ", len(test_indices))
        for train_index, test_index in zip(train_indices, test_indices):
            print("Iter count: ", iter_count, flush=True)
            iter_count += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print("X_train: ", X_train.shape)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Scale and do PCA on the target embeddings.
            scaler_target = StandardScaler()
            y_train = scaler_target.fit_transform(y_train)
            y_test = scaler_target.transform(y_test)

            pca = PCA(n_components=20, random_state=42)
            X_train = pca.fit_transform(X_train)
            x_test = pca.transform(X_test)

            # Now train the model.
            alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            clf = RidgeCV(alphas=alphas, gcv_mode='svd', scoring='neg_mean_squared_error', alpha_per_target=True)
            # clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=LeaveOneOut())  # Setting cv=None uses default 5 fold cross-validation.
            clf.fit(y_train, X_train)
            pred = clf.predict(y_test)
            preds_list.append(pred)
            y_test_list.append(X_test)
            # print("Predictions: ", clf.predict(X_test))
            # print("Actual: ", y_test)
            # print("Accuracy: ", accuracy)
            # Use 2vs2 accuracy here.
        accuracy, cosine_diff = extended_2v2(preds_list, y_test_list)
        if do_corr:
            # Do correlation calculation here.
            correlation_values = get_dim_corr(preds_list, y_test_list)
        print("Accuracy: ", accuracy, flush=True)
        fold_accuracies.append(accuracy)
        if permuted:
            return fold_accuracies, correlation_values
        return np.mean(fold_accuracies), correlation_values




def sentiment_classification(X=None, y=None, participant=None, iters=50, loocv=False, permuted=False, current_seed=-1, use_nc=False, brain_type='wholeBrain',
                             remove_neutral=False):

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
        # NOTE: load_betas_and_stims for nc not implemented.
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
        iter_count = 0
        if permuted:
            sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2)
        else:
            sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2, random_state=42)
        if use_nc:
            X, y = load_betas_and_stims(participant, brain_type, exp_type='sentiment', remove_neutral=remove_neutral)
            print("y: ", y)
            print("len(y): ", len(y))
            

            for i in range(iters):
                X_train, X_test, y_train, y_test = get_train_and_test_from_nc(X, y, seed=i)
                print("Iter count: ", iter_count)
                iter_count += 1
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                param_grid = {'C':  [0.001, 0.01, 0.1, 1, 10, 100]}
                model = LogisticRegression(max_iter=500)
                clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=None)  # Setting cv=None uses default 5 fold cross-validation.
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
                fold_accuracies.append(accuracy)
            if permuted:
                return fold_accuracies
            return np.mean(fold_accuracies)
        else:
            for train_index, test_index in sss.split(X, y):
                print("Iter count: ", iter_count)
                iter_count += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
                model = LogisticRegression(max_iter=500)
                clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=None)  # Setting cv=None uses default 5 fold cross-validation.
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
                # print("Predictions: ", clf.predict(X_test))
                # print("Actual: ", y_test)
                # print("Accuracy: ", accuracy)
                fold_accuracies.append(accuracy)
            if permuted:
                return fold_accuracies
            return np.mean(fold_accuracies)


def congruency_classification(X=None, y=None, participant=None, iters=50, permuted=False, current_seed=-1, use_nc=False, brain_type='wholeBrain'):
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
    if permuted:
        sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2)
    else:
        sss = StratifiedShuffleSplit(n_splits=iters, test_size=0.2, random_state=42)
    iter_count = 0
    if use_nc:
        X, y = load_betas_and_stims(participant, brain_type, exp_type='congruency')
        for i in range(iters):
            print("iter: ", i)
            X_train, X_test, y_train, y_test = get_train_and_test_from_nc(X, y, seed=i)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            param_grid = {'C':  [0.001, 0.01, 0.1, 1, 10, 100]}
            model = LogisticRegression(max_iter=500)
            clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=None)  # Setting cv=None uses default 5 fold cross-validation.
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            fold_accuracies.append(accuracy)
        if permuted:
            return fold_accuracies
        return np.mean(fold_accuracies)            
    else:
        for train_index, test_index in sss.split(X, y):
            print(f"Iteration: {iter_count}", flush=True)
            iter_count += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


            param_grid = {'C':  [0.001, 0.01, 0.1, 1, 10, 100]}
            model = LogisticRegression(max_iter=500)
            clf = GridSearchCV(model, param_grid, n_jobs=-1, cv=None)  # Setting cv=None uses default 5 fold cross-validation.
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            fold_accuracies.append(accuracy)
        if permuted:
            return fold_accuracies
        return np.mean(fold_accuracies)