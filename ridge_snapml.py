"""
File that implements Ridge Regression using SnapML.
"""
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler
from functions import load_nifti_and_w2v, two_vs_two, leave_two_out
from snapml import LinearRegression
import time

def cv_snapml(part=None):
    """
    :param part: Accepts a list of participants. Example: [1003, 1006]
    :return: 2v2 accuracy for the participant.
    """
    # Do ridge regression with GridSearchCV here.
    # Run the analysis for each participant here.
    print('Calling cross_validation_nested.')

    participant_accuracies = {}

    if type(part) == list:
        participants = part
    else:
        participants = [1003]  # , 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
    for participant in participants:
        print(participant)
        x, y, stims = load_nifti_and_w2v(1003)
        print('loaded data')

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        print('Scaled Data')

        # Load the data and the stims to do a leave two out cv.
        # Load the nifti, the word vectors, and the stim and then leave out two samples on which you'll do 2v2.

        # Write a function to do the leave-two-out cv. This returns the train and test indices.
        train_indices, test_indices = leave_two_out(stims)
        print('Decided indices')
        preds_list = []
        y_test_list = []
        i = 0
        start = time.time()
        for train_index, test_index in zip(train_indices, test_indices):
            print('Iteration: ', i)
            i += 1

            # model = Ridge(solver='cholesky')
            # ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
            # clf = GridSearchCV(model, param_grid=ridge_params, n_jobs=-1, scoring='neg_mean_squared_error', cv=8, verbose=5) # Setting cv=10 so that 4 samples are used for validation.
            # clf.fit(x[train_index], y[train_index])
            # preds = clf.predict(x[test_index])

            # ridge = Ridge(solver='cholesky', alpha=100000)
            ridge = LinearRegression(max_iter=1000, regularizer=0.0001, use_gpu=True, fit_intercept=True, n_jobs=32, penalty='l2')

            ridge.fit(x_scaled[train_index], y[train_index])
            preds = ridge.predict(x_scaled[test_index])

            # Store the preds in an array and all the ytest with the indices.

            preds_list.append(preds)
            y_test_list.append(y[test_index])

        accuracy = two_vs_two(preds_list, y_test_list)

        participant_accuracies[participant] = accuracy
        print(participant_accuracies)
    stop = time.time()
    print('Total time: ', stop - start)
    print(participant_accuracies)