import pandas as pd
import numpy as np
import math
from nilearn import image as img
import pickle as pk
import matplotlib.pyplot as plt
import os
import glob
from sklearn.linear_model import Ridge, RidgeCV
import time

from functions import store_avg_tr, map_stimuli_w2v, load_nifti_and_w2v, list_diff, \
    two_vs_two, store_trs_spm, store_trs_fsl, leave_two_out, store_masked_trs_spm, store_betas_spm
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler


def avg_trs():
    # Function to store Averaged concatenated TRs on GDrive.
    nifti_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\Preprocessed\Reg_to_Std_and_Str\\"
    participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
    for participant in participants:
        print(participant)
        file_name = nifti_path + "P_" + str(participant) + "\\"
        file_name_a = glob.glob(file_name + "Synonym_RunA*\\filtered_func_data.nii")
        file_name_b = glob.glob(file_name + "Synonym_RunB*\\filtered_func_data.nii")
        store_avg_tr(participant, file_name_a, file_name_b)


def create_w2v_mappings():
    """
    Retrieve word2vec vectors from Word2Vec for two-word stimuli only for now.
    :return: Nothing; stores the concatenated vectors to disk.
    """
    participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
    all_stims = []
    for participant in participants:
        stims = map_stimuli_w2v(participant)
        all_stims.extend(stims)

    all_stims_set = list(set(all_stims))

    # Now load the Word2Vec model.
    model = KeyedVectors.load_word2vec_format("G:\\jw_lab\\jwlab_eeg\\regression\\GoogleNews-vectors-negative300.bin.gz", binary=True)

    stim_vector_dict = {}
    for stim in all_stims_set:
        words = stim.split()
        vector = []

        # Each word vector should be of size 600.
        for word in words:
            word_vector = model[word]
            vector.extend(word_vector.tolist())
        stim_vector_dict[stim] = vector

    np.savez_compressed('G:\comlam\embeds\\two_words_stim_w2v_concat_dict.npz', stim_vector_dict)



def cross_validation_nested(part=None, avg_w2v=False, mean_removed=False, load_avg_trs=False, masked=False, permuted=False ,store_cosine_diff=False,
                            beta=True, beta_mask_type='gm'):
    """
    :param part: Accepts a list of participants. Example: [1003, 1006]. List of integers.
    :avg_w2v: To predict avg w2v vectors or concat w2v vectors. Boolean.
    :mean_removed: Whether to use mean removed data or not. Boolean.
    :load_avg_trs: Whether to load avg_trs or concat_trs. Boolean.
    :masked: Whether to use masked data or not. Boolean.
    :return: None but prints 2v2 accuracy for the participant.
    """
    # Do ridge regression with GridSearchCV here.
    # Run the analysis for each participant here.
    print('Calling cross_validation_nested.')

    participant_accuracies = {}
    cosine_diff_dict = {}

    if type(part) == list:
        participants = part
    else:
        # participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
        participants = [1016]
    for participant in participants:
        print(participant)
        x, y, stims = load_nifti_and_w2v(participant, avg_w2v=avg_w2v, mean_removed=mean_removed, load_avg_trs=load_avg_trs, masked=masked, permuted=permuted,
                                         beta=beta, beta_mask_type=beta_mask_type)
        print('loaded data')


        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        print('Scaled Data')

        # Load the data and the stims to do a leave two out cv.
        # Load the nifti, the word vectors, and the stim and then leave out two samples on which you'll do 2v2.

        # Write a function to do the leave-two-out cv. This returns the train and test indices.
        train_indices, test_indices = leave_two_out(stims)

        ## [[[1,2,4,5], [6,7] ], [[2,4,5,6], [1, 7]], ....   ]
        print('Decided indices')
        preds_list = []
        y_test_list = []
        i = 0
        start = time.time()
        for train_index, test_index in zip(train_indices, test_indices):

            if i % 100 == 0:
                print('Iteration: ', i)
            i += 1

            # model = Ridge(solver='cholesky')
            # ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
            # clf = GridSearchCV(model, param_grid=ridge_params, n_jobs=-1, scoring='neg_mean_squared_error', cv=8, verbose=5) # Setting cv=10 so that 4 samples are used for validation.
            # clf.fit(x[train_index], y[train_index])
            # preds = clf.predict(x[test_index])


            alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]
            # Uses LOOCV by default to tune hyperparameter tuning.
            cv_model = RidgeCV(alphas=alphas, gcv_mode='svd', scoring='neg_mean_squared_error', alpha_per_target=True)

            cv_model.fit(x_scaled[train_index], y[train_index])
            preds = cv_model.predict(x_scaled[test_index])

            # Store the preds in an array and all the ytest with the indices.

            preds_list.append(preds)
            y_test_list.append(y[test_index])

        accuracy, cosine_diff = two_vs_two(preds_list, y_test_list, store_cos_diff=store_cosine_diff)
        cosine_diff_dict[participant] = cosine_diff



        participant_accuracies[participant] = accuracy
        print(participant_accuracies)

    if store_cosine_diff:
        np.savez_compressed("/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/debug_logs_files/cosine_diffs_2v2.npz", cosine_diff_dict)
    if permuted:
        # Save the permutation test results.
        timestr = time.strftime("%Y%m%d-%H%M%S")
        np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/results/permuted/P{participant}/{participant}_{timestr}.npz",participant_accuracies)
    stop = time.time()
    print('Total time: ', stop - start)
    print(participant_accuracies)


cross_validation_nested(avg_w2v=False, mean_removed=False, load_avg_trs=False, masked=True, permuted=True, store_cosine_diff=False,
                        beta=True, beta_mask_type='gm')

# parts = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019, 1024]
# parts = [1003, 1006, 1008, 1010]
parts = [1016]
for p in parts:
    print("Participant: ", p)
    # try:
    store_betas_spm(p, 'sentiment', mask_type='gm')
    # except Exception as e:
    #     print("Participant not found or something: ", e)
    #     pass
# store_trs_fsl(1012, 'sentiment', remove_mean=False)