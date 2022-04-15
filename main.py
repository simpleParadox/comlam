import pandas as pd
import numpy as np
import math
from nilearn import image as img
import pickle as pk
import matplotlib.pyplot as plt
import os
import glob
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
import time

from functions import store_avg_tr, map_stimuli_w2v, load_nifti_and_w2v, list_diff, \
    two_vs_two, store_trs_spm, store_trs_fsl, leave_two_out, store_masked_trs_spm, store_betas_spm, get_dim_corr, leave_one_out, extended_2v2, \
    get_violin_plot, extended_euclidean_2v2
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


def create_w2v_mappings(mean=False):
    """
    Retrieve word2vec vectors from Word2Vec for two-word stimuli only for now.
    :return: Nothing; stores the concatenated vectors to disk.
    """
    # participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
    participants = [1030]
    all_stims = []
    for participant in participants:
        stims = map_stimuli_w2v(participant)
        all_stims.extend(stims)

    all_stims_set = list(set(all_stims))
    spaced_words = []
    for word in all_stims_set:
        words = word.split('_')
        temp = ' '.join(words)
        spaced_words.append(temp)


    # Now load the Word2Vec model.
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)

    stim_vector_dict = {}
    for stim in all_stims_set:
        # words = stim.split()
        words = stim.split('_')
        vector = []

        # Each word vector should be of size 600.
        for word in words:
            word_vector = model[word]
            if mean:
                vector.append(word_vector.tolist())
            else:
                vector.extend(word_vector.tolist())
        if mean:
            vector = np.mean(vector, axis=0)
        stim_vector_dict[stim] = vector


    np.savez_compressed('embeds/sixty_two_word_stims_avg.npz', stim_vector_dict)

# create_w2v_mappings(mean=True)

def cross_validation_nested(decoding=True, part=None, avg_w2v=False, mean_removed=False, load_avg_trs=False, masked=False, permuted=False ,store_cosine_diff=False, nifti_type='rf',
                            beta=True, beta_mask_type='gm', embedding_type='w2v', metric='2v2', leave_one_out_cv=False, predict_sentiment=True):
    """

    :param decoding: Whether to do a decoding or encoding analysis.
    :param part: Accepts a list of participants. Example: [1003, 1006]. List of integers.
    :param avg_w2v: To predict avg w2v vectors or concat w2v vectors. Boolean.
    :param mean_removed: Whether to use mean removed data or not. Boolean.
    :param load_avg_trs: Whether to load avg_trs or concat_trs. Boolean.
    :param masked: Whether to use masked data or not. Boolean.
    :param permuted: Whether do the shuffle the labels before model fitting.
    :param store_cosine_diff: Whether to store the cosine differences from the 2v2 test (Only for decoding).
    :param nifti_type: 'rf' (for non-mni space) or 'wrf' (for mni space)
    :param beta: Whether to use beta weights to train the model.
    :param beta_mask_type: Can be 'gm' or 'roi'.
    :param embedding_type: Can be 'w2v' for Word2Vec or 'roberta' for RoBERTa embeddings.
    :param metric: '2v2' or 'corr'. If 'decoding' is set to True, then use the '2v2' or 'corr'. If decoding=False, then use 1 vs 2 test with euclidean distance.
    :param leave_one_out_cv: Whether to use leave_one_out_cv or leave_to_out_cv. Boolean.
    :param predict_sentiment: Whether to predict the sentiment vectors.
    :return: None
    """
    # Do ridge regression with GridSearchCV here.
    # Run the analysis for each participant here.

    participant_accuracies = {}
    cosine_diff_dict = {}
    participant_correlations = {}
    avg_r2 = []

    if type(part) == list:
        participants = part
    else:
        # participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
        participants = [1014]#, 1006, 1007, 1008, 1010, 1012, 1016, 1017, 1019, 1024]
    for participant in participants:
        print(participant)
        x, y, stims = load_nifti_and_w2v(participant, avg_w2v=avg_w2v, mean_removed=mean_removed, load_avg_trs=load_avg_trs, masked=masked, permuted=permuted,
                                         beta=beta, beta_mask_type=beta_mask_type, embedding_type=embedding_type, predict_sentiment=predict_sentiment)
        # print('loaded data')


        # Load the data and the stims to do a leave two out cv.
        # Load the nifti, the word vectors, and the stim and then leave out two samples on which you'll do 2v2.

        # Write a function to do the leave-two-out cv. This returns the train and test indices.
        if leave_one_out_cv:
            train_indices, test_indices = leave_one_out(stims)
        else:
            train_indices, test_indices = leave_two_out(stims)

        ## [[[1,2,4,5], [6,7] ], [[2,4,5,6], [1, 7]], ....   ]
        # print('Decided indices')
        preds_list = []
        y_test_list = []
        i = 0
        start = time.time()
        r2_values = []
        for train_index, test_index in zip(train_indices, test_indices):
            print('Iteration: ', i)
            # if i == 4:
            #     break
            i += 1

            # model = Ridge(solver='cholesky')
            # ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
            # clf = GridSearchCV(model, param_grid=ridge_params, n_jobs=-1, scoring='neg_mean_squared_error', cv=8, verbose=5) # Setting cv=10 so that 4 samples are used for validation.
            # clf.fit(x[train_index], y[train_index])
            # preds = clf.predict(x[test_index])


            scaler = StandardScaler()
            if decoding:
                X_train = scaler.fit_transform(x[train_index])
                X_test = scaler.transform(x[test_index])
                y_train = y[train_index]
                y_test = y[test_index]
            else:
                X_train = scaler.fit_transform(y[train_index])
                X_test = scaler.transform(y[test_index])
                y_train = x[train_index]
                y_test = x[test_index]


            alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]
            # Uses LOOCV by default to tune hyperparameter tuning.
            cv_model = RidgeCV(alphas=alphas, gcv_mode='svd', scoring='neg_mean_squared_error', alpha_per_target=True)

            # if decoding:
            cv_model.fit(X_train, y_train)  # Decoding
            preds = cv_model.predict(X_test)
            # else:
            #     cv_model.fit(y_train, X_train)  # Encoding
            #     preds = cv_model.predict(y_test)


            # Store the preds in an array and all the ytest with the indices.

            preds_list.append(preds)
            y_test_list.append(y_test)
            if leave_one_out_cv == False:
                avg_r2.append(r2_score(preds, y_test))
            # else:
            #     print("Cannot calculate R-squared for less than two-samples.")


        if decoding:
            # If this is a decoding analysis, then use the following metrics.
            if metric == '2v2':
                if leave_one_out_cv == False:
                    accuracy, cosine_diff = two_vs_two(preds_list, y_test_list, store_cos_diff=store_cosine_diff)
                else:
                    # There are 16 total predictions. Use the extended 2v2 test.
                    accuracy, cosine_diff = extended_2v2(np.array(preds_list), np.array(y_test_list), store_cos_diff=store_cosine_diff)
                cosine_diff_dict[participant] = cosine_diff
                participant_accuracies[participant] = accuracy
            elif metric == 'corr':
                dim_corrs = get_dim_corr(preds_list, y_test_list)
                participant_correlations[participant] = dim_corrs
        else:
            # Encoding analysis, use the 2vs2 test but with euclidean distance.
            accuracy = extended_euclidean_2v2(np.array(preds_list), np.array(y_test_list))
            participant_accuracies[participant] = accuracy



        # cosine_diff_dict[participant] = cosine_diff
        #
        # participant_accuracies[participant] = accuracy
    if metric == '2v2':
        print("Participant Accuracies: ", participant_accuracies)
    else:
        print("Participant Correlations: ", participant_correlations)
    if leave_one_out_cv == False:
        print("Averaged r2: ", np.mean(avg_r2))

    if store_cosine_diff:
        np.savez_compressed("/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/debug_logs_files/cosine_diffs_2v2.npz", cosine_diff_dict)
    if permuted:
        # Save the permutation test results.
        timestr = time.strftime("%Y%m%d-%H%M%S")
        np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/results/permuted/P{participant}/{participant}_{timestr}.npz",participant_accuracies)


    # Code for storing the violin plots.
    if metric == 'corr':
        for p, corrs in participant_correlations.items():
            fig = get_violin_plot(p, corrs)
            fig.savefig(f"graphs/violin plots/{p}_beta_dict_{embedding_type}.png")
    stop = time.time()
    print('Total time: ', stop - start)


cross_validation_nested(decoding=True, avg_w2v=False, mean_removed=False, load_avg_trs=False, masked=True, permuted=False, store_cosine_diff=False, nifti_type='wrf',
                        beta=True, beta_mask_type='roi', embedding_type='sixty_w2v', metric='2v2', leave_one_out_cv=True,
                        predict_sentiment=False)

# parts = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019, 1024]
# parts = [1003, 1006, 1008, 1010]
# parts = [1016]
# for p in parts:
#     print("Participant: ", p)
#     # try:
#     store_betas_spm(p, 'sentiment', mask_type='gm')
    # except Exception as e:
    #     print("Participant not found or something: ", e)
    #     pass
# store_trs_fsl(1012, 'sentiment', remove_mean=False)