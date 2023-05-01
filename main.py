import pandas as pd
import numpy as np
import math
from nilearn import image as img
import pickle as pk
import matplotlib.pyplot as plt
import os
import sys
import glob
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegressionCV
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

from functions import store_avg_tr, map_stimuli_w2v, load_nifti_and_w2v, list_diff, \
    two_vs_two, store_trs_spm, store_trs_fsl, leave_two_out, store_masked_trs_spm, store_betas_spm, get_dim_corr, leave_one_out, extended_2v2, \
    get_violin_plot, extended_euclidean_2v2, load_nifti, load_y, load_nifti_by_run
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split, GridSearchCV

# Load custom scripts.
from exp_funcs import congruency_classification, sentiment_classification, decoding_analysis, encoding_analysis

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


def classify_sentiment_or_congruency_or_decoding(participant=None, run=4, type='sentiment',
                                     iters=50, loocv=False, permuted=False, brain_type='wholeBrain', way='3', 
                                     use_nc=False, remove_neutral=False, embedding_type='bertweet', avg_w2v=False, do_corr=False, correlation_store_file_prefix='bertweet', non_zero_shot_folder_path=''):
    """

    :param participant: participant number.
    :param run: beta run value
    :param type: 'sentiment' or 'congruency'
    :param iters: CV iterations.
    :param loocv: Default=False. Whether to use loocv. Ignored for type='congruency'.
    :param permuted: Whether to permute the labels for permutation test.
    :return: Accuracy score for the experiment.
    """

    # Load data for the given participant and the corresponding run.

    # Based on the experiment type, call the appropriate function from the exp_funcs.py file.
    if type == 'decoding':
        if loocv:
            # If doing leave one out cross-validation.
            if not use_nc:
                x_data = load_nifti_by_run(participant, type=brain_type, run=run)
                y_data = load_y(participant=participant, embedding_type=embedding_type, sentiment=False, congruent=False, avg_w2v=avg_w2v)
            if use_nc:
                X = y = None
                return decoding_analysis(X=X, y=y, participant=participant, loocv=loocv, iters=iters, permuted=permuted, use_nc=use_nc)
        else:
            # Not doing the loocv. This is the non-zero-shot analysis.
            assert non_zero_shot_folder_path != ''
            # Align the x_data and the y_data here based on their keys. Assign them separately otherwise considers them as the same object.
            # Load the data from the specific folder.
            x_data = np.load(non_zero_shot_folder_path, allow_pickle=True)['arr_0'].tolist()
            y_data = load_y(participant=participant, embedding_type=embedding_type, sentiment=False, congruent=False, avg_w2v=avg_w2v)
            x = []
            y = []
            for stim in x_data.keys():
                x.extend(x_data[stim])
                y.append(y_data[stim])    
            # Repeat the numer of the word embeddings to match the length of x_data.
            y = np.repeat(y, len(x) // len(y), axis=0)
            assert len(y) == len(x)
        
        if loocv:
            # Align the x_data and the y_data here based on their keys. Assign them separately otherwise considers them as the same object.
            x = []
            y = []

            for stim in x_data.keys():
                x.append(x_data[stim])
                y.append(y_data[stim])


        X = np.array(x)
        print("X shape: ", X.shape)
        print("Y shape: ", y.shape)
        

        if permuted:
            np.random.shuffle(y)

        # Do pca on the embeddings.
        # First scale the embeddings.
        # Then select 20 components and transform the embeddings.

       
        return decoding_analysis(X=X, y=y, participant=participant, loocv=loocv, iters=iters, permuted=permuted)

    elif type == 'encoding':
        print("Doing encoding")
        if not use_nc:
            x_data = load_nifti_by_run(participant, type=brain_type, run=run)
            y_data = load_y(participant=participant, embedding_type=embedding_type, sentiment=False, congruent=False, avg_w2v=avg_w2v)            
        if use_nc:
            X = y = None
            return encoding_analysis(X=X, y=y, participant=participant, loocv=loocv, iters=iters, permuted=permuted, use_nc=use_nc, brain_type=brain_type)

        # Align the x_data and the y_data here based on their keys. Assign them separately otherwise considers them as the same object.
        x = []
        y = []

        for stim in x_data.keys():
            x.append(x_data[stim])
            y.append(y_data[stim])
        
        X = np.array(x)
        y = np.array(y)
        print("X shape: ", X.shape)
        print("Y shape: ", y.shape)


        if permuted:
            np.random.shuffle(y)

        # Do pca on the embeddings.
        # First scale the embeddings.
        # Then select 20 components and transform the embeddings.

       
        return encoding_analysis(X=X, y=y, participant=participant, loocv=loocv, iters=iters, permuted=permuted, do_corr=do_corr, correlation_prefix_file_name=correlation_store_file_prefix)

    elif type == 'sentiment':
        # Do sentiment classification.
        # Load the data.
        if not use_nc:
            x_data = load_nifti_by_run(participant, type=brain_type, run=run)
            y_data = load_y(participant=participant, sentiment=True, congruent=False) 
        else:
            X = y = None
            return sentiment_classification(X=X, y=y, participant=participant, loocv=loocv, iters=iters, permuted=permuted)
        x = []
        y = []

        for stim in x_data.keys():
            x.append(x_data[stim])
            y.append(y_data['labels'][stim])

        X = np.array(x)
        y = np.array(y)
        
        

        if permuted:
            np.random.shuffle(y)
        # Now keep only those stims which have positive or negative sentiment.

        if remove_neutral:
            print("Removing neutral")
            removed_idxs = []
            for idx, sent in enumerate(y):
                if sent == 'Neut':
                    removed_idxs.append(idx)
            X = np.delete(X, removed_idxs, axis=0)
            y = np.delete(y, removed_idxs, axis=0)


        encoder = LabelEncoder()
        y_transformed = encoder.fit_transform(y)

        return sentiment_classification(X=X, y=y_transformed, participant=participant, loocv=loocv, iters=iters, permuted=permuted)

    elif type == 'congruency':
        if not use_nc:
            x_data = load_nifti_by_run(participant, brain_type, run=run)
            y_data = load_y(participant=participant, sentiment=False,
                        congruent=True, way=way)
        else:
            X = y = None
            return congruency_classification(X=X, y=y_transformed, participant=participant, loocv=loocv, iters=iters, permuted=permuted)

        x = []
        y = []

        for stim in y_data['labels'].keys():
            x.append(x_data[stim])
            y.append(y_data['labels'][stim])

        X = np.array(x)
        y = np.array(y)

        if permuted:
            np.random.shuffle(y)
        # Now keep only those stims which have positive or negative sentiment.
        if remove_neutral:
            removed_idxs = []
            for idx, sent in enumerate(y):
                if sent == 'Neut':
                    removed_idxs.append(idx)
            X = np.delete(X, removed_idxs, axis=0)
            y = np.delete(y, removed_idxs, axis=0)


        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        

        # The following does stratified shuffle split with iters splits.
        return congruency_classification(X=X, y=y, participant=participant, iters=iters, permuted=permuted)



# def do_decoding():



def across_congruent_cv(participant=None, run=4, train_type='con',
                        test_type='inc', metric='2v2',
                        permutation=False, iterations=1,
                        observed_acc=1.0):
    # Load train data.
    # if train_type == 'con':
    #     test_type = 'inc'
    # else:
    #     test_type = 'con'


    x, y, stims = load_nifti_and_w2v(participant, avg_w2v=False, mean_removed=False, load_avg_trs=False,
                                     masked=True, permuted=False,
                                     beta=False, beta_mask_type='wholeBrain', embedding_type='sixty_w2v',
                                     predict_sentiment=False, run=run,
                                     whole_brain=True, priceNine=False)


    # First get the training data for one congruency type.
    train_idx = []
    test_idx = []
    trs_to_use = pd.read_excel(
        f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{1014}_2k/sentiment/TRsToUse_AlphaOrder_P{1014}_2.xlsx")
    trs_groups = trs_to_use.groupby(by=['combinedStim'])
    for idx, row in enumerate(trs_groups):
        stim = row[0]
        sent = row[1].iloc[0]['Polarity'][-3:]
        if sent == train_type.title():
            train_idx.append(idx)
        if sent == test_type.title():
            test_idx.append(idx)
    X_train = x[train_idx]
    y_train = y[train_idx]

    X_test = x[test_idx]
    y_test = y[test_idx]

    if permutation:
        np.random.shuffle(y_train)


    # Now let's do a simple classification.
    alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]
    if permutation:
        perm_accs = []
    for iters in range(iterations):
        print("Iteration: ", iters)
        cv_model = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', alpha_per_target=True)

        cv_model.fit(X_train, y_train)  # Decoding
        preds = cv_model.predict(X_test)



        if metric == '2v2':
            accuracy, cosine_diff = extended_2v2(preds, y_test)
            if permutation:
                perm_accs.append(accuracy)
    if permutation:
        print("Number of times above obs_acc: ", np.sum(np.array(perm_accs) > observed_acc))
        p_val = np.sum(np.array(perm_accs) > observed_acc) / len(perm_accs)
        print("Average permutation accuracy: ", np.mean(perm_accs))

        # print(f" 2 vs 2 Accuracy: {accuracy}")


    if permutation:
        return p_val
    return np.round(accuracy,2)







def cross_validation_nested(decoding=True, part=None, avg_w2v=False, mean_removed=False, load_avg_trs=False, masked=False, permuted=False ,store_cosine_diff=False, nifti_type='rf',
                            beta=True, beta_mask_type='gm', embedding_type='w2v', metric='2v2', leave_one_out_cv=False, sentiment=False, congruent=False,
                            pca=False, pca_brain = False, pca_vectors = False, pca_threshold=0.95, show_variance_explained=True,
                            iterations=1, scale_target=False, run=10, whole_brain=False, priceNine=False, motor=False, divide_by_congruency=False, congruency_type='con',
                            observed_acc=1.0):
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
    :param sentiment: Whether to predict the sentiment vectors.
    :return: None
    """
    # Do ridge regression with GridSearchCV here.
    # Run the analysis for each participant here.

    participant_accuracies = {}
    cosine_diff_dict = {}
    participant_correlations = {}
    participant_corrs_means = []
    participant_pvals = {}
    avg_r2 = []

    if type(part) == list:
        participants = part
    else:
        # participants = [1004, 1006, 1007, 1008, 1010, 1012, 1016, 1017, 1019]
        # participants = [1014, 1030, 1005, 1033]#, 100436, 1007, 1008, 1010, 1012, 1016, 1017, 1019, 1024]
        participants = []#[1030, 1005, 1033]

    for participant in participants:
        print(participant)
        observed_acc = observed_acc
        iter_acc = []
        participant_corrs_means = []
        for iter in range(iterations):
            # print("Iteration: ", iter)



            if sentiment or congruent:
                x_data = load_nifti(participant, load_avg_trs=load_avg_trs, beta=beta, beta_mask_type=beta_mask_type, masked=masked)
                y_data = load_y(participant=participant, embedding_type=embedding_type, avg_w2v=avg_w2v, sentiment=sentiment, congruent=congruent)
                x = []
                y = []

                for stim in y_data['stims'].values():
                    x.append(x_data[stim])
                    y.append(y_data[stim])

                x = np.array(x)
                y = np.array(y)

                if permuted:
                    np.random.shuffle(y)
                # Now keep only those stims which have positive or negative sentiment.

                stims = [s for s in y_data['stims'].values()]


                encoder = LabelEncoder()
                y = encoder.fit_transform(y)



            else:
                x, y, stims = load_nifti_and_w2v(participant, avg_w2v=avg_w2v, mean_removed=mean_removed, load_avg_trs=load_avg_trs, masked=masked, permuted=permuted,
                                                 beta=beta, beta_mask_type=beta_mask_type, embedding_type=embedding_type, predict_sentiment=False, run=run,
                                                 whole_brain=whole_brain, priceNine=priceNine, motor=motor)
                # 'stims' has the same order as that of the niftis.

                if divide_by_congruency:
                    x_mod = []
                    trs_to_use = pd.read_excel(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_2k/sentiment/TRsToUse_AlphaOrder_P{participant}_2.xlsx")
                    trs_groups = trs_to_use.groupby(by=['combinedStim'])
                    for idx, row in enumerate(trs_groups):
                        stim = row[0]
                        sent = row[1].iloc[0]['Polarity'][-3:]
                        if sent == congruency_type.title():
                            x_mod.append(idx)
                    x = x[x_mod]
                    y = y[x_mod]
                    stims = np.array(stims)
                    stims = stims[x_mod]
                    stims = stims.tolist()



                # print("Brain shape: ", x.shape)




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
            x_explained_variance_ratio = []
            y_explained_variance_ratio = []

            for train_index, test_index in zip(train_indices, test_indices):
                # print('Index iteration: ', i)
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
                    if scale_target:
                        scaler_target = StandardScaler()
                        y_train = scaler_target.fit_transform(y_train)
                        y_test = scaler_target.transform(y_test)
                    if pca:
                        if show_variance_explained:

                            pca = PCA(n_components=pca_threshold)
                            pca.fit(X_train)
                            var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
                            var1 = pca.explained_variance_ratio_
                            x_explained_variance_ratio.append(var)

                            # pca = PCA(n_components=pca_threshold)
                            # pca.fit(y_train)
                            # var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
                            # var1 = pca.singular_values_
                            # y_explained_variance_ratio.append(var1)


                        if pca_brain:
                            # print("PCA Brain")
                            pca = PCA(n_components=pca_threshold, random_state=42)
                            X_train = pca.fit_transform(X_train)
                            X_test = pca.transform(X_test)
                        if pca_vectors:
                            # print("PCA Vectors")
                            pca = PCA(n_components=pca_threshold, random_state=42)
                            y_train = pca.fit_transform(y_train)
                            y_test = pca.transform(y_test)




                else:
                    X_train = scaler.fit_transform(y[train_index])
                    X_test = scaler.transform(y[test_index])
                    y_train = x[train_index]
                    y_test = x[test_index]


                alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10]
                logreg_alphas = [10000000, 1000000, 100000, 10000, 1000, 100, 10, 2, 1, 0.2, 0.1]
                # Uses LOOCV by default to tune hyperparameter tuning.
                if sentiment or congruent:
                    cv_model = LogisticRegressionCV(Cs=logreg_alphas, cv=5, max_iter=1000)
                else:
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

            if show_variance_explained:
                avg_x_var_explained = np.mean(x_explained_variance_ratio, axis=0)
                avg_y_var_explained = np.mean(y_explained_variance_ratio, axis=0)
                # return avg_x_var_explained

                plt.clf()
                plt.plot(avg_x_var_explained)
                plt.title("Percent of variance explained for fMRI.")
                plt.ylabel("Proportion of variance explained")
                plt.ylabel("Singular Values")
                plt.xlabel("Number of components")
                # plt.axvline(np.where(avg_x_var_explained >=95.0)[0][0], 0, 1)
                plt.show()

                plt.clf()
                plt.plot(avg_y_var_explained)
                plt.title("Percent of variance explained for vectors (w2v average).")
                plt.ylabel("Percent of variance explained")
                plt.xlabel("Number of components")
                plt.axvline(np.where(avg_y_var_explained >= 95.0)[0][0], 0, 1)
                plt.show()




            if decoding:
                # If this is a decoding analysis, then use the following metrics.
                if metric == 'accuracy':
                    accuracy = accuracy_score(y_test_list, preds_list)
                    iter_acc.append(accuracy)
                elif metric == '2v2':
                    if leave_one_out_cv == False:
                        accuracy, cosine_diff = two_vs_two(preds_list, y_test_list, store_cos_diff=store_cosine_diff)
                        iter_acc.append(accuracy)
                    else:
                        # There are 60 total predictions. Use the extended 2v2 test.
                        accuracy, cosine_diff = extended_2v2(preds_list, y_test_list, store_cos_diff=store_cosine_diff)
                    if store_cosine_diff:
                        cosine_diff_dict[participant] = cosine_diff
                    iter_acc.append(accuracy)
                    print("Accuracy: ", accuracy)
                    print("Iter acc:", iter_acc)
                elif metric == 'corr':
                    dim_corrs = get_dim_corr(preds_list, y_test_list)
                    participant_correlations[participant] = np.mean(dim_corrs)
                    if permuted:
                        participant_corrs_means.append(np.mean(dim_corrs))

            else:
                # Encoding analysis, use the 2 vs 2 test but with euclidean distance.
                accuracy = extended_euclidean_2v2(np.array(preds_list), np.array(y_test_list))
                participant_accuracies[participant] = accuracy

            # Setting the mean participant accuracy across all the iterations.





        participant_accuracies[participant] = np.mean(iter_acc)

            # print(f"Iteration {iter} r2 score: ", r2_score(preds_list, y_test_list))


        # Do the p-value calculation here. Count how many permuted accuracies are greater than the observed accuracies.
        # print("np.sum iter_acc > observed_acc", np.sum(np.array(iter_acc) > observed_acc))
        if metric == '2v2':
            participant_pvals[participant] = np.sum(np.array(iter_acc) > observed_acc) / len(iter_acc)
        # participant_pvals[participant] = (np.sum(np.array(participant_corrs_means)  > observed_acc)) / len(participant_corrs_means)
        #
        # print(f"Participant {participant} p-values over {iterations} permutation iterations: ", participant_pvals)




        # cosine_diff_dict[participant] = cosine_diff
        #
        # participant_accuracies[participant] = accuracy
    if metric == '2v2' or metric == 'accuracy':
        print(f"Permuted={permuted} Mean participant accuracy over {iterations} iterations: ", participant_accuracies)
        participant_accuracies_list = [v for v in participant_accuracies.values()]
        for a in participant_accuracies_list:
            print("{:.2f}".format(a*100))

        if permuted:
            print(f"Participant p-values over {iterations} permutation iterations: ", participant_pvals)
    else:
        print("Participant Correlations: ", participant_correlations)
        print("Participant Correlations: ", participant_pvals)
    if leave_one_out_cv == False:
        print("Averaged r2: ", np.mean(avg_r2))

    if store_cosine_diff:
        np.savez_compressed("/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/debug_logs_files/cosine_diffs_2v2.npz", cosine_diff_dict)
    # if permuted:
    #     # Save the permutation test results.
    #     timestr = time.strftime("%Y%m%d-%H%M%S")
    #     np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/results/permuted/P{participant}/{participant}_{timestr}.npz",participant_accuracies)


    # Code for storing the violin plots.
    # if metric == 'corr':
    #     for p, corrs in participant_correlations.items():
    #         fig = get_violin_plot(p, corrs)
    #         fig.savefig(f"graphs/violin plots/{p}_beta_dict_{embedding_type}.png")
    stop = time.time()
    print('Total time: ', stop - start)

    if metric == 'corr':
        return participant_correlations[participant]
    return participant_accuracies[participant]

# runs = [4, 5, 6, 7, 8, 9, 10]
# # runs = [4,5,6,7,8]
# # runs = [6]
# variances_explained = []
# parts = [1030, 1032]#, 1030, 1032, 1038]
# p_accs = {}
# for p in parts:
#     p_accs[p] = {}

# # print(p_accs)
# for p in parts:
#     for run in runs:
#         print("CV betas made with runs: ", run)
#         score = cross_validation_nested(decoding=True, part=[p], avg_w2v=False, mean_removed=False, load_avg_trs=False, masked=True, permuted=False, store_cosine_diff=False, nifti_type='wrf',
#                                 beta=False, beta_mask_type='wholeBrain', embedding_type='sixty_w2v', metric='2v2', leave_one_out_cv=True,
#                                 sentiment=False, congruent=False, pca=True, pca_brain=False, pca_vectors=True, pca_threshold=0.95, show_variance_explained=False,
#                                 iterations=50, scale_target=True, run=run, whole_brain=False, priceNine=False, motor=True, divide_by_congruency=False, congruency_type='inc', observed_acc=0.061)
#         # score = across_congruent_cv(participant=p, run=run, train_type='inc', test_type='con',metric='2v2', permutation=True, iterations=50, observed_acc=0.66)
#         p_accs[p][run] = score
#     print("All accuracies:", p_accs)
    # print(var_ex_brain)
    # variances_explained.append(var_ex_brain)

#
# for varx, run in zip(variances_explained, runs):
#     plt.plot(varx, label=run)
# plt.title("PCA on brain for different runs: PriceNine")
# plt.legend()
# plt.show()



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


# runs = [4, 5, 6, 7, 8, 9, 10]

# args = argparse.ArgumentParser(description='Run complam analysis with arguments. Do not use quotes for the arguments. Currently the code runs for participants 1014')
# requiredArgs.add_argument("-perm", "--permutation", help="Boolean: True or False. Whether to run permutation test or not.", required=True)
# requiredArgs.add_argument("-w", "--way", help="Integer. 2 or 3. For congruency and sentiment classification. What way classification to run.", required=False)
# requiredArgs.add_argument("-e", "--embedding", help="bertweet or avg_w2v or concat_w2v. Required if --analysis is decoding or encoding", required=False)
# requiredArgs.add_argument("-a", "-analysis", help="Analysis type: decoding or encoding or classification.", required=True)
# requiredArgs.add_argument("-b","--brain_type", help="wholeBrain, priceNine, or motor. Brain type to use.", required=True)
# requiredArgs.add_argument("-exp", "--exp_type", help="sentiment or congruency. Required only when doing classification analysis. Default is sentiment", required=False)
# requiredArgs.add_argument("-rn", "--remove_neutral", help="Boolean. Whether to remove neutral stimulus examples from the dataset when doing sentiment or congruency classification. Default=False". required=False)
# requiredArgs.add_argument("-part", "--participant", help="Participant to run the analysis for. Select one the list [1014, 1030, 1032, 1037, 1038]", required=True)
# requiredArgs.add_argument("-r", "--run", help="all or specify a run number (to run the analysis for the specific titration run). Example -r=all or -r=4.", required=True)

runs = [10]
variances_explained = []
# parts = [1014, 1030, 1032, 1038, 'Pilot_08Feb23']
# parts = [1014]
parts = ['Pilot_08Feb23']
# parts = [int(sys.argv[2])]
p_accs = {}
for p in parts:
    p_accs[p] = {}



# Arguments, currently being set manually.
permutation = False
print("Permuted: ", permutation)
iters = 50
exp_type = 'decoding' #sys.argv[4]  # 'congruency', 'sentiment','decoding', or 'encoding'
loocv = False #if sys.argv[5] == 'False' else True  # This controls if you're doing zero-shot or not.
brain_type = 'wholeBrain'#sys.argv[1]  # 'wholeBrain', 'priceNine', or 'motor' passing this as a command-line argument.
way = '3'#sys.argv[3]
use_nc = False
remove_neutral = False
embedding_type = 'sixty_w2v' # sixty_w2v or bertweet or twitter_w2v
avg_w2v = False # False if you wanna use concatenated word2vec or True for averaged word2vec.
do_corr = False
correlation_prefix_file_name = None
if do_corr:
    if embedding_type == 'sixty_w2v':
        correlation_prefix_file_name = 'concat_w2v'
        if avg_w2v:
            correlation_prefix_file_name = 'avg_w2v'
        
non_zero_shot_folder_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/other/Pilot_08Feb23/NonZeroShot/Twos/Pilot_08Feb23_beta_dict.npz'#str(sys.argv[7])
            

print(exp_type)
print("way: ", way)
if use_nc:
    print("Using noise ceiling")
    if permutation:
        iter_scores = classify_sentiment_or_congruency_or_decoding(participant=p, type=exp_type, iters=iters, loocv=loocv, permuted=permutation, brain_type=brain_type, way=way, use_nc=use_nc)
        p_accs[p]= iter_scores
    else:
        score = classify_sentiment_or_congruency_or_decoding(participant=p, type=exp_type, iters=iters, loocv=loocv, permuted=permutation, brain_type=brain_type, way=way, use_nc=use_nc)
        p_accs[p] = score
    print("All accuracies:", p_accs)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/saved_results/permutation/{exp_type}/{p}/{timestr}_iter_scores_{way}_way.npz", p_accs)

    
else:
    print("Not using noise ceiling")
    for p in parts:
        if p != 1037:
            print("Participant: ", p)
            runs = [4, 5, 6, 7, 8, 9, 10]
            # runs = [9,10]
            # runs = [4, 5]
            # runs = [6, 7]
            # runs = [8, 9]
            # runs = [int(sys.argv[6])]0.006254762927217605
        else:
            runs = [4, 5, 6, 7, 8]
        for run in runs:
            print("CV betas made with runs: ", run)
            print("Calling classify sentiment, congruency, and decoding function")
            if permutation:
                iter_scores = classify_sentiment_or_congruency_or_decoding(participant=p, run=run, type=exp_type, iters=iters, loocv=loocv, permuted=permutation, brain_type=brain_type, way=way, remove_neutral=remove_neutral, embedding_type=embedding_type, avg_w2v=avg_w2v, do_corr=do_corr, correlation_store_file_prefix=correlation_prefix_file_name)
                p_accs[p][run] = iter_scores
            else:
                score = classify_sentiment_or_congruency_or_decoding(participant=p, run=run, type=exp_type, iters=iters, loocv=loocv, permuted=permutation, brain_type=brain_type, way=way, remove_neutral=remove_neutral, embedding_type=embedding_type, avg_w2v=avg_w2v, do_corr=do_corr, correlation_store_file_prefix=correlation_prefix_file_name, non_zero_shot_folder_path=non_zero_shot_folder_path)
                p_accs[p][run] = score
                # print("All accuracies:", p_accs)
        if permutation:
            # Save the permutation test scores as an npz file.
            timestr = time.strftime("%Y%m%d-%H%M%S")
            r = np.random.randint(0,10000,1)[0]
            np.savez_compressed(f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/saved_results/permutation/{exp_type}/{p}/{timestr}_{r}_{brain_type}_iter_scores_{exp_type}_{embedding_type}_way_{way}_avg_w2v_{str(avg_w2v)}.npz", p_accs)
        else:
            print("All accuracies:", p_accs)
    print("All accuracies:", p_accs)