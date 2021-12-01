import pandas as pd
import numpy as np
import math
from nilearn import image as img
import pickle as pk
import matplotlib.pyplot as plt
import os
import glob
import regex as re
from functions import store_avg_tr, map_stimuli_w2v, load_nifti_and_w2v
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split


def avg_trs():
    # Function to store Averaged concatenated TRs on GDrive.
    nifti_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\Preprocessed\Reg_to_Std_and_Str\\"
    participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
    for participant in participants:
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


def cross_validation_nested(part, synonym_condition):
    """
    :param part: Accepts a list of participants. Example: [1003, 1006]
    :param synonym_condition: [The synonym task. Accepts 'A' or 'B']
    :return: 2v2 accuracy for the participant.
    """
    # Do ridge regression with GridSearchCV here.
    # Run the analysis for each participant here.

    if type(part) == list:
        participants = part
    else:
        participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
    for participant in participants:
        x, y, stims = load_nifti_and_w2v(participant, synonym_condition)

        # Load the data and the stims to do a leave two out cv.
        # Load the nifti, the word vectors, and the stim and then leave out two samples on which you'll do 2v2.

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)




