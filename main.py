import pandas as pd
import numpy as np
import math
from nilearn import image as img
import pickle as pk
import matplotlib.pyplot as plt
import os
import glob
import regex as re
from functions import store_avg_tr, map_stimuli_w2v
from gensim.models import KeyedVectors


def avg_trs():
    # Function to store Averaged concatenated TRs on GDrive.
    nifti_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\Preprocessed\Reg_to_Std_and_Str\\"
    tr_meta_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\"
    participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
    for participant in participants:
        print(participant)
        file_name = nifti_path + "P_" + str(participant) + "\\"
        for file in glob.glob(file_name + "Synonym_RunB*\\filtered_func_data.nii"):
            store_avg_tr(participant, file)


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


def ridge():
    # Do ridge regression with GridSearchCV here.
    # Run the analysis for each participant here.
    participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]


    for participant in participants:
        nifti_data = load_nifti(participant)
        tr_meta_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\1003_TRsToUse.xlsx"
        nifti_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\Preprocessed\Reg_to_Std_and_Str\P_1003\Synonym_RunA_13.feat\\filtered_func_data.nii"
        participant_tr = get_avg_tr(tr_meta_path, nifti_path)
