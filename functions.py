import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nilearn import image as img


def get_avg_trs():
    # Get avg TRs for all participants from GDrive.
    pass
    

def store_avg_tr(participant, nifti_path):

    # Now avg the TRs for each stimuli and for each corresponding 'run' and for each stimuli.
    tr_meta_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\" + str(participant) + "_TRsToUse.xlsx"
    metadata = pd.read_excel(tr_meta_path) # Remove this in the final version.


    stims =  metadata.iloc[:,[0,2,7]].values
    # Keep only stims with two words.
    stims_two_words = [val for val in stims if val[0].count(' ')==1]

    #---------------------------------------------------------------
    # Now retrieve from the nifti file.
    np_a = img.get_data(nifti_path)

    # Now preprocess the nifti and store it later in a directory called 'avg_tr_nifti' on GDrive.
    np_a = np.transpose(np_a, (3, 0, 1, 2))

    np_a_rs = np_a.reshape(np_a.shape[0], -1)

    mean_voxels_a = np.mean(np_a_rs, axis=0)

    # Removing the mean voxel value.
    for i, row in enumerate(np_a_rs):
        np_a_rs[i] = np_a_rs[i] - mean_voxels_a


    # Now map the nifti indexes in the TRsToUse excel file to the fmri data.
    run1_nifti = {}
    run2_nifti = {}
    # Could be optimized but it's okay for now.
    for stim in stims_two_words:
        run1_nifti[stim[0]] = []
        run2_nifti[stim[0]] = []

    # Store the corresponding fMRI files using the TR numbers from the excel sheet.
    for stim in stims_two_words:
        if stim[1] == 1:
            run1_nifti[stim[0]].append(np_a_rs[stim[2]])
        else:
            run2_nifti[stim[0]].append(np_a_rs[stim[2]])

    # Now average the TRs.
    run1_nifti_avg = {}
    run2_nifti_avg = {}

    for key, val in run1_nifti.items():
        run1_nifti_avg[key] = np.mean(val, axis=0)
    for key, val in run2_nifti.items():
        run2_nifti_avg[key] = np.mean(val, axis=0)

    # Now concatenate the TRs for each stim.
    concat_TR = {}
    for key, val in run1_nifti_avg.items():
        # Make sure run1 and run2 exists for all the two-word stims.
        if key in run1_nifti_avg.keys() and key in run2_nifti_avg.keys():
            if type(run1_nifti_avg[key]) == np.ndarray and type(run2_nifti_avg[key]) == np.ndarray:
                # print(key)
                tr_run1 = np.reshape(run1_nifti_avg[key],(-1, run1_nifti_avg[key].shape[0]))
                tr_run2 = np.reshape(run2_nifti_avg[key],(-1, run2_nifti_avg[key].shape[0]))
                concat_TR[key] = np.concatenate((tr_run1, tr_run2), axis=1)


    # Now store the values.
    np.savez_compressed(f'E:\My Drive\CoMLaM_rohan\CoMLaM\\avg_trs_concat\Synonym_RunB\\P_{participant}B.npz', concat_TR)

    # Load like this.
    # temp = np.load('E:\My Drive\CoMLaM_rohan\CoMLaM\\avg_trs_concat\\P_1003.npz', allow_pickle=True)['arr_0'].tolist()


def map_stimuli_w2v(participant):
    tr_meta_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\" + str(participant) + "_TRsToUse.xlsx"
    metadata = pd.read_excel(tr_meta_path)  # Remove this in the final version.

    stims = metadata.iloc[:, 0].values
    # Keep only stims with two words.
    stims_two_words = [val for val in stims if val.count(' ')==1]
    stims_two_words = list(set(stims_two_words))

    return stims_two_words


def load_nifti_and_w2v(participant, synonym_condition):
    """

    :param participant: The particpant for which the fMRI data needs to be loaded. Takes an integer.
    :param synonym_condition: The synonym condition for which to load the fMRI data. Takes either strings 'A' or 'B'.
    :return: the nifti file for the participant and the corresponding condition.
    """
    path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\avg_trs_concat\\"
    nifti_path = path + f"Synonym_Run{synonym_condition}\\P_{participant}{synonym_condition}.npz"
    nifti_data = np.load(nifti_path, allow_pickle=True)['arr_0'].tolist()

    w2v_path = "G:\comlam\embeds\\two_words_stim_w2v_concat_dict.npz"
    w2v_data = np.load(w2v_path, allow_pickle=True)['arr_0'].tolist()

    # Now map the nifti data to the corresponding concatenated w2v vectors.
    x_data = []
    y_data = []

    for stim, fmri in nifti_data.items():
        x_data.append(fmri)
        y_data.append(w2v_data[stim])

    return x_data, y_data




def load_stim_vectors(participant):
    pass