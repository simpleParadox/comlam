import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nilearn import image as img
from scipy.spatial.distance import cosine
import platform
import glob

def store_trs_spm(participant, task, remove_mean=False):

    """
    Stores the avg and concatenated files for each stimuli for each participant.
    :param participant: The number of participant. Accepts a single integer. E.g. 1003.
    :param task: Either 'sentiment' or 'synonym'
    :return: None

    """
    # participant = 1012  # Remove this when not testing.
    # task = 'sentiment'

    # Define the paths for GDrive.
    spm_path = "E:\.shortcut-targets-by-id\\1R-Ea0u_BCBsGnEX6RJL09tfLMV_hmwWe\CoMLaM\Preprocessed\SPM\\"
    participant_path = spm_path + "P" + str(participant) + "\\" + task + "\\"
    tr_meta_path = "E:\.shortcut-targets-by-id\\1R-Ea0u_BCBsGnEX6RJL09tfLMV_hmwWe\CoMLaM\\" + str(participant) + "_TRsToUse.xlsx"
    metadata = pd.read_excel(tr_meta_path)
    run1_path = participant_path + "run01\\"
    run2_path = participant_path + "run02\\"


    # Select only the stimuli that has two words only.
    ## Find the rows where the stim is more than two words.
    non_two_word_phrases_idx = [i for i in range(len(metadata)) if metadata.iloc[i,0].count(' ') > 1]
    two_stim_metadata = metadata.drop(labels=non_two_word_phrases_idx, axis=0)

    # First group the TR_metadata by stimuli.
    grouped_metadata = two_stim_metadata.groupby(['words'])

    # Then each row and read corresponding file name.

    # Load all the fMRI data to remove the mean voxels from each of the stored fMRI file.

    nifti_run_1_all_files = []
    nifti_run_2_all_files = []
    if remove_mean == True:
        for file in glob.glob(run1_path + "*.nii"):
            nifti = img.get_data(file)
            nifti_rs = np.array(nifti.reshape(1, -1))
            nifti_run_1_all_files.append(nifti_rs)

        for file in glob.glob(run2_path + "*.nii"):
            nifti = img.get_data(file)
            nifti_rs = np.array(nifti.reshape(1, -1))
            nifti_run_2_all_files.append(nifti_rs)

        mean_voxels_a = np.mean(nifti_run_1_all_files, axis=0)
        mean_voxels_b = np.mean(nifti_run_2_all_files, axis=0)


    concat_tr = {}
    counter = 0
    for row in grouped_metadata:
        print(counter)
        counter += 1
        # 'row' is a tuple.
        stim = row[0]
        df = row[1] # Second index is the dataframe for the stimulus.
        runs = df['run'].values
        nifti_idx = df['Nifti'].values

        # Now do the retrieval of fMRI data here and averaging and concatenating.
        j = 0
        k = 1
        first_point_nifti_files = []
        second_point_nifti_files = []
        while(j < 7 and k < 8):
            run_j = runs[j]
            nifti_j = int(nifti_idx[j])

            run_k = runs[k]
            nifti_k = int(nifti_idx[k])

            left_j_1 = str(nifti_j).zfill(5)
            left_j_2 = str(nifti_j).zfill(6)

            left_k_1 = str(nifti_k).zfill(5)
            left_k_2 = str(nifti_k).zfill(6)

            # print("Nifti j", nifti_j)
            # print('Nifti k', nifti_k)


            if run_j == 1:
                file_name = glob.glob(run1_path + f"*{left_j_1}-{left_j_2}-*.nii")
                # print("Run j=1 ", file_name)
                data_j_1 = img.get_data(file_name)
                # Now store the file in an array.
                np_a_j_1 = np.transpose(data_j_1, (3, 0, 1, 2))
                np_a_rs_j_1 = np_a_j_1.reshape(np_a_j_1.shape[0], -1)
                if remove_mean == True:
                    np_a_rs_j_1 = np_a_rs_j_1 - mean_voxels_a
                first_point_nifti_files.append(np_a_rs_j_1)
            elif run_j == 2:
                # Read from the 'run02; folder
                file_name = glob.glob(run2_path + f"*{left_j_1}-{left_j_2}-*.nii")
                # print("Run j=2 ", file_name)
                data_j_2 = img.get_data(file_name)
                # Now store the file in an array for later averaging.
                np_a_j_2 = np.transpose(data_j_2, (3, 0, 1, 2))
                np_a_rs_j_2 = np_a_j_2.reshape(np_a_j_2.shape[0], -1)
                if remove_mean == True:
                    np_a_rs_j_2 = np_a_rs_j_2 - mean_voxels_b
                first_point_nifti_files.append(np_a_rs_j_2)

            if run_k == 1:
                file_name = glob.glob(run1_path + f"*{left_k_1}-{left_k_2}-*.nii")
                # print("Run k=1 ", file_name)
                data_k_1 = img.get_data(file_name)
                # Now store the file in an array.
                np_b_k_1 = np.transpose(data_k_1, (3, 0, 1, 2))
                np_a_rs_k_1 = np_b_k_1.reshape(np_b_k_1.shape[0], -1)
                if remove_mean == True:
                    np_a_rs_k_1 = np_a_rs_k_1 - mean_voxels_a
                second_point_nifti_files.append(np_a_rs_k_1)

            elif run_k == 2:
                file_name = glob.glob(run2_path + f"*{left_k_1}-{left_k_2}-*.nii")
                # print("Run k=2 ", file_name)
                data_k_2 = img.get_data(file_name)
                # Now store the file in an array.
                np_a_k_2 = np.transpose(data_k_2, (3, 0, 1, 2))
                np_a_rs_k_2 = np_a_k_2.reshape(np_a_k_2.shape[0], -1)
                if remove_mean == True:
                    np_a_rs_k_2 = np_a_rs_k_2 - mean_voxels_b
                second_point_nifti_files.append(np_a_rs_k_2)

            j += 2
            k += 2

        # Average and concatenate.
        first_point_avg = np.mean(first_point_nifti_files, axis=0)
        second_point_avg = np.mean(second_point_nifti_files, axis=0)
        concat_tr[stim] = np.concatenate((first_point_avg, second_point_avg), axis=1)
    if remove_mean == True:
        np.savez_compressed(f"G:\comlam\spm\sentiment\P{participant}_mean_removed.npz", concat_tr)
    else:
        np.savez_compressed(f"G:\comlam\spm\sentiment\P{participant}.npz", concat_tr)


def store_trs_fsl(participant, task, remove_mean=False):
    # participant = 1012
    # task = 'sentiment'
    fsl_path = "E:\.shortcut-targets-by-id\\1R-Ea0u_BCBsGnEX6RJL09tfLMV_hmwWe\CoMLaM\Preprocessed\FSL\\"
    participant_path = fsl_path + "P" + str(participant) + "\\" + task + "\\"
    tr_meta_path = "E:\.shortcut-targets-by-id\\1R-Ea0u_BCBsGnEX6RJL09tfLMV_hmwWe\CoMLaM\\" + str(participant) + "_TRsToUse.xlsx"
    metadata = pd.read_excel(tr_meta_path)
    run1_path = participant_path + "run01\\"
    run2_path = participant_path + "run02\\"

    # Select only the stimuli that has two words only.
    ## Find the rows where the stim is more than two words.
    non_two_word_phrases_idx = [i for i in range(len(metadata)) if metadata.iloc[i, 0].count(' ') > 1]
    two_stim_metadata = metadata.drop(labels=non_two_word_phrases_idx, axis=0)

    # First group the TR_metadata by stimuli.
    grouped_metadata = two_stim_metadata.groupby(['words'])


    np_a = img.get_data(run1_path + "filtered_func_data.nii")
    np_a = np.transpose(np_a, (3, 0, 1, 2))
    np_a_rs = np_a.reshape(np_a.shape[0], -1)
    mean_voxels_a = np.mean(np_a_rs, axis=0)

    np_b = img.get_data(run2_path + "filtered_func_data.nii")
    np_b = np.transpose(np_b, (3, 0, 1, 2))
    np_b_rs = np_b.reshape(np_b.shape[0], -1)
    mean_voxels_b = np.mean(np_b_rs, axis=0)

    if remove_mean == True:
        for i, row in enumerate(np_a_rs):
            np_a_rs[i] = np_a_rs[i] - mean_voxels_a
        for i, row in enumerate(np_b_rs):
            np_b_rs[i] = np_b_rs[i] - mean_voxels_b

    concat_tr = {}
    counter = 0

    for row in grouped_metadata:
        print(counter)
        counter += 1
        stim = row[0]
        df = row[1]
        runs = df['run'].values
        tr_nums = df['TRNum'].values  # Actually the file index in python is 1 less than the actual number in the list.

        j = 0
        k = 1
        first_point_nifti_files = []
        second_point_nifti_files = []
        while (j < 7 and k < 8):
            run_j = runs[j]
            tr_num_j = int(tr_nums[j]) - 1 # Subtracting 1 because the data processed in MatLab was 1-indexed.

            run_k = runs[k]
            tr_num_k = int(tr_nums[k]) - 1


            if run_j == 1:
                first_point_nifti_files.append(np_a_rs[tr_num_j])
            elif run_j == 2:
                first_point_nifti_files.append(np_b_rs[tr_num_j])

            if run_k == 1:
                second_point_nifti_files.append(np_a_rs[tr_num_k])
            elif run_k == 2:
                second_point_nifti_files.append(np_b_rs[tr_num_k])

            j += 2
            k += 2

        first_point_avg = np.mean(first_point_nifti_files, axis=0)
        second_point_avg = np.mean(second_point_nifti_files, axis=0)
        concat_tr[stim] = np.concatenate((first_point_avg, second_point_avg), axis=1)

    np.savez_compressed(f"G:\comlam\\fsl\\sentiment\P{participant}.npz", concat_tr)



def store_avg_tr(participant, nifti_patha, nifti_pathb):
    # Don't use this function.

    # participant = 1003
    # nifti_patha = "smb://localhost/Google Drive/My Drive/CoMLaM_rohan/CoMLaM/Preprocessed/Reg_to_Std_and_Str/P_0999/Synonym_RunA_12.feat/filtered_func_data.nii"
    # nifti_pathb = "smb://localhost/Google Drive/My Drive/CoMLaM_rohan/CoMLaM/Preprocessed/Reg_to_Std_and_Str/P_0999/Synonym_RunB_13.feat/filtered_func_data.nii"
    # Now avg the TRs for each stimuli and for each corresponding 'run' and for each stimuli.
    tr_meta_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\" + str(participant) + "_TRsToUse.xlsx"
    metadata = pd.read_excel(tr_meta_path)  # Remove this in the final version.


    stims =  metadata.iloc[:,[0,2,7]].values
    # Keep only stims with two words.
    stims_two_words = [val for val in stims if val[0].count(' ')==1]

    #---------------------------------------------------------------
    # Now retrieve from the nifti file.
    np_a = img.get_data(nifti_patha)
    np_b = img.get_data(nifti_pathb)

    # Now preprocess the nifti and store it later in a directory called 'avg_tr_nifti' on GDrive.
    np_a = np.transpose(np_a, (3, 0, 1, 2))
    np_a_rs = np_a.reshape(np_a.shape[0], -1)
    mean_voxels_a = np.mean(np_a_rs, axis=0)

    np_b = np.transpose(np_b, (3, 0, 1, 2))
    np_b_rs = np_b.reshape(np_b.shape[0], -1)
    mean_voxels_b = np.mean(np_b_rs, axis=0)

    # Removing the mean voxel value.
    for i, row in enumerate(np_a_rs):
        np_a_rs[i] = np_a_rs[i] - mean_voxels_a

    for i, row in enumerate(np_b_rs):
        np_b_rs[i] = np_b_rs[i] - mean_voxels_b


    # Now map the nifti indexes in the TRsToUse excel file to the fmri data.
    run1_nifti = {}
    run2_nifti = {}
    # Could be optimized but it's okay for now.
    for stim in stims_two_words:
        run1_nifti[stim[0]] = []
        run2_nifti[stim[0]] = []

    # Store the corresponding fMRI files using the TR numbers from the excel sheet.
    # For run1 -> from SynonymA folder, for run2 -> from SynonymB folder.
    for stim in stims_two_words:
        if stim[1] == 1:
            run1_nifti[stim[0]].append(np_a_rs[stim[2]])
        else:
            run2_nifti[stim[0]].append(np_b_rs[stim[2]])

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
                tr_run1 = np.reshape(run1_nifti_avg[key], (-1, run1_nifti_avg[key].shape[0]))
                tr_run2 = np.reshape(run2_nifti_avg[key], (-1, run2_nifti_avg[key].shape[0]))
                concat_TR[key] = np.concatenate((tr_run1, tr_run2), axis=1)


    # Now store the values.
    np.savez_compressed(f'E:\My Drive\CoMLaM_rohan\CoMLaM\\avg_trs_concat\\P_{participant}_concat.npz', concat_TR)

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


def load_nifti_and_w2v(participant, avg_w2v=False, mean_removed=False):
    """

    :param participant: The particpant for which the fMRI data needs to be loaded. Takes an integer.
    :return: the nifti file for the participant and the corresponding condition.
    """
    system = platform.system()
    if system == 'Windows':
        # For local development.
        path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\spm\\sentiment"
        if avg_w2v == False:
            w2v_path = "G:\comlam\embeds\\two_words_stim_w2v_concat_dict.npz"
        else:
            w2v_path = "G:\comlam\embeds\\two_words_stim_w2v_avg_dict.npz"
    elif system == 'Linux':
        # For Compute Canada development.
        path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/"
        if avg_w2v == False:
            w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/two_words_stim_w2v_concat_dict.npz"
        else:
            w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/two_words_stim_w2v_avg_dict.npz"

    if mean_removed == True:
        nifti_path = path + f"P{participant}_mean_removed.npz"
    else:
        nifti_path = path + f"P{participant}.npz"
    nifti_data = np.load(nifti_path, allow_pickle=True)['arr_0'].tolist()


    w2v_data = np.load(w2v_path, allow_pickle=True)['arr_0'].tolist()

    # Now map the nifti data to the corresponding concatenated w2v vectors.
    x_data = []
    y_data = []

    for stim, fmri in nifti_data.items():
        x_data.append(fmri.tolist())
        y_data.append(w2v_data[stim])

    x_temp = np.array(x_data)
    y_temp = np.array(y_data)

    x = np.reshape(x_temp, (x_temp.shape[0], x_temp.shape[2]))

    # Also loading the stimuli phrases.
    stims = []
    for stim in nifti_data.keys():
        stims.append(stim)

    return x, y_temp, stims

def two_vs_two(preds, ytest):
    total_points = 0
    points = 0
    print(type(preds))
    print(type(ytest))
    j = 0
    for pred, y_true in zip(preds, ytest):

        s_i_pred = pred[0].tolist()
        s_j_pred = pred[1].tolist()
        s_i = y_true[0].tolist()
        s_j = y_true[1].tolist()


        # print("s_i_pred", s_i_pred)
        # print("s_j_pred", s_j_pred)
        # print('s_i ', s_i)
        # print('s_j ', s_j)


        dsii = cosine(s_i, s_i_pred)
        dsjj = cosine(s_j, s_j_pred)
        dsij = cosine(s_i, s_j_pred)
        dsji = cosine(s_j, s_i_pred)

        if dsii + dsjj <= dsij + dsji:
            points += 1
        total_points += 1

    return points * 1.0 / total_points  # Multiplying by 1.0 for floating point conversion.

def extended_2v2(y_test, preds):
    """
    There are two additions to this function over the previous two_vs_two test.
    1. The grid figures will be symmetric now.
    2. Each pair of words is compared only once.
    """
    points = 0
    total_points = 0
    n_words = 12

    for i in range(preds.shape[0] - 1):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(i + 1, preds.shape[0]):
            temp_score = 0
            s_j = y_test[j]
            s_j_pred = preds[j]

            dsii = cosine(s_i, s_i_pred)
            dsjj = cosine(s_j, s_j_pred)
            dsij = cosine(s_i, s_j_pred)
            dsji = cosine(s_j, s_i_pred)

            if dsii + dsjj <= dsij + dsji:
                points += 1
                temp_score = 1  # If the 2v2 test does not pass then temp_score = 0
            total_points += 1


def leave_two_out(stims):
    """
    Return the indices for all leave-two-out cv.
    :param stims: the stimuli strings
    :return: the training and test sets.
    """

    # Find out all the pairs.
    all_test_pairs = []
    all_train_pairs = []
    for i in range(len(stims) - 1):
        for j in range(i + 1, len(stims)):
            test_pair = [i, j]
            all_test_pairs.append(test_pair)
            train_indices_temp = np.arange(len(stims)).tolist()
            train_pairs = list_diff(train_indices_temp, test_pair)
            all_train_pairs.append(train_pairs)

    return all_train_pairs, all_test_pairs

def list_diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

def load_stim_vectors(participant):
    pass