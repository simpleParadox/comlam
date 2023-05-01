import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nilearn import image as img
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import platform
import glob
import scipy.stats as stats

from sklearn.model_selection import LeaveOneOut
import seaborn as sns
import matplotlib.pyplot as plt



def store_stim_sentiment():
    stim_ratings_sheet = pd.read_csv('data/SM_s_SCL-OPP___Our_Stim_List.xlsx - Master.csv')

    # Get stimuli phrases.
    stim_array = np.load('embeds/two_words_stim_w2v_avg_dict.npz', allow_pickle=True)['arr_0'].tolist()
    stims = [key for key in stim_array.keys()]

    sentiment_scores = []

    # temp = stim_ratings_sheet[stim_ratings_sheet['Term'].isin(stims)]
    df_stims = []
    for stim in stims:
        try:
            df_stims.append(stim_ratings_sheet[stim_ratings_sheet['Term']==stim].Term.values.tolist()[0])
            sentiment_scores.append(stim_ratings_sheet[stim_ratings_sheet['Term']==stim].iloc[:,1].values.tolist()[0])
        except Exception as e:
            print(f"Problem with stimulus: {stim}", e)

    # Adding 'a nose job' instead of 'nose job' because I think they are the same.
    df_stims.append('nose job')
    sentiment_scores.append(stim_ratings_sheet[stim_ratings_sheet['Term']=='a nose job'].iloc[:,1].values.tolist()[0])

    # Make a dictionary from two lists using 'zip'.
    stim_ratings_dict = dict(zip(df_stims, sentiment_scores))

    np.savez_compressed('embeds/sentiment_ratings.npz', stim_ratings_dict)



def store_stim_sentiment(congruent=False):
    pass

def store_betas_spm(participant, task='sentiment', mask_type='gm'):
    """

    :param participant: The participant for which preprocessing needs to be done.
    :param task: 'sentiment' or 'synonym', default is 'sentiment'
    :param mask_type: Type of mask. 'gm' for gray matter. 'roi' for region of interest.
    :return: None but stores the processed .npz file to disk.
    """

    spm_path = "E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\\"
    participant_path = spm_path + "P" + str(participant) + "\\" + task + "\\"
    mask_path = participant_path + f"betas_concat_RPfile_{mask_type}Mask\\"
    conditions_mapping = participant_path + f"multCondnsP{participant}.xlsx"

    # Read the conditions mapping file.
    conditions_files = pd.read_excel(conditions_mapping)

    conditions_map_grouped = conditions_files.groupby(["Beta"])

    stims_dict = {}
    for row in conditions_map_grouped:
        stims_dict[int(row[0])] = row[1]['words'].values.tolist()[0]


    # beta_maps_to_nifti = conditions_files['Beta'].dropna().values
    nifti_dict = {}
    for file_number in stims_dict.keys():
        padded_file_number = str(file_number).zfill(4)
        file = glob.glob(mask_path + f"*{padded_file_number}.nii")
        f = img.get_data(file)
        beta = np.array(f)
        beta_transposed = np.transpose(beta, (3, 0, 1, 2))
        beta_reshaped = beta_transposed.reshape(beta_transposed.shape[0], -1)
        # Now drop nan.
        beta_no_nan = beta_reshaped[~np.isnan(beta_reshaped)]
        nifti_dict[stims_dict[file_number]] = beta_no_nan

    np.savez_compressed(f"G:\comlam\data\spm\sentiment\masked\\beta_weights\\beta_{mask_type}Mask\\P{participant}.npz", nifti_dict)



def store_masked_trs_spm(participant, task, remove_mean=False, avg_tr=False):
    """

    :param participant: The number of participant. Accepts a single integer. E.g. 1003.
    :param task: string, sentiment or synonym.
    :param remove_mean: whether to subtract the mean fMRI data from each functional image or not.
    :param avg_tr: If true - average the TRs, else - concatenate.
    :return: None, but store the result in a directory.
    """
    # participant = 1012  # Remove this when not testing.
    # task = 'sentiment'

    # Define the paths for GDrive.
    spm_path = "E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\\"
    participant_path = spm_path + "P" + str(participant) + "\\" + task + "\\"
    tr_meta_path = "E:\Shared drives\Varshini_Brea\CoMLaM\\" + str(participant) + "_TRsToUse.xlsx"
    metadata = pd.read_excel(tr_meta_path)
    run1_path = participant_path + "masked_w_run01\\"
    run2_path = participant_path + "masked_w_run02\\"

    # Select only the stimuli that has two words only.
    ## Find the rows where the stim is more than two words.
    non_two_word_phrases_idx = [i for i in range(len(metadata)) if metadata.iloc[i, 0].count(' ') > 1]
    two_stim_metadata = metadata.drop(labels=non_two_word_phrases_idx, axis=0)

    # First group the TR_metadata by stimuli.
    grouped_metadata = two_stim_metadata.groupby(['words'])

    # Load all the fMRI data to remove the mean voxels from each of the stored fMRI file.
    nifti_run_1_all_files = []
    nifti_run_2_all_files = []
    if remove_mean == True:
        for file in glob.glob(run1_path + "*.nii.npz"):
            print("removing mean")
            print(file)
            nifti = np.load(file, allow_pickle=True)['arr_0']
            nifti_run_1_all_files.append(nifti)

        for file in glob.glob(run2_path + "*.nii.npz"):
            print("removing mean")
            print(file)
            nifti = np.load(file, allow_pickle=True)['arr_0']
            nifti_run_2_all_files.append(nifti)

        mean_voxels_a = np.mean(nifti_run_1_all_files, axis=0)
        mean_voxels_b = np.mean(nifti_run_2_all_files, axis=0)

    result_tr = {}
    counter = 0
    for row in grouped_metadata:
        print(counter)
        counter += 1
        # 'row' is a tuple.
        stim = row[0]
        df = row[1]  # Second index is the dataframe for the stimulus.
        runs = df['run'].values
        nifti_idx = df['Nifti'].values

        # Now do the retrieval of fMRI data here and averaging and concatenating.
        j = 0
        k = 1
        first_point_nifti_files = []
        second_point_nifti_files = []
        while (j < 7 and k < 8):
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
                try:
                    file_name = glob.glob(run1_path + f"*{left_j_1}-{left_j_2}-*.nii.npz")[0]
                except:
                    file_name = glob.glob(run1_path + f"*{left_j_1}-{left_j_2}-*.nii.npz")
                # print("Run j=1 ", file_name)
                # if type(file_name) == list:
                #     file_name = file_name[0]
                np_a_rs_j_1 = np.load(file_name, allow_pickle=True)['arr_0']
                # Now store the file in an array.
                # np_a_j_1 = np.transpose(data_j_1, (3, 0, 1, 2))
                # np_a_rs_j_1 = np_a_j_1.reshape(np_a_j_1.shape[0], -1)
                if remove_mean == True:
                    np_a_rs_j_1 = np_a_rs_j_1 - mean_voxels_a
                first_point_nifti_files.append(np_a_rs_j_1)
            elif run_j == 2:
                # Read from the 'run02; folder
                try:
                    file_name = glob.glob(run2_path + f"*{left_j_1}-{left_j_2}-*.nii.npz")[0]
                except:
                    file_name = glob.glob(run2_path + f"*{left_j_1}-{left_j_2}-*.nii.npz")
                # print("Run j=2 ", file_name)
                # if type(file_name) == list:
                #     file_name = file_name[0]
                np_a_rs_j_2 = np.load(file_name, allow_pickle=True)['arr_0']
                # Now store the file in an array for later averaging.
                # np_a_j_2 = np.transpose(data_j_2, (3, 0, 1, 2))
                # np_a_rs_j_2 = np_a_j_2.reshape(np_a_j_2.shape[0], -1)
                if remove_mean == True:
                    np_a_rs_j_2 = np_a_rs_j_2 - mean_voxels_b
                first_point_nifti_files.append(np_a_rs_j_2)

            if run_k == 1:
                try:
                    file_name = glob.glob(run1_path + f"*{left_k_1}-{left_k_2}-*.nii.npz")[0]
                except:
                    file_name = glob.glob(run1_path + f"*{left_k_1}-{left_k_2}-*.nii.npz")
                # if type(file_name) == list:
                #     file_name = file_name[0]
                # print("Run k=1 ", file_name)
                np_a_rs_k_1 = np.load(file_name, allow_pickle=True)['arr_0']
                # Now store the file in an array.
                # np_b_k_1 = np.transpose(data_k_1, (3, 0, 1, 2))
                # np_a_rs_k_1 = np_b_k_1.reshape(np_b_k_1.shape[0], -1)
                if remove_mean == True:
                    np_a_rs_k_1 = np_a_rs_k_1 - mean_voxels_a
                second_point_nifti_files.append(np_a_rs_k_1)

            elif run_k == 2:
                try:
                    file_name = glob.glob(run2_path + f"*{left_k_1}-{left_k_2}-*.nii.npz")[0]
                except:
                    file_name = glob.glob(run2_path + f"*{left_k_1}-{left_k_2}-*.nii.npz")
                # print("Run k=2 ", file_name)
                # if type(file_name) == list:
                #     file_name = file_name[0]
                np_a_rs_k_2 = np.load(file_name, allow_pickle=True)['arr_0']
                # Now store the file in an array.
                # np_a_k_2 = np.transpose(data_k_2, (3, 0, 1, 2))
                # np_a_rs_k_2 = np_a_k_2.reshape(np_a_k_2.shape[0], -1)
                if remove_mean == True:
                    np_a_rs_k_2 = np_a_rs_k_2 - mean_voxels_b
                second_point_nifti_files.append(np_a_rs_k_2)

            j += 2
            k += 2

        # Average and concatenate.
        first_point_avg = np.mean(first_point_nifti_files, axis=0)
        second_point_avg = np.mean(second_point_nifti_files, axis=0)
        if avg_tr:
            result_tr[stim] = np.mean((first_point_avg, second_point_avg), axis=0)
        else:
            result_tr[stim] = np.concatenate((first_point_avg, second_point_avg))

    print(f'Participant {participant}: saved.')
    if remove_mean == True:
        if avg_tr:
            print("Average TRs mean removed.")
            np.savez_compressed(f"data/spm/sentiment/masked/rf/avg_trs\\P{participant}_avg_mean_removed.npz", result_tr)
        else:
            print("Average TRs non mean removed.")
            np.savez_compressed(f"data/spm/sentiment/masked/rf/concat_trs\\P{participant}_mean_removed.npz", result_tr)
    else:
        if avg_tr:
            print("Average TRs non mean removed.")
            np.savez_compressed(f"/data/spm\sentiment\\masked\\wrf\\avg_trs\\P{participant}_avg.npz", result_tr)
        else:
            print("Concat TRs non mean removed.")
            np.savez_compressed(f"/data/spm\sentiment\\masked\\wrf\\concat_trs\\P{participant}.npz", result_tr)


def store_trs_spm(participant, task, remove_mean=False, avg_tr=False):

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


    result_tr = {}
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
        if avg_tr:
            result_tr[stim] = np.mean((first_point_avg, second_point_avg), axis=0)
        else:
            result_tr[stim] = np.concatenate((first_point_avg, second_point_avg), axis=1)
    print(f'Participant {participant}: saved.')
    if remove_mean == True:
        np.savez_compressed(f"/data/spm\sentiment\\avg_trs\\P{participant}_avg_mean_removed.npz", result_tr)
    else:
        np.savez_compressed(f"/data/spm\sentiment\\avg_trs\\P{participant}_avg.npz", result_tr)


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
    # tr_meta_path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\" + str(participant) + "_TRsToUse.xlsx"
    metadata = pd.read_excel(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea/CoMLaM/Preprocessed/SPM/P{participant}_2k/multCondnsOnsetsJoinedP{participant}_2.xlsx")
    # metadata = pd.read_excel(tr_meta_path)  # Remove this in the final version.

    stims = metadata.iloc[:, 0].values
    # Keep only stims with two words.
    stims_two_words = [val for val in stims if val.count('_')==1]
    stims_two_words = list(set(stims_two_words))

    return stims_two_words


def load_nifti_and_w2v(participant, avg_w2v=False, mean_removed=False, load_avg_trs=False, masked=False, permuted=False, nifti_type='rf',
                       beta=True, beta_mask_type='gm', embedding_type='w2v', predict_sentiment=False, run=10, whole_brain=False, priceNine=False, motor=False):
    """
    :param participant: The particpant for which the fMRI data needs to be loaded. Takes an integer.
    :return: the nifti file for the participant and the corresponding condition.
    """
    system = platform.system()
    if system == 'Windows':
        # For local development.
        # path = "E:\My Drive\CoMLaM_rohan\CoMLaM\\spm\\sentiment\\"
        if masked:
            if load_avg_trs:
                path = f"data/spm/sentiment/masked/{nifti_type}/avg_trs\\"
            else:
                path = f"data/spm/sentiment/masked/{nifti_type}/concat_trs\\"
        else:
            if load_avg_trs:
                path = "/data/spm\sentiment\\avg_trs\\"
            else:
                path = "/data/spm\sentiment\\"
        if embedding_type == 'w2v':
            if avg_w2v == False:
                w2v_path = "G:\comlam\embeds\\two_words_stim_w2v_concat_dict.npz"
            else:
                w2v_path = "G:\comlam\embeds\\two_words_stim_w2v_avg_dict.npz"
        elif embedding_type == 'roberta':
            w2v_path = "G:\comlam\embeds\\roberta_two_word_pooler_output_vectors.npz"

    elif system == 'Linux':
        # For Compute Canada development.
        if masked:
            if beta:
                # Load beta weights.
                beta_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/masked/beta_weights/"
            if load_avg_trs:
                path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/masked/{nifti_type}/avg_trs/"
            else:
                path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/masked/{nifti_type}/concat_trs/"
        else:
            if load_avg_trs:
                path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/avg_trs/"
            else:
                path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/"
        if embedding_type == 'w2v':
            if avg_w2v == False:
                w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/two_words_stim_w2v_concat_dict.npz"
            else:
                w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/two_words_stim_w2v_avg_dict.npz"
        elif embedding_type == 'sixty_w2v':
                if avg_w2v == False:
                    w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/sixty_two_word_stims_concat.npz"
                else:
                    w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/sixty_two_word_stims_avg.npz"
        elif embedding_type == 'roberta':
            # Load roberta sentiment embeddings (pooler_output).
            w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/roberta_two_word_pooler_output_vectors.npz"
    elif system == 'Darwin':
        # For MacOS local development.
        # First set of conditions for retrieving the fMRI data.
        if masked:
            if beta:
                # Load beta weights.
                beta_path = "/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/masked/beta_weights/"
            if load_avg_trs:
                path = f"/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/{nifti_type}/avg_trs/"
            else:
                path = f"/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/masked/{nifti_type}/concat_trs/"
        else:
            if load_avg_trs:
                path = "/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/avg_trs/"
            else:
                path = "/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/"
        # Second set of conditions for retrieving the embeddings/ratings.
        if predict_sentiment:
            w2v_path = "embeds/sentiment_ratings.npz"
        else:
            if embedding_type == 'w2v':
                if avg_w2v == False:
                    w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/two_words_stim_w2v_concat_dict.npz"
                else:
                    w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/two_words_stim_w2v_avg_dict.npz"
            elif embedding_type == 'roberta':
                # Load roberta sentiment embeddings (pooler_output).
                w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/roberta_two_word_pooler_output_vectors.npz"
            elif embedding_type == 'sixty_w2v':
                if avg_w2v == False:
                    w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/sixty_two_word_stims_concat.npz"
                else:
                    w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/sixty_two_word_stims_avg.npz"
            elif embedding_type == 'sixty_roberta':
                w2v_path = '/Users/simpleparadox/Desktop/Projects/comlam/embeds/roberta_sixty_two_word_pooler_output_vectors.npz'

    if beta and not whole_brain and not motor:
        nifti_path = beta_path + f"beta_{beta_mask_type}Mask/P{participant}_{beta_mask_type}_beta_dict.npz"
        print(nifti_path)
    elif whole_brain:
        run_suffix = str(run).zfill(2)
        nifti_path = f"/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/wholeBrain/P{participant}_wholeBrain_beta_dict_{run_suffix}runs.npz"
    elif priceNine:
        run_suffix = str(run).zfill(2)
        nifti_path = f"/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/priceNine/P{participant}_priceNine_beta_dict_{run_suffix}runs.npz"
    elif motor:
        run_suffix = str(run).zfill(2)
        nifti_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/motor_ba12346/P{participant}_motor_beta_dict_{run_suffix}runs.npz"
    else:
        if mean_removed == True:
            if load_avg_trs:
                print("Load avg TRs and mean removed")
                nifti_path = path + f"P{participant}_avg_mean_removed.npz"
            else:
                print("Load concat TRs and mean removed")
                nifti_path = path + f"P{participant}_mean_removed.npz"
        else:
            if load_avg_trs:
                print("Load avg TRs and non mean removed")
                nifti_path = path + f"P{participant}_avg.npz"
            else:
                print("Load concat TRs and non mean removed")
                nifti_path = path + f"P{participant}.npz"

    print("Nifti Path:", nifti_path)

    nifti_data = np.load(nifti_path, allow_pickle=True)['arr_0'].tolist()
    w2v_data = np.load(w2v_path, allow_pickle=True)['arr_0'].tolist()

    # Now map the nifti data to the corresponding concatenated w2v vectors.
    x_data = []
    y_data = []

    stim_keys = [k for k in w2v_data.keys()]

    for stim, fmri in nifti_data.items():
        if "_" in stim and 'w2v' not in embedding_type:
            # For stims if they have an underscore.
            stim = ' '.join(stim.split('_'))

        if '_' in stim_keys[0]:
            stim = '_'.join(stim.split(' '))
        x_data.append(fmri.tolist())
        y_data.append(w2v_data[stim])

    x_temp = np.array(x_data)
    y_temp = np.array(y_data)
    if predict_sentiment:
        y_temp = np.reshape(y_temp, (y_temp.shape[0],-1))


    # The following line was for the unmasked data I think.
    # x = np.reshape(x_temp, (x_temp.shape[0], x_temp.shape[2]))

    if permuted:
        np.random.shuffle(y_temp)

    # Also loading the stimuli phrases.
    stims = []
    for stim in nifti_data.keys():
        stims.append(stim)

    return x_temp, y_temp, stims


def extended_euclidean_2v2(preds, y_test):
    total_points = 0
    points = 0

    for i in range(preds.shape[0] - 1):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(i + 1, preds.shape[0]):
            temp_score = 0
            s_j = y_test[j]
            s_j_pred = preds[j]

            dsii = euclidean(s_i, s_i_pred)
            dsjj = euclidean(s_j, s_j_pred)
            dsij = euclidean(s_i, s_j_pred)
            dsji = euclidean(s_j, s_i_pred)

            if dsii + dsjj <= dsij + dsji:
                points += 1
            total_points += 1

    return points * 1.0 / total_points



def two_vs_two(preds, ytest, store_cos_diff=False):
    total_points = 0
    points = 0
    # print(type(preds))
    # print(type(ytest))

    cosine_diffs = []

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
            if store_cos_diff:
                cosine_diffs.append(dsii + dsjj - dsij - dsji)
        total_points += 1

    return points * 1.0 / total_points, cosine_diffs  # Multiplying by 1.0 for floating point conversion.


def evaluate_non_zero_shot():
    """
    Method to evaluate the 
    """
    pass


def extended_2v2(y_test, preds, store_cos_diff=False):
    """
    Calculate accuracy for each possible pair.
    """
    points = 0
    total_points = 0
    n_words = 12
    cosine_diffs = []

    # if type(y_test) == list:
    #     y_test = np.array(y_test)
    #     preds = np.array(preds)

    for i in range(len(preds) - 1):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(i + 1, len(preds)):
            temp_score = 0
            s_j = y_test[j]
            s_j_pred = preds[j]

            dsii = cosine(s_i, s_i_pred)
            dsjj = cosine(s_j, s_j_pred)
            dsij = cosine(s_i, s_j_pred)
            dsji = cosine(s_j, s_i_pred)

            if dsii + dsjj <= dsij + dsji:
                points += 1
                if store_cos_diff:
                    cosine_diffs.append(dsii + dsjj - dsij - dsji)
            total_points += 1

    return points * 1.0 / total_points, cosine_diffs


def leave_two_out(stims):
    """
    Return the indices for all leave-two-out cv.
    :param stims: the stimuli strings
    :return: the training and test sets.
    """

    # Find out all the pairs.
    all_train_pairs = []
    all_test_pairs = []

    for i in range(len(stims) - 1):
        for j in range(i + 1, len(stims)):
            test_pair = [i, j]
            all_test_pairs.append(test_pair)
            train_indices_temp = np.arange(len(stims)).tolist()
            train_pairs = list_diff(train_indices_temp, test_pair)
            all_train_pairs.append(train_pairs)

    return all_train_pairs, all_test_pairs

def leave_one_out(stims):
    loo = LeaveOneOut()
    all_test_pairs = []
    all_train_pairs = []
    for train_index, test_index in loo.split(stims):
        all_train_pairs.append(train_index.tolist())
        all_test_pairs.append(test_index.tolist())

    return all_train_pairs, all_test_pairs

def list_diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

def get_dim_corr_for_numpy(ypred, ytest):
    ypred = np.squeeze(ypred)
    ytest = np.squeeze(ytest)
    dim_corrs = []
    for i in range(ypred.shape[1]):
        r, p_value = stats.pearsonr(ypred[:, i], ytest[:, i])
        dim_corrs.append(r)
    return dim_corrs

def get_dim_corr(ypred, ytest):
    """
    Calculate dimension wise correlation for the word vectors. This implementation is general. As long as the two tensors are matrices, it should work.
    :return: Correlation

    NOTE: ypred and ytest should have equal number of dimensions for dimension 1.
    """
    # assert ypred.shape[1] == ytest.shape[1]

    # np.corrcoef(ypred, ytest)
    if type(ypred) is not np.ndarray:
        ypred = np.array(ypred)
        ypred = ypred.reshape(ypred.shape[0], ypred.shape[2])
    if type(ytest) is not np.ndarray:
        ytest = np.array(ytest)
        ytest = ytest.reshape(ytest.shape[0], ytest.shape[2])
    dim_corrs = []
    for i in range(ypred.shape[1]):
        r, p_value = stats.pearsonr(ypred[:, i], ytest[:, i])
        dim_corrs.append(r)
    return dim_corrs

def get_dim_corr_encoding(ypred_list, ytest_list):
    pass
    

def get_violin_plot(participant, corr_values):
    """
    :param participant: the participant for which the violin plot is being generated.
    :param corr_values: corr_values for which
    :return:
    """
    fig = plt.figure()
    sns.violinplot(x=corr_values)
    fig.suptitle(f"{participant} corr_values")
    return fig

def load_nifti_by_run(participant, type='wholeBrain', run=4):

    system = platform.system()
    run_suffix = str(run).zfill(2)
    if system == 'Darwin':
            nifti_path = f"/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/{type}/P{participant}_{type}_beta_dict_{run_suffix}runs.npz"
    elif system == 'Linux':
        if type != 'motor':
            if isinstance(participant, str) and 'Pilot' in participant:
                nifti_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/{type}/{participant}_{type}_beta_dict_{run_suffix}runs.npz"
            else:
                nifti_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/{type}/P{participant}_{type}_beta_dict_{run_suffix}runs.npz"
        elif type == 'motor':
            if isinstance(participant, str) and 'Pilot' in participant:
                nifti_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/motor_ba12346/{participant}_{type}_beta_dict_{run_suffix}runs.npz"
            else:
                nifti_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/motor_ba12346/P{participant}_{type}_beta_dict_{run_suffix}runs.npz"

    print("Nifti path: ", nifti_path)
    nifti_data = np.load(nifti_path, allow_pickle=True)['arr_0'].tolist()
    return nifti_data
    





def load_nifti(participant, load_avg_trs=False, beta=False, beta_mask_type='roi', masked=False):
    system = platform.system()
    if system == 'Linux':
        #NOTE: This 'if' block maybe be broken and may not cover all the use cases.
        # For Compute Canada development.
        if masked:
            if beta:
                # Load beta weights.
                beta_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/masked/beta_weights/"
            if load_avg_trs:
                path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/masked/{nifti_type}/avg_trs/"
            else:
                path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/masked/{nifti_type}/concat_trs/"
        else:
            if load_avg_trs:
                path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/avg_trs/"
            else:
                path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/concat_trs/"

    elif system == 'Darwin':
        # For MacOS local development.
        # First set of conditions for retrieving the fMRI data.

        if beta:
            # Load beta weights.
            if masked:
                beta_path = "/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/masked/beta_weights/"
                nifti_path = beta_path + f"beta_{beta_mask_type}Mask/P{participant}_{beta_mask_type}_beta_dict.npz"
        else:
            # Raw data.
            if load_avg_trs:
                path = "/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/raw/unmasked/"
                nifti_path = path + f"P{participant}_avg.npz"
            else:
                path = "/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/raw/unmasked"
                nifti_path = path + f"P{participant}.npz"
        # Second set of conditions for retrieving the embeddings/ratings.

    nifti_data = np.load(nifti_path, allow_pickle=True)['arr_0'].tolist()
    return nifti_data




def load_y(participant='', embedding_type='w2v', avg_w2v=False, sentiment=False, congruent=False, way='3'):
    """
    :param participant: May be redundant but tells you to load the 'y' for the participant.
    :param embedding_type: 'w2v', 'sixty_w2v', or 'roberta'
    :param avg_w2v: Whether to use average w2v or concat w2v. Boolean, default: False.
    :param sentiment: Whether to predict the sentiment of the two-word stimuli.
    :param congruent: Whether to predict the congruency of the stimuli.
    :return:
    """
    system = platform.system()
    if system == 'Darwin':
        if sentiment:
            w2v_path = "embeds/all_sentiment.npz"
        elif congruent:
            w2v_path = f"embeds/all_congruency.npz"
        else:
            if embedding_type == 'w2v':
                if avg_w2v == False:
                    w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/two_words_stim_w2v_concat_dict.npz"
                else:
                    w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/two_words_stim_w2v_avg_dict.npz"
            elif embedding_type == 'roberta':
                # Load roberta sentiment embeddings (pooler_output).
                w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/roberta_two_word_pooler_output_vectors.npz"
            elif embedding_type == 'sixty_w2v':
                if avg_w2v == False:
                    w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/sixty_two_word_stims_concat.npz"
                else:
                    w2v_path = "/Users/simpleparadox/Desktop/Projects/comlam/embeds/sixty_two_word_stims_avg.npz"
            elif embedding_type == 'sixty_roberta':
                w2v_path = '/Users/simpleparadox/Desktop/Projects/comlam/embeds/roberta_sixty_two_word_pooler_output_vectors.npz'
    elif system == 'Linux':
        if sentiment:
            w2v_path = "embeds/all_sentiment.npz"
        elif congruent:
            if way == '3':
                w2v_path = f"embeds/all_congruency.npz"
            elif way == '2':
                w2v_path = f"embeds/all_congruency_2_way.npz"
        else:
            print("Load embeddings for decoding")
            if embedding_type == 'w2v':
                if avg_w2v == False:
                    w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/two_words_stim_w2v_concat_dict.npz"
                else:
                    w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/two_words_stim_w2v_avg_dict.npz"
            elif embedding_type == 'roberta':
                # Load roberta sentiment embeddings (pooler_output).
                w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/roberta_two_word_pooler_output_vectors.npz"
            elif embedding_type == 'sixty_w2v':
                if avg_w2v == False:
                    w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/sixty_two_word_stims_concat.npz"
                else:
                    w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/sixty_two_word_stims_avg.npz"
            elif embedding_type == 'sixty_roberta':
                w2v_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/roberta_sixty_two_word_pooler_output_vectors.npz'
            elif embedding_type == 'bertweet':
                w2v_path = "/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/bertweet_pooler_output.npz"
            elif embedding_type == 'twitter_w2v':
                if avg_w2v == False:
                    w2v_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/twitter_word2vec_sixty_concat.npz'
                else:
                    w2v_path = '/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/twitter_word2vec_sixty_average.npz'


    w2v_data = np.load(w2v_path, allow_pickle=True)['arr_0'].tolist()

    return w2v_data



def get_train_and_test_samples(participant, load_avg_trs, beta, beta_mask_type,
                               avg_w2v, roberta, sentiment, congruent,
                               train_size=6, total_samples=10, permuted=False):
    """
    :train_size: The number of samples in the train set. Default=6
    :total_samples: The number of samples in the dataset. Default=10
    Implement a resampling procedure so that the test set has all the samples averaged to get one sample and the remaining are in the training set.
    :return: Training and test sets.
    """

    # First we need to group the samples according to the stimuli.
    x = load_nifti(participant, load_avg_trs, beta=beta, beta_mask_type=beta_mask_type)
    y = load_y(avg_w2v=avg_w2v, roberta=roberta, sentiment=sentiment, congruent=congruent)





