import numpy as np
from nilearn.image import get_data
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# Define function that reads in a nifti file and stores it as a numpy array.
def beta_to_npz(participant, runs=10, type='2k', brain_type='wholeBrain'):
    """
    The function takes in all the beta files and stores a dictionary where the key is the stim and the value is the gray matter beta file (as a numpy ndarray). Nan values are dropped.
    :param participant: Integer. The participant for which betas to numpy processing needs to be done. e.g. 1004
    :return: None but stores the file to a directory.
    """
    # Define paths
    # Mac path
    # Mac path -> Don't forget to change user.
    brain_type = brain_type
    # local_participant_path = f"/Users/simpleparadox/Documents/comlam_raw/P{participant}/study/roi_beta_1.97_durns/"
    local_participant_study_path = f"/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/{brain_type}/"

    run_suffix = str(runs).zfill(2)

    remote_participant_path = f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_{type}/sentiment/Titration/{brain_type[0].upper() + brain_type[1:]}/1stLevel_concat{brain_type[0].upper() + brain_type[1:]}_{run_suffix}Runs/"
    # remote_participant_study_path = f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_{type}/sentiment/"
    s = 'mb'
    if type == '2k': s = ''
    multCondnsFile = pd.read_excel(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_{type}/multCondnsOnsetsJoinedP{participant}_2{s}.xlsx")

    # Windows path
    # remote_participant_path = f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\sentiment\\betas_concat_RPfile_roiMask_MANUAL_nonZeroDurns/"
    # remote_participant_study_path = f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\sentiment\\"
    # multCondnsFile = pd.read_excel(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\sentiment\\multCondnsP{participant}.xlsx")

    beta_file_numbers = multCondnsFile['Beta']
    beta_file_numbers = beta_file_numbers.dropna().values
    beta_numbers_list = beta_file_numbers.tolist()

    # sorted_stims = sorted(set(multCondnsFile.iloc[:,0].values.tolist()))
    sorted_stims = multCondnsFile.iloc[:, 0].values.tolist()

    # Now iterate over beta_numbers and read the corresponding files.
    stim_index = 0
    beta_dict = {}
    for beta in beta_numbers_list:
        file_number = str(int(beta)).zfill(4)
        print(file_number)
        f = glob.glob(remote_participant_path + f"*{file_number}.nii")

        # Now get the data and convert to numpy.
        print(f)
        nifti = np.array(get_data(f[0]))
        nifti = nifti[~np.isnan(nifti)]
        beta_dict[sorted_stims[stim_index]] = nifti
        stim_index += 1


    # Now save the beta_dict as an .npz array.
    # The result will be later stored on compute canada on which the decoding will be done.
    np.savez_compressed(local_participant_study_path + f"P{participant}_{brain_type}_beta_dict_{run_suffix}runs.npz", beta_dict)



def raw_to_npz(participant, type='2k', avg_tr=False):

    # Use the TRsToUse files.
    #NOTE: This function is broken -> Fix this. It doesn't considere the right TRs for averaging or concatenating.

    # participant = 1
    stims = ['absolute_mess', 'accidentally_open', 'augmented_reality', 'back_injury', 'bad_dreams', 'best_shit', 'bloody_hilarious', 'busy_working', 'cancer_free', 'completely_unusable', 'constant_pain', 'coughing_fit', 'crazy_talented', 'dead_gorgeous', 'drama_free', 'emotional_fair', 'even_sadder', 'extra_shot', 'freaking_cool', 'friend_died', 'good_smell', 'grand_theft', 'great_sadness', 'guilty_pleasures', 'happiness_overload', 'happy_accident', 'heart_sinks', 'horrible_daughter', 'job_losses', 'kind_words', 'latest_issue', 'laughing_hard', 'leave_feedback', 'like_hell', 'long_friendship', 'long_journey', 'losing_faith', 'lost_art', 'love_sucks', 'lovely_shot', 'major_headache', 'nose_job', 'poor_effort', 'pretty_amazing', 'pretty_grim', 'proven_guilty', 'real_disaster', 'seriously_great', 'sexy_beast', 'small_achievement', 'smart_creature', 'squeaky_clean', 'stroll_down', 'super_pissed', 'truly_gone', 'unbelievably_happy', 'wicked_excited', 'wonderful_break', 'worst_luck', 'year_gone']
    trs_to_use = pd.read_excel(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_{type}/sentiment/TRsToUse_P{participant}_2.xlsx")

    trs_groups = trs_to_use.groupby(by=['RunNum'])

    path_to_raw_trs = f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_{type}/sentiment/rawRealigned/"

    files = [os.path.basename(f) for f in glob.glob(path_to_raw_trs + "rfCoMLaM*.nii")]
    files = sorted(files)


    runs_from_fnames = []
    r = re.compile("-00\d{2}-")

    for f in files:
        runs_from_fnames.append(int(r.search(f).group(0)[-3:-1]))

    runs_from_fnames = set(runs_from_fnames)


    # Mapping the files and the modified run numbers.
    mod_runs = runs_from_fnames
    runs = [i for i in range(1,len(mod_runs))]
    run_mapping = dict(zip(mod_runs, runs))

    files_runs_dict = {}
    for i in runs:
        files_runs_dict[i] = []


    # Grouping the files based on their runs.
    for key, value in run_mapping.items():
        for f in files:
            if f"-00{key}-" in f:
                files_runs_dict[run_mapping[key]].append(f)

    tr_stims = []
    tr_files = []


    for row in trs_groups:
        data = row[1]
        run = row[0]
        print(run)

        for stim, tr_nums in zip(data['combinedStim'], data['TRsToUse']):
            tr_stims.append(stim)
            all_stim_trs = []
            for tr_num in tr_nums:
                file_number = str(tr_num).zfill(6)
                dict_files = files_runs_dict[run]
                tr_file_name = list(set([s for i, s in enumerate(dict_files) if file_number in s]))[0]
                nifti = get_data(path_to_raw_trs + tr_file_name)
                all_stim_trs.append(nifti)
            if avg_tr:
                niftis = np.mean(all_stim_trs, axis=0)
            else:
                niftis = np.concatenate(all_stim_trs, axis=0)

            tr_files.append(niftis)

    stim_and_trs = dict(zip(tr_stims, tr_files))

    np.savez_compressed(f"/Users/simpleparadox/Desktop/Projects/comlam/data/spm/sentiment/raw/unmasked/{participant}_raw.npz", stim_and_trs)


def store_sentiment(participant, runs=10, type='2k', congruent=False):
    participant = 1014
    type='2k'
    trs_to_use = pd.read_excel(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_{type}/sentiment/TRsToUse_AlphaOrder_P{participant}_2.xlsx")

    sentiment_dict = {}
    sentiment_dict['stims'] = {}
    trs_groups = trs_to_use.groupby(by=['combinedStim'])
    for idx, row in enumerate(trs_groups):
        stim = row[0]
        sent = row[1].iloc[0]['Polarity'][-3:]

        if congruent:
            if sent != 'eut':
                sentiment_dict[stim] = sent
                sentiment_dict['stims'][idx] = stim



    np.savez_compressed(f"embeds/congruency.npz", sentiment_dict)

    # Now store the beta numbers.

# runs = [4, 5, 6, 7, 8, 9, 10]
runs = [4,5,6,7,8]
# runs = [6]
for run in runs:
    beta_to_npz(1030, runs=run, type='2k', brain_type='priceNine')
