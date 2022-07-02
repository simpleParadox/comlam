import pandas as pd
import numpy as np
from scipy.io import savemat
import os
import ast
import glob

# participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
# participants = [1005, 1014, 1030, 1033]
participants = [1032]

for participant in participants:
    # Windows path.
    # trs_to_use = pd.read_excel(f"E:\Shared drives\Varshini_Brea\CoMLaM\\Preprocessed\\SPM\\P{participant}\\sentiment\\multCondnsP{participant}.xlsx")
    # participant = 1032
    # Mac path
    t = '2k'
    save_path = glob.glob(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_*/")
    f = glob.glob(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_*/multCondnsOnsetsJoinedP{participant}_2k.xlsx")
    # trs_to_use = pd.read_excel(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_*/multCondnsOnsetsJoinedP{participant}_2.xlsx")

    trs_to_use = pd.read_excel(f[0])

    mdict = {}

    names = []
    onsets = []
    durations = []

    case_onsets = {}



    for row in trs_to_use.iterrows():
        if t=='2k':
            names.append(row[1]['combinedStim'])
        else:
            names.append(row[1]['stimulus'])
        # onsets.append(ast.literal_eval(row[1]['ConcatOnset']))

        onset_list = ast.literal_eval(row[1]['ConcatOnset'])

        for j in range(4, len(onset_list)+1):
            if j not in case_onsets.keys():
                case_onsets[j] = [np.array(onset_list[:j], dtype=float).tolist()]
            else:
                case_onsets[j].append(np.array(onset_list[:j], dtype=float).tolist())

        durations.append(0.0)

    #
    for j in range(4, len(onset_list)+1):
        mdict['names'] = names
        mdict['onsets'] = case_onsets[j]
        mdict['durations'] = durations
        savemat(save_path[0] + f"p{participant}_multiCondns_ConcatOnset_{j}_runs.mat", mdict)


    # names = sorted(set(trs_to_use["words"].values.tolist()))
    #
    # mdict1 = {}
    # # mdict2 = {}
    # mdict1['names'] = names
    # # mdict2['names'] = names
    #
    #
    # trs_group = trs_to_use.groupby(["words", "run"], as_index=True, group_keys=True)
    #
    # run_1_onsets = []
    # # run_2_onsets = []
    # run_1_durations = []
    # # run_2_durations = []
    # for name in names:
    #     run_1_group_df = trs_to_use[(trs_to_use.words == name)]# & (trs_to_use.run == 1)]
    #     # run_2_group_df = trs_to_use[(trs_to_use.words == name) & (trs_to_use.run == 2)]
    #
    #     stim_onsets_run_1 = run_1_group_df["ConcatOnset"].values.tolist()
    #     temp = np.array(stim_onsets_run_1, dtype=np.float32).tolist()
    #     # stim_onsets_run_2 = sorted(set(run_2_group_df["Zeroed_StimOnset"].values.tolist()))
    #
    #     run_1_onsets.append(temp)
    #     # run_2_onsets.append(stim_onsets_run_2)
    #
    #     # run_1_durations.append(np.zeros(len(stim_onsets_run_1)).tolist())
    #     # run_2_durations.append(np.zeros(len(stim_onsets_run_2)).tolist())
    #     run_1_durations.append(1.97)
    #
    #
    # mdict1['onsets'] = run_1_onsets
    # # mdict2['onsets'] = run_2_onsets
    #
    # mdict1['durations'] = run_1_durations
    # # mdict2['durations'] = run_2_durations
    #
    # # if not os.path.exists(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\matlab_scripts"):
    # #     os.makedirs(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\matlab_scripts")
    #
    #
    # # savemat(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\matlab_scripts\\p{participant}_multi_conds_fmri_spec_run_concat_1.97_durns.mat", mdict1)
    # savemat(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea/CoMLaM/Preprocessed/SPM/P{participant}/matlab_scripts/p{participant}_multi_conds_fmri_spec_run_concat_1.97_durns.mat", mdict1)
    # # savemat(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\matlab_scripts\\p{participant}_multi_conds_fmri_spec_run_2.mat", mdict2)