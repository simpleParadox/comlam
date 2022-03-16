import pandas as pd
import numpy as np
from scipy.io import savemat
import os

# participants = [1003, 1004, 1006, 1007, 1008, 1010, 1012, 1013, 1016, 1017, 1019]
participants = [1024]

for participant in participants:
    trs_to_use = pd.read_excel(f"E:\Shared drives\Varshini_Brea\CoMLaM\\Preprocessed\\SPM\\P{participant}\\sentiment\\multCondnsP{participant}.xlsx")

    names = sorted(set(trs_to_use["words"].values.tolist()))

    mdict1 = {}
    # mdict2 = {}
    mdict1['names'] = names
    # mdict2['names'] = names


    trs_group = trs_to_use.groupby(["words", "run"], as_index=True, group_keys=True)

    run_1_onsets = []
    # run_2_onsets = []
    run_1_durations = []
    # run_2_durations = []
    for name in names:
        run_1_group_df = trs_to_use[(trs_to_use.words == name)]# & (trs_to_use.run == 1)]
        # run_2_group_df = trs_to_use[(trs_to_use.words == name) & (trs_to_use.run == 2)]

        stim_onsets_run_1 = run_1_group_df["ConcatOnset"].values.tolist()
        temp = np.array(stim_onsets_run_1, dtype=np.float32).tolist()
        # stim_onsets_run_2 = sorted(set(run_2_group_df["Zeroed_StimOnset"].values.tolist()))

        run_1_onsets.append(temp)
        # run_2_onsets.append(stim_onsets_run_2)

        # run_1_durations.append(np.zeros(len(stim_onsets_run_1)).tolist())
        # run_2_durations.append(np.zeros(len(stim_onsets_run_2)).tolist())
        run_1_durations.append(0.0)


    mdict1['onsets'] = run_1_onsets
    # mdict2['onsets'] = run_2_onsets

    mdict1['durations'] = run_1_durations
    # mdict2['durations'] = run_2_durations

    if not os.path.exists(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\matlab_scripts"):
        os.makedirs(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\matlab_scripts")


    savemat(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\matlab_scripts\\p{participant}_multi_conds_fmri_spec_run_concat.mat", mdict1)
    # savemat(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\matlab_scripts\\p{participant}_multi_conds_fmri_spec_run_2.mat", mdict2)