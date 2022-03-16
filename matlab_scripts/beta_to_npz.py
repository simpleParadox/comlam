import numpy as np
from nilearn.image import get_data
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Define function that reads in a nifti file and stores it as a numpy array.
def beta_to_npz(participant):
    """
    The function takes in all the beta files and stores a dictionary where the key is the stim and the value is the gray matter beta file (as a numpy ndarray). Nan values are dropped.
    :param participant: Integer. The participant for which betas to numpy processing needs to be done. e.g. 1004
    :return: None but stores the file to a directory.
    """
    # Define paths
    # Mac path
    # Mac path -> Don't forget to change user.
    # local_participant_path = f"/Users/simpleparadox/Documents/comlam_raw/P{participant}/study/1st_Level/"
    # local_participant_study_path = f"/Users/simpleparadox/Documents/comlam_raw/P{participant}/study/"

    # remote_participant_path = f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea/CoMLaM/Preprocessed/SPM/P{participant}/sentiment/betas_concat_RPfile_roi/"
    # remote_participant_study_path = f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea/CoMLaM/Preprocessed/SPM/P{participant}/sentiment/"
    # multCondnsFile = pd.read_excel(f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea/CoMLaM/Preprocessed/SPM/P{participant}/sentiment/multCondnsP{participant}.xlsx")

    # Windows path
    remote_participant_path = f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\sentiment\\betas_concat_RPfile_roiMask_MANUAL_nonZeroDurns/"
    remote_participant_study_path = f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\sentiment\\"
    multCondnsFile = pd.read_excel(f"E:\Shared drives\Varshini_Brea\CoMLaM\Preprocessed\SPM\P{participant}\sentiment\\multCondnsP{participant}.xlsx")

    beta_file_numbers = multCondnsFile['Beta']
    beta_file_numbers = beta_file_numbers.dropna().values
    beta_numbers_list = beta_file_numbers.tolist()

    sorted_stims = sorted(set(multCondnsFile.iloc[:,0].values.tolist()))

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
    np.savez_compressed(remote_participant_study_path + f"P{participant}_roi_beta_dict_manual_nonzerodurns.npz", beta_dict)


beta_to_npz(1024)