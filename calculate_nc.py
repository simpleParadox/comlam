from cProfile import label
from noise_ceiling import compute_ncsnr, compute_nc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nilearn.image import get_data
import glob
from sklearn.model_selection import train_test_split


def load_betas_and_stims(participant, brain_type, exp_type='congruency', remove_neutral=False, pos_neg_only=False):
    """
    This function also does noise ceiling calculation.
    """
    # beta_path = f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_2k/sentiment/Results_betaPerTrial/{brain_type}/"
    brain_type = brain_type[0].upper() + brain_type[1:]
    beta_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/P{participant}/Results_betaPerTrial/{brain_type}/"
    # stims_path = f"/Volumes/GoogleDrive/Shared drives/Varshini_Brea_Rohan/CoMLaM/Preprocessed/SPM/P{participant}_2k/sentiment/TRsToUse_AlphaOrder_P{participant}_2.xlsx"
    stims_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/TRsToUse_AlphaOrder_P1014_2.xlsx"

    stims_file = pd.read_excel(stims_path)

    all_stims = stims_file['combinedStim']

    # now load the congruency classification
    if exp_type == 'congruency':
        congruency_stims_file = np.load("/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/all_congruency.npz", allow_pickle=True)['arr_0'].tolist()
        stims_congruency = []
        for s in all_stims:
            stims_congruency.append(congruency_stims_file['labels'][s])
    elif exp_type == 'sentiment':
        sentiment_stims_file = np.load("/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/embeds/all_sentiment.npz", allow_pickle=True)['arr_0'].tolist()
        stims_congruency = []
        for s in all_stims:
            stims_congruency.append(sentiment_stims_file['labels'][s])
    elif exp_type == 'decoding':
        pass # TODO: Need to implement code for returning decoding embeddings.
    
    

    

    beta_file_numbers = [i for i in range(1,601)]
    # stim_numbers = [i for i in range(1,61)]
    # stims = np.tile(stim_numbers, 10)
    # stims = stims - 1
    print("Participant", participant)


    all_betas = []
    for beta in beta_file_numbers:
        file_number = str(int(beta)).zfill(4)
        # print(file_number)
        f = glob.glob(beta_path + f"*{file_number}.nii")
        # print(f)
        nifti = np.array(get_data(f[0])).astype(dtype='float32')
        nifti = nifti[~np.isnan(nifti)]
        all_betas.append(nifti)

    all_betas = np.array(all_betas)

    # Compute the noise ceiling.
    betas = all_betas
    betas = (betas - betas.mean(axis=0)) / (betas.std(axis=0) + 1e-7)


    # Now time for the preprocessing the stims.
    # TODO: Need to implement for returning embeds.

    if exp_type == 'sentiment' and remove_neutral:
        removed_idxs = []
        stims_congruency = np.array(stims_congruency)
        for idx, sent in enumerate(stims_congruency):
            if sent == 'Neut':
                removed_idxs.append(idx)
        betas = np.delete(betas, removed_idxs, axis=0)
        stims_congruency = np.delete(stims_congruency, removed_idxs, axis=0)

    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(stims_congruency)


    return betas, labels

def get_train_and_test_from_nc(betas, stims, seed):
    # Now divide the betas into training and testing sets and calculate the noise ceiling.
    # print("Stims: ", stims)
    betas_train, betas_test, stims_train, stims_test = train_test_split(betas, stims, test_size=0.2, random_state=seed, stratify=stims)
    train_ncsnr = compute_ncsnr(betas_train, stims_train)
    nc = compute_nc(train_ncsnr, num_averages=1)
    nc_value = np.percentile(nc, 95)
    train_mask = nc >= nc_value
    print("num of voxels", sum(train_mask))
    train_data = betas_train[:, train_mask]
    test_data = betas_test[:, train_mask]
    return train_data, test_data, stims_train, stims_test
        