import pandas as pd
import numpy as np
import math
from nilearn import image as img
import pickle as pk
import matplotlib.pyplot as plt
import os
import glob
import regex as re


###
def load_nifti(path):
    np_a = img.get_data(path)

    # Transpose the nifti file so that there are 'n' (324) images of 102x102x64(slices) of them.
    np_a = np.transpose(np_a, (3, 0, 1, 2))

    # For each 3d fMRI image, concatenate the volumetric pixels to get one single array.
    np_a_rs = np_a.reshape(np_a.shape[0], -1)

    mean_voxels_a = np.mean(np_a_rs, axis=0)

    for i, row in enumerate(np_a_rs):
        np_a_rs[i] = np_a_rs[i] - mean_voxels_a

    return np_a_rs

path = "E:\My Drive\CoMLaM_rohan\CoMLaM\Preprocessed\Reg_to_Std_and_Str\P_1003\Synonym_RunA_13.feat\\filtered_func_dataA.nii"
load_nifti(path)


def ridge():
    # Do ridge regression with GridSearchCV here.
    pass