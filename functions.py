import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nilearn import image as img

def get_avg_tr(tr_path, nifti_path):
    metadata = pd.read_csv(tr_path)
    nifti = img.get_data(nifti_path)

    # Now preprocess the nifti and store it in a directory called 'avg_tr_nifti' on GDrive.




def map_stimuli_w2v():
    pass
