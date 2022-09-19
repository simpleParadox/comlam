import numpy as np
import pandas as pd
from scipy import stats
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from mne.stats import fdr_correction


def permutation_test_titration(participant, runs, obs_acc, exp_type):
    # First read the permutation accuracy files.
    permutation_results_path = f"/home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/saved_results/permutation/{exp_type}/{participant}/"
    perm_results = []
    run_p_values = {}
    non_corrected_p_values = []
    for f in glob.glob(permutation_results_path + "*.npz"):
        # 'f' is the file name.
        results = np.load(f, allow_pickle=True)['arr_0'].tolist()
        # print(results)
        perm_results.append(results)
    print("Len perm_results: ", len(perm_results))
    
    
    for idx, run in enumerate(runs):
        run_perm_values = []
        for i in range(len(perm_results)):
            # print(perm_results[i][participant])
            # break
            current_perm_result = perm_results[i][participant][run]  # This is a 50 dimensional vector (or whatever the number of cv iters is).
            # run_perm_values.extend(current_perm_result)
            run_perm_values.append(np.mean(current_perm_result))
        print("Length of run_perm_values: ", len(run_perm_values))
        # Now estimate a null distribution using kde.
        non_perm_acc = obs_acc[idx]
        kde = stats.gaussian_kde(run_perm_values)
        # Now integrate from the obs_acc to 1.
        p_value = kde.integrate_box_1d(non_perm_acc, 1.0)
        non_corrected_p_values.append(p_value)
        run_p_values[run] = p_value
        sns.kdeplot(run_perm_values)
        plt.savefig(f"{participant}_{exp_type}_{run}.png")

    # Do p-value correction for multiple comparisons.
    alpha_level = 0.05
    reject_fdr, pvalues_fdr = fdr_correction(non_corrected_p_values, alpha=alpha_level, method='negcorr')

    
    return run_p_values, pvalues_fdr



participant = 1014
runs = [4, 5, 6, 7, 8, 9, 10]
obs_acc = [ 0.3616,
0.465,
0.5016,
0.45,
0.385,
0.412,
0.47
]
run_p_values, pvalues_fdr = permutation_test_titration(participant, runs, obs_acc,'congruency')
print("P-values are: ", run_p_values)
print("Corrected p-values are: ", pvalues_fdr)