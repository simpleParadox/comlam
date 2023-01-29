#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=00:08:00
#SBATCH --array=1-350
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=P1014_motor_all_runs_decoding_bertweet_50_iters_80_20_split_no_nc_no_loocv_perm
# SBATCH --output=%x-%j.out
#SBATCH --output=out_files/%x-%j.out

source ~/comlam/bin/activate

python main.py motor 1014 3 decoding True 6
##### Sixth arg is titration number, 7th argument is whether to use loocv = False/True. Remember, indices start from 0.

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=1 --mem-per-cpu=8000 --time=00:10:00