#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=00:40:00
#SBATCH --array=1-550
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=1014_decoding_twitter_w2v_avg_perm_motor
#SBATCH --output=out_files/%x-%j.out

source ~/comlam/bin/activate

python main.py motor 1014 3 decoding True 6 /home/rsaha/projects/def-afyshe-ab/rsaha/projects/comlam/data/spm/sentiment/other/P1014_2k/NonZeroShot/Twos/P1014_2k_beta_dict.npz
##### 7th argument is whether to use loocv = False/True. Remember, indices start from 0.

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=1 --mem-per-cpu=8000 --time=00:10:00
