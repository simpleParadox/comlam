#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=01:00:00
#SBATCH --array=1-550
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=Pilot_08_Feb_decoding_w2vconcat_perm
# SBATCH --output=%x-%j.out
#SBATCH --output=out_files/%x-%j.out

source ~/comlam/bin/activate

python main.py motor 1014 3 decoding True 6
##### 7th argument is whether to use loocv = False/True. Remember, indices start from 0.

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=1 --mem-per-cpu=8000 --time=00:10:00
