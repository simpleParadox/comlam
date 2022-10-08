#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=01:00:00
# SBATCH --array=1-100
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000
#SBATCH --job-name=1030_and_1032_motor_area_2_way_sentiment_pos_neg_only
#SBATCH --output=%x-%j.out
# SBATCH --output=out_files/%x-%j.out

source ~/comlam/bin/activate

python main.py

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=8 --mem-per-cpu=8000 --time=00:20:00