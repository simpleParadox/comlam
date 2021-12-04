#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=rrg-afyshe
#SBATCH --time=10:00:00
# SBATCH --array=42-120
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=10000
#SBATCH --job-name=12m_w2v_from_eeg_no_base_no_highfilter
#SBATCH --output=%x-%j.out

source ~/base/bin/activate

python main.py
##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=rrg-afyshe --cpus-per-task=1 --mem-per-cpu=8000 --time=1:00:00