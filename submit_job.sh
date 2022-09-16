#!/bin/bash
#SBATCH --mail-user=rsaha@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=def-afyshe-ab
#SBATCH --time=8:30:00
# SBATCH --array=1-450
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000
# SBATCH --job-name=12m_tgm_eeg_to_w2v_inside_scaling_no_seed_second_run
#SBATCH --job-name=9m_animacy_noise_ceiling_50_iters_no_seed_90_percentile
# SBATCH --output=out_files/%x-%j.out
#SBATCH --output=%x-%j.out

source ~/comlam/bin/activate

# python classification/notebooks/cluster_analysis_perm_reg_overlap.py $SLURM_ARRAY_TASK_ID
python main.py

##############--job-name=9m_perm_avg_trials_ps-w2v_from_eeg_09-07-2021_100-10-50iters-shift-r-50
################ salloc --account=def-afyshe-ab --cpus-per-task=2 --mem-per-cpu=8000 --time=00:10:00

pip install -U scikit-learn
pip install pickle
pip install nilearn
pip install gensim