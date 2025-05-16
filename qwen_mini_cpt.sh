#!/bin/bash
#SBATCH --job-name=mini_qwen_test           # job name
#SBATCH --output=qwen_mini_cpt_%j.out      # STDOUT → hello_slurm_<jobid>.out
#SBATCH --error=qwen_mini_cpt_%j.err       # STDERR → hello_slurm_<jobid>.err
#SBATCH --time=02:00:00                  # time
#SBATCH --partition=k2-gpu-v100            # partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                       # single‐process job
#SBATCH --cpus-per-task=4                # one CPU
#SBATCH --mem=16G                 # 1 GB RAM
#SBATCH --mail-type=ALL       # email on start, end, or fail
#SBATCH --mail-user=josephmcinerney7575@gmail.com 
module load python3/3.10.5/gcc-9.3.0 # availible python
source /mnt/scratch2/users/40460549/cpt-dail/myenv_new/bin/activate
cd $SLURM_SUBMIT_DIR                     # ensure we’re in the project dir
srun python eval_base_qwen_mini.py 
srun python #mini_CPT.py 
srun python #eval_CPT_irish.py
               # run your script