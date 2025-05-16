#!/bin/bash
#SBATCH --job-name=mini_qwen_test           # job name
#SBATCH --output=hello_slurm_%j.out      # STDOUT → hello_slurm_<jobid>.out
#SBATCH --error=hello_slurm_%j.err       # STDERR → hello_slurm_<jobid>.err
#SBATCH --time=02:00:00                  # time
#SBATCH --partition=k2-hipri             # partition
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1                       # single‐process job
#SBATCH --cpus-per-task=4                # one CPU
#SBATCH --mem-per-cpu=16G                 # 1 GB RAM
#SBATCH --mail-type=ALL       # email on start, end, or fail
#SBATCH --mail-user=josephmcinerney7575@gmail.com 
module load python3/3.10.5/gcc-9.3.0 # availible python
cd $SLURM_SUBMIT_DIR                     # ensure we’re in the project dir
srun python mini_CPT.py               # run your script