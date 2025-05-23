#!/bin/bash
#SBATCH --job-name=mini_qwen_a100
#SBATCH --output=./out/qwen_mini_cpt_%j.out
#SBATCH --error=./err/qwen_mini_cpt_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --partition=k2-gpu-v100   #k2-gpu-v100, k2-gpu-interactive
# srun --partition=k2-gpu-interactive --gres=gpu:2 --time=30:00 --pty bash 
# view gpu partitions: sinfo -o "%P %G %D %C %t %N"  sinfo | grep gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com
module load python3/3.10.5/gcc-9.3.0 # availible python
module load libs/nvidia-cuda/12.4.0/bin # cuda
source /mnt/scratch2/users/40460549/cpt-dail/myenv_new/bin/activate
cd $SLURM_SUBMIT_DIR                     # ensure we’re in the project dir
#srun python eval_base_qwen_mini.py 
srun python mini_CPT.py

#accelerate launch \
  #--num_processes 2 \
 # --mixed_precision no \
 #   mini_CPT.py