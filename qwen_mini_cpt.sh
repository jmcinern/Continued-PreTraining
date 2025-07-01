#!/bin/bash
#SBATCH --job-name=CPT_JM
#SBATCH --output=./out/qwen_mini_cpt_%j.out
#SBATCH --error=./err/qwen_mini_cpt_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=k2-gpu-v100  
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com
module load python3/3.10.5/gcc-9.3.0 # availible python
module load libs/nvidia-cuda/12.4.0/bin # cuda
module load openmpi #multi process
source /mnt/scratch2/users/40460549/cpt-dail/myenv_new/bin/activate
pip install --no-cache-dir -r "requirements.txt"
export WANDB_API_KEY="2dab6162cdfdc1b28724ac4ce95bb597d7f85994"
cd $SLURM_SUBMIT_DIR                     # ensure we’re in the project dir
deepspeed --num_gpus=2 mini_CPT.py


# OLD COMMANDS
# srun --partition=k2-gpu-interactive --gres=gpu:2 --time=1:30:00 --pty bash 
# view gpu partitions: sinfo -o "%P %G %D %C %t %N"  sinfo | grep gpu
#srun python eval_base_qwen_mini.py 
#srun python mini_CPT.py
#accelerate launch \
  #--num_processes 2 \
 # --mixed_precision no \
  #k2-gpu-v100, k2-gpu-interactive
