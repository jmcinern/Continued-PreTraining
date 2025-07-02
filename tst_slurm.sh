#!/bin/bash
#SBATCH --job-name=TEST_CPT_JM
#SBATCH --output=./out/test_cpt_%j.out
#SBATCH --error=./err/test_cpt_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=k2-gpu-v100  
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com

module load python3/3.10.5/gcc-9.3.0 # available python
module load libs/nvidia-cuda/12.4.0/bin # cuda
source /mnt/scratch2/users/40460549/cpt-dail/myenv_new/bin/activate
pip install --no-cache-dir -r "tst_requirements.txt"
cd $SLURM_SUBMIT_DIR

# Run the inference test script
python AAA_test_CPT.py