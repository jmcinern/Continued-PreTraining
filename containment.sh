#!/bin/bash
#SBATCH --job-name=containment
#SBATCH --output=./out/containment.%j.out
#SBATCH --error=./err/containment.%j.err
#SBATCH --time=00:01:00
#SBATCH --partition=k2-lowpri  # Changed from k2-gpu-v100
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=64  # Reduced from 256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com

# Load only required modules (no GPU modules needed)
module load python3/3.10.5/gcc-9.3.0
source /mnt/scratch2/users/40460549/cpt-dail/myenv_new/bin/activate
pip install -r requirements_containment.txt

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

python containment.py