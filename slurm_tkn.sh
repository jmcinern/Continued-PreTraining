#!/bin/bash
#SBATCH --job-name=Qwen_Tokenizer_Train
#SBATCH --output=./out/tkn_train_%j.out
#SBATCH --error=./err/tkn_train_%j.err
#SBATCH --time=00:45:00
#SBATCH --partition=k2-lowpri  # Using a CPU partition is better for this task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   # Increased CPUs as tokenizer training can be parallelized
#SBATCH --mem=128G          # Requesting 128GB of RAM, which should be plenty

# --- Environment Setup (copied from your other script) ---
echo "Loading modules..."
module load python3/3.10.5/gcc-9.3.0
source /mnt/scratch2/users/40460549/cpt-dail/myenv_new/bin/activate
echo "Environment activated."

# --- Install Dependencies ---
echo "Installing requirements from requirements_tokenizer_train.txt..."
pip install --no-cache-dir -r "requirements_tokenizer_train.txt"
echo "Installation complete."

# --- Run the Tokenizer Script ---
echo "Changing to submission directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

export HF_KEY=""
python=qwen_tokenizer.py
