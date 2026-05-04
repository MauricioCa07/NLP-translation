#!/bin/bash

#SBATCH --job-name="compiling"
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=59
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1-01:00:00
#SBATCH --output=logs/%x_%j.out      
#SBATCH --error=logs/%x_%j.err      


#==============================================================================
# SCRIPT BODY
#==============================================================================

echo "================================================="
echo "Starting job $SLURM_JOB_ID on host $(hostname)"
echo "Job name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Total number of tasks: $SLURM_NTASKS"
echo "================================================="
echo

# --- Environment Setup ---
#module load cuda/12.5
#module load cudnn-8.1.1.33-11.2-oneapi-2022.0.0-62fa2gf
module load miniconda3/25.5.1



#conda activate tf210
source venv_nuevo/bin/activate 


python test.py
#python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

echo
echo "================================================="
echo "Job finished with exit code $?"
echo "================================================="
