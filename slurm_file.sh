#!/bin/bash

#SBATCH --job-name="Transformers-Model"
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=10G
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
module load cuda/12.5
module load miniconda3/25.5.1


source tf/bin/activate    



python train.py
#python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
#python3 -c "import tensorflow as tf; print(tf.__version__)"
#python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"


echo
echo "================================================="
echo "Job finished with exit code $?"
echo "================================================="
