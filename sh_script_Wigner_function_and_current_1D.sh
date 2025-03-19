#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/Wigner_function_and_current_1D_%J.out
#SBATCH --cpus-per-task=16


module load scicomp-python-env

slurm python3 triton_Wigner_function_and_current_1D.py -f results/Wigner_function_and_current_1D