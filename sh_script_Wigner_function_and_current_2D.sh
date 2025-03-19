#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=3000M
#SBATCH --output=output-13-3.out
#SBATCH --cpus-per-task=5


module load scicomp-python-env

slurm python3 triton_Wigner_function_and_current_2D.py -f ${WRKDIR}/output_file_Wigner_function_and_current_2D