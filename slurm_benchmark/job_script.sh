#!/bin/bash
#SBATCH --job-name=my_conda_job  # Job name
#SBATCH --output=my_conda_job.out  # Standard output and error log
#SBATCH --error=my_conda_job.err  # Error log file
#SBATCH --partition=general  # Partition name
#SBATCH --nodes=1  # Number of nodes
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=4  # Number of CPU cores per task
#SBATCH --mem=60G  # Total memory
#SBATCH --time=01:00:00  # Time limit hrs:min:sec

# Load Conda module
module load miniconda  # Adjust based on your system configuration

# Initialize Conda
source $CONDA_PREFIX/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate myenv

# Run your Python script
python slurm_benchmark.py 
