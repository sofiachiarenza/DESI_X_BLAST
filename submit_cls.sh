#!/bin/bash
#SBATCH --job-name=QSO                  # Give your job a meaningful name
#SBATCH --output=log/%x.out          # Save standard output to log folder
#SBATCH --error=log/%x.err           # Save standard error to log folder
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=40              # Use all available cores on the node
#SBATCH --mem=0                         # Request all available memory
#SBATCH --time=01:00:00                 # Wall time
#SBATCH -p defq                         # Partition
#SBATCH --mail-user=schiarenza@perimeterinstitute.ca
#SBATCH --mail-type=ALL

srun python compute_cell_QSO_new.py