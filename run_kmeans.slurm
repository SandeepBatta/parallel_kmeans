#!/bin/bash

#SBATCH -J py_kmeans
#SBATCH -p general
#SBATCH -A xxxxxx
#SBATCH -o %j.txt
#SBATCH -e %j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sabatta@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=30G


module load python
# module load cuda
module load openmpi

# Activate a virtual env if you have one
source ./venv/bin/activate

srun python parallel_kmeans.py --n_points 500000 --k 3 --dim 2 --max_iter 1000                                                                                                                                                                               13,15         All