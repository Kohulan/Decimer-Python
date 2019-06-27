#!/bin/sh
#SBATCH --nodes= 2 #Node allocation
#SBATCH --cpus-per-task=48 #Number of thread allocation
#SBATCH --priority=TOP
#SBATCH --partition=b_fat #Partition Allocation More details about the partion available at ara-wiki.rz.uni-jena.de/Hauptseite
#SBATCH --time=200:00:00 #Maximum amount of running time
#SBATCH --no-kill 
#SBATCH --mail-type=All
#SBATCH --mail-user=user@gmail.com
#SBATCH --no-requeue
module load nvidia/cuda/9 # loading modules
module load tools/tensorflow/1.8.0

time python 3layernet_0.004_modified_2048.py
