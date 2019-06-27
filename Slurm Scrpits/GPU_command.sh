#!/bin/sh
#SBATCH --nodes=1 #Node allocation
#SBATCH --cpus-per-task=48 #Number of thread allocation
#SBATCH --priority=TOP 
#SBATCH --partition=gpu_v100 # Partition Allocation More details about the partion available at ara-wiki.rz.uni-jena.de/Hauptseite
#SBATCH --gres=gpu:1 # Number of GPUs , If not used the code doesn't run on GPUs instead it will run on CPUs
#SBATCH --time=200:00:00 # Maximum amount of running time
#SBATCH --no-kill
#SBATCH --mail-type=All
#SBATCH --mail-user=user@gmail.com 
#SBATCH --no-requeue
module load nvidia/cuda/9 #loading Modules
module load tools/tensorflow/1.8.0

time python Cnn_modified_alexnet.py # Command to run the desired code
