#!/bin/bash --login
#SBATCH --partition=gpu
#SBATCH --job-name=res-04
#SBATCH -o out/c04-w10-1536/output.out
#SBATCH -e out/c04-w10-1536//error.err
#SBATCH -n 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=340G

module load anaconda/2024.06
source activate cnn

# Run training
#python main.py
torchrun --nproc_per_node=2 main-c04.py
#torchrun --nproc_per_node=2 main-c04.py eval=True ckpt=epoch_7.pt