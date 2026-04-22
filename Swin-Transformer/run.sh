#!/bin/bash --login
#SBATCH --partition=gpu
#SBATCH --job-name=c04-s140-w00
#SBATCH -o out/output.out
#SBATCH -e out/error.err
#SBATCH -n 1
#SBATCH --tasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:2

# Run your machine learning script stretched-cnn-half-10.py
module load anaconda/2024.06
source activate swin

# Run training
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py --cfg configs/swinv2_c04_s140_w00.yaml --output out/train/ --accumulation-steps 32 --use-checkpoint

# Run evaluation
#python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main.py --cfg configs/swinv2/custom/swinv2_c04_s140_w00.yaml --output out/eval/ --accumulation-steps 32 --resume out/patch16_win24/c04-s140-w00/ckpt_epoch_99.pth --eval