#!/bin/bash

#SBATCH --job-name=lab1
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --mem=2GB

module load languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch
echo "Start"
python train_mnist.py
echo "Done"
