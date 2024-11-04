#!/bin/bash

#SBATCH --job-name=lab1
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --mem=2GB
#SBATCH --account=COMS033444

module load languages/python/tensorflow-2.16.1
echo "Start"
python train_mnist.py
echo "Done"
