#!/usr/bin/env bash
srun \
    --partition gpu \
    --gres gpu:1 \
    -A comsm0018 \
    --reservation comsm0018-lab1 \
    --mem 64GB \
    -t 0-00:15 \
    python train_mnist.py
