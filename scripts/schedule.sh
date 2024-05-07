#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py logger.wandb.name=without_batchnorm model.net._target_=src.models.components.cnn.AlexNet &
python src/train.py logger.wandb.name=with_batchnorm

