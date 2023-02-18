#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 10:00:00
#SBATCH -o posttrain_procy_qa-%j.out
#SBATCH --gres gpu:4

export HF_DATASETS_CACHE='/sdb/zke4/dataset_cache'
export TRANSFORMERS_CACHE='/sdb/zke4/model_cache'
#export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=3 python playground.py
