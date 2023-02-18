#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o posttrain_procy_qa-%j.out
#SBATCH --gres gpu:2

export HF_DATASETS_CACHE='/sdb/zke4/dataset_cache'
export TRANSFORMERS_CACHE='/sdb/zke4/model_cache'
#export TRANSFORMERS_OFFLINE=1
max_samples=640000

for idrandom in  0
do
  for pt_task in 0
  do
    python -m torch.distributed.launch --nproc_per_node 4 --use_env posttrain.py \
    --per_device_train_batch_size 62 \
    --fp16\
    --max_seq_length 164 \
    --max_samples ${max_samples} \
    --idrandom ${idrandom} \
    --ntasks 6 \
    --pt_task ${pt_task} \
    --baseline 'dga'
  done
done

