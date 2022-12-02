#!/bin/bash

SPLEEN_DATA_DIR="./data/"

for seed in 1 2 3
do 
   for size in 32 48 64 80 96
   do
      python ./spleen_experiments/spleen_3d_wandb_unet.py $size $SPLEEN_DATA_DIR
      python ./spleen_experiments/spleen_3d_wandb_unetr.py $size $SPLEEN_DATA_DIR
      python ./spleen_experiments/spleen_3d_wandb_attention_unet.py $size $SPLEEN_DATA_DIR
   done
done