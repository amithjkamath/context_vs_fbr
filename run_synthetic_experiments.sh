#!/bin/bash

for run in 1 2 3
do 
   for size in 32 48 64 80 96
   do
      python ./synthetic_experiments/synthetic_3d_wandb_unet.py $size
      python ./synthetic_experiments/synthetic_3d_wandb_unetr.py $size
      python ./synthetic_experiments/synthetic_3d_wandb_attention_unet.py $size
   done
done 