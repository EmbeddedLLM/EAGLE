#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch -m --num_processes 4 --mixed_precision=bf16 train.main --tmpdir /media/data4/eagle_experiments/data/sharegpt_0_67999_mufp16 \
--configpath /home/tan/tjtanaa/EAGLE/train/vicuna_7B_config.json \
--outdir /media/data4/eagle_experiments/vicuna7b \
--cpdir /media/data4/eagle_experiments/vicuna7b/cpdir \
--basepath /media/data3/Storage/lmsys_vicuna-7b-v1.3