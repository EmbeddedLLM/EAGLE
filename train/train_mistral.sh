#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch -m --num_processes 1 --mixed_precision=bf16 train.main_mistral \
--tmpdir /media/data4/eagle_experiments/data_mistral_instruct/sharegpt_0_67999_mufp16 \
--configpath /home/tan/tjtanaa/EAGLE/train/mistral_7B_instruct_config.json \
--outdir /media/data4/eagle_experiments/mistral7binstruct \
--cpdir /media/data4/eagle_experiments/mistral7binstruct/cpdir \
--basepath /media/data3/Storage/mistralai_Mistral-7B-Instruct-v0.2 \
--bs 1