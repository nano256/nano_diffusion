#!/bin/bash

# Preparation: Clean up memory
pkill -9 python
python -c 'from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()'
rm -rf /tmp/streaming/*
sleep 3

# Step 1: Precompute 64x64 latents
accelerate launch --multi_gpu --num_processes 2 micro_diffusion/datasets/prepare/jdb/precompute_64.py \
    --datadir ./datadir/jdb/mds/train/ \
    --savedir ./datadir/jdb/mds_latents_sdxl1_dfnclipH14_64/train/ \
    --vae stabilityai/stable-diffusion-xl-base-1.0 \
    --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
    --batch_size 64 \
    --image_resolutions 64

# Step 2: Train 64x64 model with masking
composer train.py --config-path ./configs --config-name micro_res_64_pretrain.yaml \
    exp_name=MicroMicroDiT_mask_75_res_64_pretrain model.train_mask_ratio=0.75

# Step 3: Finetune 64x64 model without masking 
composer train.py --config-path ./configs --config-name micro_res_64_pretrain.yaml \
    exp_name=MicroMicroDiT_mask_0_res_64_finetune model.train_mask_ratio=0.0 \
    trainer.load_path=./trained_models/MicroMicroDiT_mask_75_res_64_pretrain/latest-rank0.pt