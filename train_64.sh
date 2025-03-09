#!/bin/bash

# install requirements 
pip install -e .

mkdir -p ./datadir/jdb/mds/train/

# login to hf
huggingface-cli login

# DL the JDB - MODIFIED TO DOWNLOAD MORE DATA
# Downloading 50 archives (0-49) for a substantial dataset size
# This will create a dataset ~25x larger than the original 2 archives
python micro_diffusion/datasets/prepare/jdb/download.py --datadir ./datadir/jdb/ --max_image_size 512 --min_image_size 256 --valid_ids $(seq 0 49) --num_proc 8

# convert to MDS
python micro_diffusion/datasets/prepare/jdb/convert.py --images_dir "./datadir/jdb/raw/train/imgs/" --captions_jsonl "./datadir/jdb/raw/train/train_anno_realease_repath.jsonl" --local_mds_dir "./datadir/jdb/mds/train/"

# run precompute script 64x64
# YOU MUST HAVE CUDA working
accelerate launch --gpu_ids=0 micro_diffusion/datasets/prepare/jdb/precompute_64.py \
  --datadir ./datadir/jdb/mds/train/ \
  --savedir ./datadir/jdb/mds_latents_sdxl1_dfnclipH14_64/train/ \
  --vae stabilityai/stable-diffusion-xl-base-1.0 \
  --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 \
  --batch_size 128 \
  --image_resolutions 64

# run train.py
composer train.py --config-path ./configs --config-name micro_res_64_pretrain.yaml \
    exp_name=MicroMicroDiT_mask_75_res_64_pretrain model.train_mask_ratio=0.75

# Step 3: Finetune 64x64 model without masking 
composer train.py --config-path ./configs --config-name micro_res_64_pretrain.yaml \
    exp_name=MicroMicroDiT_mask_0_res_64_finetune model.train_mask_ratio=0.0 \
    trainer.load_path=./trained_models/MicroMicroDiT_mask_75_res_64_pretrain/latest-rank0.pt