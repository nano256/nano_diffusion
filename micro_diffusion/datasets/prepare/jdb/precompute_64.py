# Modified precompute.py script to generate 64x64 latents

import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm
import shutil
import glob
from pathlib import Path

from micro_diffusion.datasets.prepare.jdb.base import (
    build_streaming_jdb_precompute_dataloader,
)
from micro_diffusion.models.utils import UniversalTextEncoder, DATA_TYPES

def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Local directory to store mds shards.',
    )
    parser.add_argument(
        '--savedir',
        type=str,
        default='',
        help='Remote path to upload MDS-formatted shards to.',
    )
    parser.add_argument(
        '--image_resolutions',
        type=int,
        nargs='+',
        default=[64],  # Only 64x64 resolution
        help='List of image resolutions to use for processing.',
    )
    parser.add_argument(
        '--save_images',
        default=False,
        action='store_true',
        help='If True, also save images, else only latents',
    )
    parser.add_argument(
        '--model_dtype',
        type=str,
        choices=('float16', 'bfloat16', 'float32'),
        default='bfloat16',
        help='Data type for the encoding models',
    )
    parser.add_argument(
        '--save_dtype',
        type=str,
        choices=('float16', 'float32'),
        default='float16',
        help='Data type to save the latents',
    )
    parser.add_argument(
        '--vae',
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='Name of VAE model to use for vision encoding.',
    )
    parser.add_argument(
        '--text_encoder',
        type=str,
        default='openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378',
        help='Name of model to use for text encoding.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size per device to use for encoding.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2024,
        help='Seed for random number generation.',
    )
    parser.add_argument(
        '--use_raw_data',
        action='store_true',
        help='Use raw data directory instead of MDS directory',
    )
    parser.add_argument(
        '--overwrite_index',
        action='store_true',
        help='Overwrite existing index.json file if it exists',
    )
    args = parser.parse_args()
    if isinstance(args.image_resolutions, int):
        args.image_resolutions = [args.image_resolutions]
    return args


def main(args):
    """Precompute image and text latents and store them in MDS format."""
    cap_key = 'caption'

    accelerator = Accelerator()
    device = accelerator.device
    device_idx = int(accelerator.process_index)

    # Set random seeds
    torch.manual_seed(device_idx + args.seed)
    torch.cuda.manual_seed(device_idx + args.seed)
    np.random.seed(device_idx + args.seed)

    # Check if the datadir exists and has the expected structure
    if not os.path.exists(args.datadir):
        print(f"Error: Data directory {args.datadir} does not exist")
        return
    
    # If using MDS format, check if index.json exists
    if not args.use_raw_data and not os.path.exists(os.path.join(args.datadir, "index.json")):
        print(f"Warning: index.json not found in {args.datadir}")
        print("Please use --use_raw_data flag to use raw data directory instead")
        print("Attempting to use raw data directory...")
        
        # Try to find raw data directory
        raw_data_path = args.datadir.replace("mds", "raw")
        if os.path.exists(raw_data_path):
            args.datadir = raw_data_path
            args.use_raw_data = True
            print(f"Using raw data directory: {args.datadir}")
        else:
            print(f"Error: Raw data directory {raw_data_path} not found")
            return

    # Create save directory if it doesn't exist
    if args.savedir and not os.path.exists(args.savedir):
        os.makedirs(args.savedir, exist_ok=True)
        print(f"Created save directory: {args.savedir}")

    # Load dataset with 64x64 resolution
    try:
        dataloader = build_streaming_jdb_precompute_dataloader(
            datadir=[args.datadir],
            batch_size=args.batch_size,
            resize_sizes=args.image_resolutions,
            drop_last=False,
            shuffle=False,
            caption_key=cap_key,
            tokenizer_name=args.text_encoder,
            prefetch_factor=2,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            use_raw_data=args.use_raw_data,
        )
        print(f'Device: {device_idx}, Dataloader sample count: {len(dataloader.dataset)}')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load models
    vae = AutoencoderKL.from_pretrained(
        args.vae,
        subfolder='vae',
        torch_dtype=DATA_TYPES[args.model_dtype],
    )
    print("Created VAE: ", args.vae)
    assert isinstance(vae, AutoencoderKL)

    text_encoder = UniversalTextEncoder(
        args.text_encoder,
        dtype=args.model_dtype,
        pretrained=True,
    )
    print("Created text encoder: ", args.text_encoder)

    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    # Set up columns for latent data
    columns = {
        cap_key: 'str',
        f'{cap_key}_latents': 'bytes',
        'latents_64': 'bytes',  # Only save 64x64 latents
    }
    if args.save_images:
        columns['jpg'] = 'jpeg'

    remote_upload = os.path.join(args.savedir, str(accelerator.process_index))
    writer = MDSWriter(
        out=remote_upload,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
        exist_ok=True,
    )

    for batch in tqdm(dataloader):
        image_64 = torch.stack(batch['image_0']).to(device)
        captions = torch.stack(batch[cap_key]).to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=DATA_TYPES[args.model_dtype]):
                # Encode 64x64 images to latents (8x8)
                latent_dist_64 = vae.encode(image_64)
                assert isinstance(latent_dist_64, AutoencoderKLOutput)
                latents_64 = (
                    latent_dist_64['latent_dist'].sample().data * vae.config.scaling_factor
                ).to(DATA_TYPES[args.save_dtype])

                # Encode text
                attention_mask = None
                if f'{cap_key}_attention_mask' in batch:
                    attention_mask = torch.stack(
                        batch[f'{cap_key}_attention_mask']
                    ).to(device)

                conditioning = text_encoder.encode(
                    captions.view(-1, captions.shape[-1]),
                    attention_mask=attention_mask,
                )[0].to(DATA_TYPES[args.save_dtype])

        try:
            if isinstance(latents_64, torch.Tensor):
                latents_64 = latents_64.detach().cpu().numpy()
            else:
                continue

            if isinstance(conditioning, torch.Tensor):
                conditioning = conditioning.detach().cpu().numpy()
            else:
                continue

            # Write the batch to the MDS file
            for i in range(latents_64.shape[0]):
                mds_sample = {
                    cap_key: batch['sample'][i][cap_key],
                    f'{cap_key}_latents': np.reshape(conditioning[i], -1).tobytes(),
                    'latents_64': latents_64[i].tobytes(),
                }
                if args.save_images:
                    mds_sample['jpg'] = batch['sample'][i]['jpg']
                writer.write(mds_sample)
        except RuntimeError:
            print('Runtime error CUDA, skipping this batch')

    writer.finish()

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    print(f'Process {accelerator.process_index} finished')
    time.sleep(10)

    # Merge the mds shards created by each device (only do on main process)
    if accelerator.is_main_process:
        shards_metadata = [
            os.path.join(args.savedir, str(i), 'index.json')
            for i in range(accelerator.num_processes)
        ]
        
        # Check if index.json already exists
        index_path = os.path.join(args.savedir, 'index.json')
        if os.path.exists(index_path) and not args.overwrite_index:
            print(f"Warning: {index_path} already exists. Skipping merge_index.")
            print("Use --overwrite_index flag to overwrite the existing index file.")
        else:
            # If overwrite_index is True or the file doesn't exist, proceed with merge
            if os.path.exists(index_path) and args.overwrite_index:
                print(f"Removing existing index file: {index_path}")
                os.remove(index_path)
            
            try:
                merge_index(shards_metadata, out=args.savedir, keep_local=True)
                print(f"Successfully merged index files to {index_path}")
            except Exception as e:
                print(f"Error merging index files: {e}")


if __name__ == '__main__':
    main(parse_args())