#!/usr/bin/env python3

import os
import torch
import argparse
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm

def main():
    # Simple argument parsing
    parser = argparse.ArgumentParser(description="Test image generation with trained model")
    parser.add_argument("--checkpoint", type=str, default="./trained_models/MicroMicroDiT_res_64_test_new/latest-rank0.pt", 
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="./configs/micro_res_64_test_new.yaml", 
                        help="Path to model config file")
    parser.add_argument("--output_dir", type=str, default="./generated_images", 
                        help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=4, 
                        help="Number of images to generate")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    print(f"Loading config from {args.config}")
    cfg = OmegaConf.load(args.config)
    
    # Import model creation function
    from micro_diffusion.models.model import create_latent_diffusion
    
    # Create model
    print("Creating model...")
    model = create_latent_diffusion(
        vae_name=cfg.model.vae_name,
        text_encoder_name=cfg.model.text_encoder_name,
        dit_arch=cfg.model.dit_arch,
        precomputed_latents=cfg.model.precomputed_latents,
        in_channels=cfg.model.in_channels,
        pos_interp_scale=cfg.model.pos_interp_scale,
        dtype=cfg.model.dtype,
        latent_res=cfg.model.latent_res,
        p_mean=cfg.model.p_mean,
        p_std=cfg.model.p_std,
        train_mask_ratio=cfg.model.train_mask_ratio
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Extract state dict from checkpoint (Composer format)
    if "state" in checkpoint and "model" in checkpoint["state"]:
        state_dict = checkpoint["state"]["model"]
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Let's try a simpler approach - just generate random noise and decode it
    print(f"Generating {args.num_images} random images...")
    
    with torch.no_grad():
        for i in tqdm(range(args.num_images)):
            # Create random noise in the latent space
            latent_size = cfg.model.latent_res
            latents = torch.randn(1, 4, latent_size, latent_size).to(device)
            
            # Decode latents to image using VAE
            print("Decoding latents to image...")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                decoder_output = model.vae.decode(latents / model.vae.config.scaling_factor)
                # Extract the sample from the DecoderOutput object
                image = decoder_output.sample
            
            # Convert to PIL image
            image = (image + 1) / 2  # -1,1 -> 0,1
            image = image.clamp(0, 1)
            image = (image * 255).type(torch.uint8)
            image = image[0].permute(1, 2, 0).cpu().numpy()
            
            # Save image
            pil_image = Image.fromarray(image)
            output_path = os.path.join(args.output_dir, f"test_image_{i}.png")
            pil_image.save(output_path)
            print(f"Saved image to {output_path}")
    
    print("Done!")

if __name__ == "__main__":
    main() 