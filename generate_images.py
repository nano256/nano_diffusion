#!/usr/bin/env python3

import os
import torch
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from micro_diffusion.models.model import create_latent_diffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using trained diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--prompt", type=str, default="a photo of a cat", help="Text prompt for image generation")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="Directory to save generated images")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    print(f"Loading config from {args.config}")
    cfg = OmegaConf.load(args.config)
    
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
    
    # Extract state dict from checkpoint
    if "state" in checkpoint:
        state_dict = checkpoint["state"]["model"]
    else:
        state_dict = checkpoint
    
    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Process prompt
    print(f"Generating {args.num_images} images with prompt: '{args.prompt}'")
    
    # Generate images
    with torch.no_grad():
        for i in tqdm(range(args.num_images)):
            # Encode text prompt
            text_embedding = model.text_encoder.encode([args.prompt])
            text_embedding = text_embedding.to(device)
            
            # Sample latents
            latents = torch.randn(1, 4, cfg.model.latent_res, cfg.model.latent_res).to(device)
            
            # Denoise latents
            latents = model.sample(
                latents,
                text_embedding,
                guidance_scale=args.guidance_scale,
                num_inference_steps=50
            )
            
            # Decode latents to image
            image = model.vae.decode(latents)
            
            # Convert to PIL image and save
            image = (image.clamp(-1, 1) + 1) / 2
            image = (image * 255).type(torch.uint8)
            image = image[0].permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
            
            # Save image
            output_path = os.path.join(args.output_dir, f"generated_{i:04d}.png")
            image.save(output_path)
            print(f"Saved image to {output_path}")

if __name__ == "__main__":
    main() 