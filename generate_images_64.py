# Sample generation script for MicroMicroDiT

import torch
import numpy as np
from micro_diffusion.models.model import create_latent_diffusion
from PIL import Image
import matplotlib.pyplot as plt
import os

def generate_images(prompts, model_path, num_images=1, guidance_scale=5.0, steps=30, seed=42):
    """Generate images using a trained MicroMicroDiT model."""
    # Create model with 64x64 resolution configuration (8x8 latents)
    model = create_latent_diffusion(
        latent_res=8,  # 64/8 = 8
        in_channels=4,
        pos_interp_scale=0.25,  # Lower for 64x64
        dit_arch="MicroMicroDiT_Nano"
    ).to('cuda')
    
    # Load trained weights
    # Note: Using weights_only=False has security implications if loading untrusted models
    # See: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
    checkpoint = torch.load(model_path)
    if 'state' in checkpoint and 'model' in checkpoint['state']:
        # Extract model weights from the nested state dictionary
        model.dit.load_state_dict({k.replace('dit.', ''): v for k, v in checkpoint['state']['model'].items() if k.startswith('dit.')})
    else:
        # Fallback to direct loading
        model.dit.load_state_dict(checkpoint)
    
    # Generate images
    all_images = []
    batch_size = 4  # Process prompts in batches
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        print(f"Generating images for prompts: {batch_prompts}")
        
        # Generate multiple images per prompt if requested
        for j in range(num_images):
            current_seed = seed + j if seed is not None else None
            images = model.generate(
                prompt=batch_prompts,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=current_seed
            )
            all_images.extend(images)
            print(f"Generated {len(images)} images")
    
    return all_images

def display_images(images, prompts=None, cols=4):
    """Display generated images in a grid."""
    rows = (len(images) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    # Convert to numpy array for consistent indexing
    axs = np.array(axs)
    
    # Ensure axs is 2D for consistent indexing
    if rows == 1 and cols == 1:
        axs = axs.reshape(1, 1)
    elif rows == 1:
        axs = axs.reshape(1, cols)
    elif cols == 1:
        axs = axs.reshape(rows, 1)
        
    for i, img in enumerate(images):
        row_idx = i // cols
        col_idx = i % cols
        
        # Convert tensor to numpy array if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
            
        axs[row_idx, col_idx].imshow(img)
        axs[row_idx, col_idx].axis('off')
        
        if prompts and i < len(prompts):
            axs[row_idx, col_idx].set_title(prompts[i][:40] + '...' if len(prompts[i]) > 40 else prompts[i])
    
    # Hide empty subplots
    for i in range(len(images), rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        axs[row_idx, col_idx].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    prompts = [
        "a cat wearing sunglasses",
        "a red sports car",
        "an astronaut on the moon",
        "a slice of pizza",
        "a tropical beach with palm trees",
        "a castle on a hill",
        "a robot playing guitar",
        "a snow-covered mountain"
    ]
    
    model_path = "./trained_models/MicroMicroDiT_mask_75_res_64_pretrain/latest-rank0.pt"
    print(f"Generating images using model: {model_path}")
    print(f"Prompts: {prompts}")
    
    images = generate_images(prompts, model_path, num_images=1, guidance_scale=5.0)
    print(f"Generated {len(images)} images")
    
    # Save individual images
    output_dir = "outputs/generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.cpu().permute(1, 2, 0).numpy()
            # Convert from float [0,1] to uint8 [0,255]
            img = (img * 255).astype('uint8')
        
        output_path = f"{output_dir}/generated_image_{i}_{prompts[i % len(prompts)].replace(' ', '_')[:30]}.png"
        pil_img = Image.fromarray(img)
        pil_img.save(output_path)
        print(f"Saved image to: {output_path}")
    
    print("Image generation complete!")
    
    # Optionally display images if in an interactive environment
    try:
        display_images(images, prompts)
    except Exception as e:
        print(f"Could not display images: {e}")
        print("Images have been saved to disk instead.")