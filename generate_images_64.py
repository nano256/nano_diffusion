# Sample generation script for MicroMicroDiT

import torch
from micro_diffusion.models.model import create_latent_diffusion
from PIL import Image
import matplotlib.pyplot as plt

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
    model.dit.load_state_dict(torch.load(model_path))
    
    # Generate images
    all_images = []
    batch_size = 4  # Process prompts in batches
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
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
    
    return all_images

def display_images(images, prompts=None, cols=4):
    """Display generated images in a grid."""
    rows = (len(images) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    if rows == 1 and cols == 1:
        axs = np.array([axs])
    elif rows == 1 or cols == 1:
        axs = axs.reshape(-1)
        
    for i, img in enumerate(images):
        if i < len(axs):
            # Convert tensor to PIL image if needed
            if isinstance(img, torch.Tensor):
                img = img.cpu().permute(1, 2, 0).numpy()
                
            axs[i].imshow(img)
            axs[i].axis('off')
            
            if prompts and i < len(prompts):
                axs[i].set_title(prompts[i][:40] + '...' if len(prompts[i]) > 40 else prompts[i])
    
    # Hide empty subplots
    for i in range(len(images), len(axs)):
        axs[i].axis('off')
        
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
    
    model_path = "./trained_models/MicroMicroDiT_mask_0_res_64_finetune/latest-rank0.pt"
    images = generate_images(prompts, model_path, num_images=1, guidance_scale=5.0)
    
    display_images(images, prompts)
    
    # Save individual images
    for i, img in enumerate(images):
        pil_img = Image.fromarray((img.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8'))
        pil_img.save(f"generated_image_{i}.png")