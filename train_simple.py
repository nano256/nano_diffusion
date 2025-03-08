#!/usr/bin/env python3

import os
import time
import hydra
import torch
import torch.nn as nn
import torch.utils.data
from composer import Trainer
from composer.models import ComposerModel
from composer.utils import reproducibility
from omegaconf import OmegaConf, DictConfig
import numpy as np
from composer.algorithms import GradientClipping
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from composer.core import Precision
from micro_diffusion.models.utils import text_encoder_embedding_format

@hydra.main(config_path='configs', config_name='micro_res_64_test', version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Main training function with a simplified dataset.
    """
    if not cfg:
        raise ValueError('Config not specified. Please provide --config-path and --config-name, respectively.')
    reproducibility.seed_all(cfg['seed'])
    
    print("Setting up model...")
    # Force precomputed_latents to true
    cfg.model.precomputed_latents = True
    print(f"Setting precomputed_latents to {cfg.model.precomputed_latents}")
    
    # Print model configuration for debugging
    print(f"Model configuration:")
    print(f"  Architecture: {cfg.model.dit_arch}")
    print(f"  Latent resolution: {cfg.model.latent_res}")
    print(f"  In channels: {cfg.model.in_channels}")
    
    # Instantiate the model
    model = hydra.utils.instantiate(cfg.model)
    print("Model instantiated successfully")
    
    # Create a simple dataset
    class SimpleLatentsDataset(torch.utils.data.Dataset):
        def __init__(self, size=100, image_size=64, cap_seq_size=77, cap_emb_dim=1024):
            self.size = size
            self.image_size = image_size
            self.cap_seq_size = cap_seq_size
            self.cap_emb_dim = cap_emb_dim
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Random latents and captions
            if self.image_size == 64:
                image_latents = torch.randn(4, 8, 8)  # 4 channels, 8x8 resolution for 64x64 images
            elif self.image_size == 256:
                image_latents = torch.randn(4, 32, 32)  # 4 channels, 32x32 resolution for 256x256 images
            else:
                image_latents = torch.randn(4, 64, 64)  # 4 channels, 64x64 resolution for 512x512 images
                
            caption_latents = torch.randn(1, self.cap_seq_size, self.cap_emb_dim)
            drop_caption_mask = torch.ones(1)
            
            return {
                'image_latents': image_latents,
                'caption_latents': caption_latents,
                'drop_caption_mask': drop_caption_mask
            }
    
    # Set up optimizer with special handling for MoE parameters
    print("Setting up optimizer...")
    moe_params = [p[1] for p in model.dit.named_parameters() if 'moe' in p[0].lower()]
    rest_params = [p[1] for p in model.dit.named_parameters() if 'moe' not in p[0].lower()]
    if len(moe_params) > 0:
        print('Reducing learning rate of MoE parameters by 1/2')
        opt_dict = dict(cfg.optimizer)
        opt_name = opt_dict['_target_'].split('.')[-1]
        del opt_dict['_target_']
        optimizer = getattr(torch.optim, opt_name)(
            params=[{'params': rest_params}, {'params': moe_params, 'lr': cfg.optimizer.lr / 2}], **opt_dict)
    else:
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.dit.parameters())
    print("Optimizer set up successfully")

    # Convert ListConfig betas to native list to avoid ValueError when saving optimizer state
    for p in optimizer.param_groups:
        p['betas'] = list(p['betas'])

    # Set up data loaders
    print("Setting up data loaders...")
    cap_seq_size, cap_emb_dim = text_encoder_embedding_format(cfg.model.text_encoder_name)
    
    # Print dataset configuration for debugging
    print(f"Dataset configuration:")
    print(f"  Image size: {cfg.dataset.image_size}")
    print(f"  Train batch size: {cfg.dataset.train_batch_size}")
    
    # Create datasets
    train_dataset = SimpleLatentsDataset(
        size=100, 
        image_size=cfg.dataset.image_size,
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim
    )
    eval_dataset = SimpleLatentsDataset(
        size=50, 
        image_size=cfg.dataset.image_size,
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataset.train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.dataset.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Created datasets with {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples")
    
    # Initialize training components
    print("Initializing training components...")
    logger, callbacks, algorithms = [], [], []

    # Set up loggers
    print("Setting up loggers...")
    for log, log_conf in cfg.logger.items():
        if '_target_' in log_conf:
            if log == 'wandb':
                wandb_logger = hydra.utils.instantiate(log_conf, _partial_=True)
                logger.append(wandb_logger(init_kwargs={'config': OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)}))
            else:
                logger.append(hydra.utils.instantiate(log_conf))

    # Configure algorithms
    print("Configuring algorithms...")
    if 'algorithms' in cfg:
        for alg_name, alg_conf in cfg.algorithms.items():
            if alg_name == 'low_precision_layernorm':
                apply_low_precision_layernorm(model=model.dit,
                                            precision=Precision(alg_conf['precision']),
                                            optimizers=optimizer)
            elif alg_name == 'gradient_clipping':
                algorithms.append(GradientClipping(clipping_type='norm', clipping_threshold=alg_conf['clip_norm']))
            else:
                print(f'Algorithm {alg_name} not supported.')

    # Set up callbacks
    print("Setting up callbacks...")
    if 'callbacks' in cfg:
        for _, call_conf in cfg.callbacks.items():
            if '_target_' in call_conf:
                print(f'Instantiating callbacks: {call_conf._target_}')
                callbacks.append(hydra.utils.instantiate(call_conf))

    print("Instantiating scheduler...")
    scheduler = hydra.utils.instantiate(cfg.scheduler)

    # disable online evals if using torch.compile
    if cfg.misc.compile:
        cfg.trainer.eval_interval = 0
        
    print("Instantiating trainer...")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        optimizers=optimizer,
        model=model,
        loggers=logger,
        algorithms=algorithms,
        schedulers=scheduler,
        callbacks=callbacks,
        precision='amp_bf16' if cfg.model['dtype'] == 'bfloat16' else 'amp_fp16',  # fp16 by default
        python_log_level='debug',
        compile_config={} if cfg.misc.compile else None  # it enables torch.compile (~15% speedup)
    )

    # Ensure models are on correct device
    print("Moving models to correct device...")
    device = next(model.dit.parameters()).device
    model.vae.to(device)
    model.text_encoder.to(device)

    print("Starting training...")
    return trainer.fit()


if __name__ == '__main__':
    train() 