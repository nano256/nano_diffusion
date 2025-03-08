#!/usr/bin/env python3

import os
import time
import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
from composer import Trainer
from composer.models import ComposerModel
from composer.utils import reproducibility
from omegaconf import OmegaConf
import numpy as np
from composer.algorithms import GradientClipping

@hydra.main(config_path='configs', config_name='micro_res_64_test', version_base=None)
def train(cfg):
    """
    Main training function.
    """
    if not cfg:
        raise ValueError('Config not specified. Please provide --config-path and --config-name, respectively.')
    reproducibility.seed_all(cfg['seed'])

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 4, 3, padding=1)
            )
            
        def forward(self, x):
            return self.encoder(x)
    
    # Create a simple dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Random latents and captions
            image_latents = torch.randn(4, 8, 8)
            caption_latents = torch.randn(1, 77, 1024)
            drop_caption_mask = torch.ones(1)
            
            return {
                'image_latents': image_latents,
                'caption_latents': caption_latents,
                'drop_caption_mask': drop_caption_mask
            }
    
    # Create a simple model that implements ComposerModel
    class SimpleComposerModel(ComposerModel):
        def __init__(self):
            super().__init__()
            self.model = SimpleModel()
            
        def forward(self, batch):
            x = batch['image_latents']
            y = self.model(x)
            loss = torch.nn.functional.mse_loss(y, x)
            return (loss, x, batch['caption_latents'])
            
        def loss(self, outputs, batch):
            return outputs[0]
        
        def eval_forward(self, batch, outputs=None):
            if outputs is None:
                outputs = self.forward(batch)
            return outputs
        
        def get_metrics(self, is_train=False):
            return {}
    
    # Create datasets
    train_dataset = SimpleDataset(size=100)
    eval_dataset = SimpleDataset(size=50)
    
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
    
    # Create model
    model = SimpleComposerModel()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        eps=cfg.optimizer.eps,
        betas=cfg.optimizer.betas
    )
    
    # Create scheduler
    scheduler = hydra.utils.instantiate(cfg.scheduler)
    
    # Initialize training components
    logger, callbacks, algorithms = [], [], []
    
    # Set up loggers
    if hasattr(cfg, 'logger'):
        for log, log_conf in cfg.logger.items():
            if '_target_' in log_conf:
                logger.append(hydra.utils.instantiate(log_conf))
    
    # Set up algorithms
    algorithms.append(GradientClipping(clipping_type='norm', clipping_threshold=cfg.algorithms.gradient_clipping.clip_norm))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.trainer.max_duration,
        eval_interval=cfg.trainer.eval_interval,
        save_interval=cfg.trainer.save_interval,
        save_folder=cfg.trainer.save_folder,
        save_overwrite=cfg.trainer.save_overwrite,
        save_num_checkpoints_to_keep=cfg.trainer.save_num_checkpoints_to_keep,
        device=cfg.trainer.device,
        precision='amp_bf16',
        seed=cfg.trainer.seed,
        algorithms=algorithms,
        loggers=logger,
        run_name=cfg.trainer.run_name,
        device_train_microbatch_size=cfg.trainer.device_train_microbatch_size
    )
    
    # Train
    return trainer.fit()

if __name__ == '__main__':
    train() 