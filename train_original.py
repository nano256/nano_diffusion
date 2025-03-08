#!/usr/bin/env python3

import time
import hydra
import torch
import os
import sys
from omegaconf import DictConfig, OmegaConf
from composer.core import Precision
from composer.utils import dist, reproducibility
from composer.algorithms import GradientClipping
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from micro_diffusion.models.utils import text_encoder_embedding_format

torch.backends.cudnn.benchmark = True  # 3-5% speedup


@hydra.main(config_path='configs', config_name='micro_res_64_test', version_base=None)
def train(cfg: DictConfig) -> None:
    """Train a micro-diffusion model using the provided configuration.

    Args:
        cfg (DictConfig): Configuration object loaded from yaml file.
    """
    try:
        if not cfg:
            raise ValueError('Config not specified. Please provide --config-path and --config-name, respectively.')
        reproducibility.seed_all(cfg['seed'])

        # Check if the test latents directory exists
        test_latents_dir = "./datadir/test_latents/train/"
        if not os.path.exists(test_latents_dir):
            raise ValueError(f"Test latents directory {test_latents_dir} does not exist. Please run generate_test_latents.py first.")

        # Force precomputed_latents to true
        cfg.model.precomputed_latents = True
        print(f"Setting precomputed_latents to {cfg.model.precomputed_latents}")
        
        # Print model configuration for debugging
        print(f"Model configuration:")
        print(f"  Architecture: {cfg.model.dit_arch}")
        print(f"  Latent resolution: {cfg.model.latent_res}")
        print(f"  In channels: {cfg.model.in_channels}")
        
        # Instantiate the model
        print("Instantiating model...")
        model = hydra.utils.instantiate(cfg.model)
        print("Model instantiated successfully")

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
        print(f"  Train data directory: {cfg.dataset.train.datadir}")
        
        print("Instantiating train loader...")
        train_loader = hydra.utils.instantiate(
            cfg.dataset.train,
            image_size=cfg.dataset.image_size,
            batch_size=cfg.dataset.train_batch_size // dist.get_world_size(),
            cap_seq_size=cap_seq_size,
            cap_emb_dim=cap_emb_dim,
            cap_drop_prob=cfg.dataset.cap_drop_prob)
        print(f"Found {len(train_loader.dataset)*dist.get_world_size()} images in the training dataset")
        time.sleep(3)

        print("Instantiating eval loader...")
        eval_loader = hydra.utils.instantiate(
            cfg.dataset.eval,
            image_size=cfg.dataset.image_size,
            batch_size=cfg.dataset.eval_batch_size // dist.get_world_size(),
            cap_seq_size=cap_seq_size,
            cap_emb_dim=cap_emb_dim)
        print(f"Found {len(eval_loader.dataset)*dist.get_world_size()} images in the eval dataset")
        time.sleep(3)

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
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    train() 