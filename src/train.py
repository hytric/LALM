import os
import sys
import argparse
import yaml
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.setup_util import instantiate_from_config, load_config
from .trainer import LALMTrainer


def setup_data_module(data_config):
    """
    Setup PyTorch Lightning DataModule from config
    
    Args:
        data_config: Data configuration dictionary
    
    Returns:
        DataModule instance
    """
    # For now, return None (will be implemented later)
    # You can create a custom DataModule class that uses instantiate_from_config
    return None


def setup_callbacks(lightning_config):
    """
    Setup callbacks from lightning config
    
    Args:
        lightning_config: Lightning configuration dictionary
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    if 'callbacks' in lightning_config:
        for callback_name, callback_config in lightning_config['callbacks'].items():
            callback = instantiate_from_config(callback_config)
            callbacks.append(callback)
    
    # Add learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    return callbacks


def setup_trainer(lightning_config, callbacks, logger, resume_from_checkpoint=None):
    """
    Setup PyTorch Lightning Trainer
    
    Args:
        lightning_config: Lightning configuration dictionary
        callbacks: List of callbacks
        logger: Logger instance
        resume_from_checkpoint: Path to checkpoint to resume from
    
    Returns:
        Trainer instance
    """
    trainer_config = lightning_config.get('trainer', {})
    
    # Extract trainer-specific arguments
    trainer_kwargs = {
        'max_epochs': trainer_config.pop('max_epochs', 100),
        'accelerator': trainer_config.pop('accelerator', 'auto'),
        'devices': trainer_config.pop('devices', 'auto'),
        'precision': trainer_config.pop('precision', '32-true'),
        'gradient_clip_val': trainer_config.pop('gradient_clip_val', None),
        'accumulate_grad_batches': trainer_config.pop('accumulate_grad_batches', 1),
        'log_every_n_steps': trainer_config.pop('log_every_n_steps', 50),
        'val_check_interval': trainer_config.pop('val_check_interval', 1.0),
        'benchmark': trainer_config.pop('benchmark', False),
        'callbacks': callbacks,
        'logger': logger,
        'resume_from_checkpoint': resume_from_checkpoint,
        # Pass remaining config as kwargs
        **trainer_config,
    }
    
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


def main(args):
    config = load_config(args.config)
    
    # Setup logger
    experiment_name = args.experiment_name or Path(args.config).stem
    log_dir = os.path.join(args.save_dir, experiment_name)
    
    if args.logger == 'wandb':
        logger = WandbLogger(name=experiment_name, save_dir=log_dir)
    else:
        logger = TensorBoardLogger(save_dir=args.save_dir, name=experiment_name)
    
    # Setup model
    model_config = config['model']
    learning_rate = model_config.get('base_learning_rate', 1e-4)
    scale_lr = model_config.get('scale_lr', False)
    
    # Create trainer module
    lightning_module = LALMTrainer(
        model_config=model_config,
        learning_rate=learning_rate,
        scale_lr=scale_lr,
    )
    
    # Setup data module (to be implemented)
    data_config = config.get('data', {})
    # data_module = setup_data_module(data_config)
    train_dataloader = None  # TODO: implement dataloader setup
    val_dataloader = None    # TODO: implement dataloader setup
    
    # Setup callbacks
    lightning_config = config.get('lightning', {})
    callbacks = setup_callbacks(lightning_config)
    
    # Setup trainer
    trainer = setup_trainer(
        lightning_config=lightning_config,
        callbacks=callbacks,
        logger=logger,
        resume_from_checkpoint=args.resume,
    )
    
    # Train
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == '__main__':
    main()

