import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.utils.common_utils import instantiate_from_config
from .trainer import LALMTrainer

# Hydra DictConfig 지원
try:
    from omegaconf import DictConfig, OmegaConf
    HAS_HYDRA = True
except ImportError:
    HAS_HYDRA = False
    DictConfig = dict


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


def _convert_to_dict(cfg):
    """Convert DictConfig to dict if needed"""
    if HAS_HYDRA and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg

def main(cfg: DictConfig):
    """
    Main training function using Hydra config.
    
    Args:
        cfg: Hydra DictConfig or dict containing all configuration
    """
    # Convert to dict for compatibility
    config = _convert_to_dict(cfg)
    
    # Setup logger
    experiment_config = config.get('experiment', {})
    experiment_name = experiment_config.get('name')
    if experiment_name is None:
        # Use dataset name if experiment name not specified
        dataset_name = config.get('dataset', {}).get('target', 'unknown')
        if isinstance(dataset_name, str):
            experiment_name = dataset_name.split('.')[-1].replace('Dataset', '')
        else:
            experiment_name = 'experiment'
    
    save_dir = experiment_config.get('save_dir', './logs')
    logger_type = experiment_config.get('logger', 'tensorboard')
    log_dir = os.path.join(save_dir, experiment_name)
    
    if logger_type == 'wandb':
        logger = WandbLogger(name=experiment_name, save_dir=log_dir)
    else:
        logger = TensorBoardLogger(save_dir=save_dir, name=experiment_name)
    
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
    
    # Setup callbacks and trainer
    trainer_config = config.get('trainer', {})
    callbacks = setup_callbacks(trainer_config)
    
    # Get resume checkpoint from trainer config
    trainer_params = trainer_config.get('trainer', {})
    resume_from_checkpoint = trainer_params.get('resume_from_checkpoint')
    
    # Setup trainer
    trainer = setup_trainer(
        lightning_config=trainer_config,
        callbacks=callbacks,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    # Train
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == '__main__':
    main()

