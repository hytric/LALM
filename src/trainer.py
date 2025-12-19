import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from utils.setup_util import instantiate_from_config
from .model.LALM import LALMModel


class LALMTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module for LALM training
    """
    
    def __init__(self, model_config, learning_rate=1e-4, scale_lr=False, inference_mode=False):
        """
        Args:
            model_config: Model configuration dictionary (will be instantiated)
            learning_rate: Learning rate
            scale_lr: Whether to scale learning rate by batch size
            inference_mode: If True, set model to inference mode (eval mode). Default is False (training mode).
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])
        
        # Instantiate model from config
        self.model: LALMModel = instantiate_from_config(model_config)
        self.learning_rate = learning_rate
        self.scale_lr = scale_lr
        self.inference_mode = inference_mode
        
        # Set inference mode only if explicitly requested
        # PyTorch Lightning automatically handles train/eval mode switching during training
        if self.inference_mode:
            self.model.set_inference_mode()
            # Will be set to eval mode when needed (e.g., during validation or test)
        
    def forward(self, audio_input, input_ids=None, attention_mask=None, labels=None, audio_attention_mask=None, **kwargs):
        """
        Forward pass
        """
        return self.model(
            audio_input=audio_input,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_attention_mask=audio_attention_mask,
            **kwargs
        )
    
    def on_train_start(self):
        """Called at the beginning of training"""
        if not self.inference_mode:
            self.train()  # Set LightningModule to train mode
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        # Lightning automatically sets model to train mode in training_step
        # No need to explicitly call train() here
        
        # Unpack batch (adjust based on your dataset's collate function)
        audio_input = batch['audio']
        input_ids = batch.get('input_ids', None)
        attention_mask = batch.get('attention_mask', None)
        labels = batch.get('labels', None)
        audio_attention_mask = batch.get('audio_attention_mask', None)
        
        # Forward pass
        outputs = self(
            audio_input=audio_input,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            audio_attention_mask=audio_attention_mask,
        )
        
        # Compute loss
        loss = outputs.loss
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_validation_start(self):
        """Called at the beginning of validation"""
        # Lightning automatically sets model to eval mode in validation
        # This hook is called for additional setup if needed
        pass
    
    def on_validation_end(self):
        """Called at the end of validation"""
        # Lightning automatically restores train mode after validation
        # This hook is called for additional cleanup if needed
        pass
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        # Eval mode for validation (handled by on_validation_start)
        # Unpack batch
        audio_input = batch['audio']
        input_ids = batch.get('input_ids', None)
        attention_mask = batch.get('attention_mask', None)
        labels = batch.get('labels', None)
        audio_attention_mask = batch.get('audio_attention_mask', None)
        
        # Forward pass with torch.no_grad() for efficiency
        with torch.no_grad():
            outputs = self(
                audio_input=audio_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                audio_attention_mask=audio_attention_mask,
            )
        
        # Compute loss
        loss = outputs.loss
        
        # Logging
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler
        """
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Scale learning rate if needed
        lr = self.learning_rate
        if self.scale_lr:
            # Scale by number of devices (if using DDP)
            lr = lr * self.trainer.num_devices
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        # Learning rate scheduler (optional)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=lr * 0.1,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def on_test_start(self):
        """Called at the beginning of test/inference"""
        # Lightning automatically sets model to eval mode in test
        # This hook is called for additional setup if needed
        pass
    
    def test_step(self, batch, batch_idx):
        """
        Test/Inference step
        """
        # Unpack batch
        audio_input = batch['audio']
        input_ids = batch.get('input_ids', None)
        attention_mask = batch.get('attention_mask', None)
        audio_attention_mask = batch.get('audio_attention_mask', None)
        
        # Forward pass with torch.no_grad() for inference
        with torch.no_grad():
            outputs = self(
                audio_input=audio_input,
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_attention_mask=audio_attention_mask,
            )
        
        return outputs
    
    def inference(self, audio_input, input_ids=None, attention_mask=None, audio_attention_mask=None, **kwargs):
        """
        Manual inference method (for external use)
        Sets model to eval mode temporarily, then restores previous state
        """
        was_training = self.training
        self.eval()  # Set LightningModule to eval mode
        
        try:
            with torch.no_grad():
                outputs = self(
                    audio_input=audio_input,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_attention_mask=audio_attention_mask,
                    **kwargs
                )
        finally:
            # Restore previous training state (unless in inference_mode)
            if not self.inference_mode and was_training:
                self.train()  # Set LightningModule back to train mode
        
        return outputs

