import torch
from torch import nn
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import src.utils.common_utils as common_utils
import os


class BaseAudioEncoder(nn.Module):
    """Base class for audio encoders"""
    def forward(self, x):
        """
        Forward pass
        
        Returns:
            hidden_states: Tuple of all layer outputs
                - Each element: (batch_size, seq_len, hidden_size)
                - Last element is the final layer output
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_hidden_size(self):
        """Get hidden size"""
        raise NotImplementedError("Subclasses must implement get_hidden_size method")
    
    def save_lora_weights(self, save_path):
        """
        Save LoRA weights only (adapter weights)
        PeftModel.save_pretrained() saves only adapter weights, not full model weights
        
        Args:
            save_path: Path to save the LoRA weights
        """
        use_lora = (hasattr(self, 'use_lora') and self.use_lora) or (hasattr(self, 'use_qlora') and self.use_qlora)
        if use_lora:
            if not isinstance(self.encoder, PeftModel):
                raise ValueError("encoder is not a PeftModel. LoRA weights cannot be saved separately.")
            os.makedirs(save_path, exist_ok=True)
            # PeftModel.save_pretrained() saves only adapter weights (LoRA weights)
            self.encoder.save_pretrained(save_path)
            print(f"Saved LoRA weights to {save_path}")
        else:
            raise ValueError("Model does not use LoRA/QLoRA")
    
    def load_lora_weights(self, lora_path):
        """
        Load LoRA weights
        
        Args:
            lora_path: Path to load the LoRA weights from
        """
        use_lora = (hasattr(self, 'use_lora') and self.use_lora) or (hasattr(self, 'use_qlora') and self.use_qlora)
        if use_lora:
            self.encoder = PeftModel.from_pretrained(self.encoder, lora_path)
            print(f"Loaded LoRA weights from {lora_path}")
        else:
            raise ValueError("Model does not use LoRA/QLoRA")
    
    def save_model_weights(self, save_path):
        """
        Save full model weights
        
        Args:
            save_path: Path to save the model weights
        """
        os.makedirs(save_path, exist_ok=True)
        # PeftModel인 경우: merge 후 전체 모델 저장
        if isinstance(self.encoder, PeftModel):
            # 원본 모델을 보존하기 위해 merge한 모델을 새로 생성
            merged_model = self.encoder.merge_and_unload()
            merged_model.save_pretrained(save_path)
            print(f"Saved full model weights (merged LoRA) to {save_path}")
        else:
            # 일반 모델인 경우: 그대로 저장
            self.encoder.save_pretrained(save_path)
            print(f"Saved model weights to {save_path}")
    
    def load_model_weights(self, load_path):
        """
        Load full model weights
        
        Args:
            load_path: Path to load the model weights from
        """
        use_lora = (hasattr(self, 'use_lora') and self.use_lora) or (hasattr(self, 'use_qlora') and self.use_qlora)
        if use_lora:
            self.encoder = PeftModel.from_pretrained(self.encoder, load_path)
        else:
            if hasattr(self, 'model_name') and 'whisper' in self.model_name.lower():
                from transformers import WhisperModel
                whisper_model = WhisperModel.from_pretrained(load_path, trust_remote_code=True)
                self.encoder = whisper_model.encoder
                self.encoder.config.output_hidden_states = True
            else:
                self.encoder = AutoModel.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    output_hidden_states=True
                )
        print(f"Loaded model weights from {load_path}")
    
    def set_inference_mode(self):
        """Set model to inference mode (eval mode, disable gradients)"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def set_training_mode(self):
        """Set model to training mode"""
        self.train()
        # LoRA/QLoRA 사용 시: LoRA 파라미터만 trainable
        use_lora = (hasattr(self, 'use_lora') and self.use_lora) or (hasattr(self, 'use_qlora') and self.use_qlora)
        if use_lora:
            if isinstance(self.encoder, PeftModel):
                # PEFT 모델의 경우, LoRA 파라미터만 trainable로 설정
                # PeftModel은 기본적으로 LoRA 파라미터만 trainable이지만,
                # set_inference_mode() 후 다시 활성화하기 위해 명시적으로 설정
                for name, param in self.encoder.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        # Full fine-tuning 사용 시: 모든 파라미터 trainable
        elif hasattr(self, 'finetune') and self.finetune:
            for param in self.parameters():
                param.requires_grad = True
    
    def inference(self, x):
        """
        Run inference (forward pass without gradients)
        
        Args:
            x: Input tensor
        Returns:
            hidden_states: Tuple of all layer outputs
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class WhisperAudioEncoder(BaseAudioEncoder):
    """
    Whisper Audio Encoder
    Only use Encoder, remove Decoder
    LoRA/QLoRA supported
    """
    def __init__(
        self,
        model_name,
        finetune=False,
        use_lora=False,
        use_qlora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        lora_path=None,
        model_path=None,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=None,
        trust_remote_code=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_lora = use_lora
        self.use_qlora = use_qlora
        self.finetune = finetune
        
        # QLoRA quantization setting
        quantization_config = None
        if use_qlora:
            if bnb_4bit_compute_dtype is None:
                bnb_4bit_compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            )
        
        # Load model (apply quantization setting if QLoRA)
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if use_qlora:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True
        
        # Load from pretrained model or saved weights
        load_from = model_path if model_path is not None else model_name
        
        # Load WhisperModel and use only encoder (remove decoder)
        from transformers import WhisperModel
        whisper_model = WhisperModel.from_pretrained(load_from, **model_kwargs)
        self.encoder = whisper_model.encoder
        # Enable output_hidden_states to get all layer outputs
        self.encoder.config.output_hidden_states = True
        
        # Apply LoRA/QLoRA
        default_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        if use_lora or use_qlora:
            self.encoder = _apply_lora(
                self.encoder,
                use_lora=use_lora,
                use_qlora=use_qlora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                lora_path=lora_path,
                task_type=None,
                default_target_modules=default_target_modules,
            )
        
        # Finetune setting
        if not use_lora and not use_qlora and finetune:
            for param in self.encoder.parameters():
                param.requires_grad = True
        elif not finetune and not use_lora and not use_qlora:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Audio input (batch_size, n_mels, seq_len) for Whisper
        Returns:
            hidden_states: Tuple of all layer outputs
                - Each element: (batch_size, seq_len, hidden_size)
                - Last element is the final layer output
        """
        outputs = self.encoder(x, output_hidden_states=True)
        return outputs.hidden_states
    
    def get_hidden_size(self):
        """Get hidden size of the encoder"""
        return self.encoder.config.d_model


class HubertAudioEncoder(BaseAudioEncoder):
    """
    HuBERT Audio Encoder
    Full fine-tuning only (LoRA not supported)
    """
    def __init__(
        self,
        model_name,
        finetune=False,
        model_path=None,
        trust_remote_code=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.finetune = finetune
        
        # Load from pretrained model or saved weights
        load_from = model_path if model_path is not None else model_name
        
        # Load model
        model_kwargs = {"trust_remote_code": trust_remote_code, "output_hidden_states": True}
        self.encoder = AutoModel.from_pretrained(load_from, **model_kwargs)
        
        # Finetune setting
        if finetune:
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Audio input (batch_size, seq_len) for HuBERT
        Returns:
            hidden_states: Tuple of all layer outputs
                - Each element: (batch_size, seq_len, hidden_size)
                - Last element is the final layer output
        """
        outputs = self.encoder(x, output_hidden_states=True)
        return outputs.hidden_states
    
    def get_hidden_size(self):
        """Get hidden size of the encoder"""
        return self.encoder.config.hidden_size


class Wav2Vec2AudioEncoder(BaseAudioEncoder):
    """
    Wav2Vec2 Audio Encoder
    Full fine-tuning only (LoRA not supported)
    """
    def __init__(
        self,
        model_name,
        finetune=False,
        model_path=None,
        trust_remote_code=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.finetune = finetune
        
        # Load from pretrained model or saved weights
        load_from = model_path if model_path is not None else model_name
        
        # Load model
        model_kwargs = {"trust_remote_code": trust_remote_code, "output_hidden_states": True}
        self.encoder = AutoModel.from_pretrained(load_from, **model_kwargs)
        
        # Finetune setting
        if finetune:
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Audio input (batch_size, seq_len) for Wav2Vec2
        Returns:
            hidden_states: Tuple of all layer outputs
                - Each element: (batch_size, seq_len, hidden_size)
                - Last element is the final layer output
        """
        outputs = self.encoder(x, output_hidden_states=True)
        return outputs.hidden_states
    
    def get_hidden_size(self):
        """Get hidden size of the encoder"""
        return self.encoder.config.hidden_size


