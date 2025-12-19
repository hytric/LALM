import torch
from torch import nn
from transformers import AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from abc import ABC, abstractmethod


class BaseAudioEncoder(nn.Module, ABC):
    """Base class for audio encoders"""
    @abstractmethod
    def forward(self, x):
        """
        Forward pass
        
        Returns:
            hidden_states: Tuple of all layer outputs
                - Each element: (batch_size, seq_len, hidden_size)
                - Last element is the final layer output
        """
        pass
    
    @abstractmethod
    def get_hidden_size(self):
        """Get hidden size"""
        pass


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
        
        # Load WhisperModel and use only encoder (remove decoder)
        from transformers import WhisperModel
        whisper_model = WhisperModel.from_pretrained(model_name, **model_kwargs)
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
        trust_remote_code=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.finetune = finetune
        
        # Load model
        model_kwargs = {"trust_remote_code": trust_remote_code, "output_hidden_states": True}
        self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        
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
        trust_remote_code=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.finetune = finetune
        
        # Load model
        model_kwargs = {"trust_remote_code": trust_remote_code, "output_hidden_states": True}
        self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
        
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



def _apply_lora(
    model,
    use_lora=False,
    use_qlora=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules=None,
    lora_path=None,
    task_type=None,
    default_target_modules=None,
):
    """
    LoRA/QLoRA apply helper function
    """
    if not use_lora and not use_qlora:
        return model
    
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    
    if lora_target_modules is None:
        lora_target_modules = default_target_modules
    
    if lora_target_modules is None:
        raise ValueError("lora_target_modules must be specified when use_lora=True or use_qlora=True")
    
    lora_config_kwargs = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bias": "none",
        "target_modules": lora_target_modules,
    }
    if task_type is not None:
        lora_config_kwargs["task_type"] = task_type
    
    lora_config = LoraConfig(**lora_config_kwargs)
    model = get_peft_model(model, lora_config)
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
    
    return model