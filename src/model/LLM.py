import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


class LLMDecoder(nn.Module):
    """
    통합 LLM Decoder - 다양한 LLM 지원 (Llama, Mistral, Qwen 등)
    QLoRA/LoRA 지원
    """
    def __init__(
        self,
        model_name,
        use_qlora=True,
        use_lora=False,  # QLoRA가 아닌 일반 LoRA 사용 시
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        lora_target_modules=None,
        lora_path=None,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=None,
        trust_remote_code=True,
        **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.use_qlora = use_qlora
        self.use_lora = use_lora
        
        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Pad token 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # QLoRA를 위한 양자화 설정
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
        
        # Base model 로드
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            **kwargs
        }
        
        if use_qlora:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["low_cpu_mem_usage"] = True
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # QLoRA를 위한 모델 준비
        if use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.config.use_cache = False
        
        # LoRA/QLoRA 적용
        if use_qlora or use_lora:
            if lora_path:
                self.model = PeftModel.from_pretrained(self.model, lora_path)
            else:
                # 기본 target modules (모델 타입에 따라)
                if lora_target_modules is None:
                    model_lower = model_name.lower()
                    if "llama" in model_lower or "mistral" in model_lower:
                        lora_target_modules = [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"
                        ]
                    elif "qwen" in model_lower:
                        lora_target_modules = [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj"
                        ]
                    else:
                        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                
                # task_type은 선택적 - CausalLM의 경우 자동 감지 가능
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    # task_type은 선택적 - None이면 자동 감지
                )
                
                self.model = get_peft_model(self.model, peft_config)
                if use_qlora:
                    print(f"Applied QLoRA with r={lora_r}, alpha={lora_alpha}")
                else:
                    print(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}")
                self.model.print_trainable_parameters()
        
        # LoRA/QLoRA를 사용하지 않을 때 모든 파라미터를 frozen
        if not use_qlora and not use_lora:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None, **kwargs):
        """Forward pass through the decoder"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )
        return outputs
    
    def generate(self, input_ids=None, inputs_embeds=None, **generation_kwargs):
        """Generate text from the model"""
        if inputs_embeds is not None:
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                **generation_kwargs
            )
        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                **generation_kwargs
            )
        return outputs
    
    def get_input_embeddings(self):
        """Get input embedding layer"""
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings()
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            if hasattr(self.model.base_model.model, 'embed_tokens'):
                return self.model.base_model.model.embed_tokens
        raise AttributeError("Cannot find embedding layer in model")
    
    def get_hidden_size(self):
        """Get hidden size of the model"""
        return self.model.config.hidden_size
    
    def save_lora_weights(self, save_path):
        """Save LoRA weights only"""
        if self.use_qlora or self.use_lora:
            self.model.save_pretrained(save_path)
            print(f"Saved LoRA weights to {save_path}")
        else:
            raise ValueError("Model does not use LoRA/QLoRA")
    
    def load_lora_weights(self, lora_path):
        """Load LoRA weights"""
        if self.use_qlora or self.use_lora:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"Loaded LoRA weights from {lora_path}")
        else:
            raise ValueError("Model does not use LoRA/QLoRA")