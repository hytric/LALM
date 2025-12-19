import torch
import torch.nn as nn
from .model.audio_encoder import BaseAudioEncoder
from .model.adapter import MultiModalProjector
from .model.LLM import LLMDecoder


class LALMModel(nn.Module):
    """
    LALM 통합 모델
    Audio Encoder + Adapter + LLM Decoder를 결합한 멀티모달 모델
    
    데이터 흐름:
    [오디오] -> [Audio Encoder] -> [Adapter] -> [LLM Decoder] -> [텍스트]
    """
    
    def __init__(
        self,
        audio_encoder: BaseAudioEncoder,
        llm_decoder: LLMDecoder,
        projector_hidden_act="gelu",
        projector_bias=False,
    ):
        """
        Args:
            audio_encoder: Audio Encoder 모델 (WhisperAudioEncoder, HubertAudioEncoder, Wav2Vec2AudioEncoder)
            llm_decoder: LLM Decoder 모델
            projector_hidden_act: Adapter의 activation 함수 ("gelu" or "relu")
            projector_bias: Adapter에 bias 사용 여부
        """
        super().__init__()
        
        self.audio_encoder = audio_encoder
        self.llm_decoder = llm_decoder
        
        # Audio encoder의 hidden size와 LLM의 hidden size 가져오기
        audio_hidden_size = audio_encoder.get_hidden_size()
        text_hidden_size = llm_decoder.get_hidden_size()
        
        # Adapter 생성
        self.adapter = MultiModalProjector(
            audio_hidden_size=audio_hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=projector_hidden_act,
            bias=projector_bias,
        )
    
    def forward(
        self,
        audio_input,
        input_ids=None,
        attention_mask=None,
        labels=None,
        audio_attention_mask=None,
        **kwargs
    ):
        """
        Forward pass
        
        Args:
            audio_input: 오디오 입력
                - Whisper: (batch_size, n_mels, seq_len)
                - HuBERT/Wav2Vec2: (batch_size, seq_len)
            input_ids: 텍스트 입력 토큰 ID (optional, instruction 등)
                Shape: (batch_size, seq_len)
            attention_mask: 텍스트 입력의 attention mask (optional)
                Shape: (batch_size, seq_len)
            labels: 타겟 텍스트 토큰 ID (training 시 사용)
                Shape: (batch_size, seq_len)
            audio_attention_mask: 오디오 입력의 attention mask (optional)
                Shape: (batch_size, audio_seq_len)
            **kwargs: LLM forward에 전달할 추가 인자
        
        Returns:
            outputs: LLMDecoder의 출력
                - logits: (batch_size, seq_len, vocab_size)
                - loss: (training 시 labels 제공하면 계산됨)
        """
        # 1. Audio Encoder 통과
        audio_hidden_states = self.audio_encoder(audio_input)
        
        # 마지막 layer의 출력 사용 (최종 representation)
        # Layer 지정 가능 (default: -1)
        if isinstance(audio_hidden_states, tuple):
            audio_features = audio_hidden_states[-1]  # (batch_size, audio_seq_len, audio_hidden_size)
        else:
            audio_features = audio_hidden_states
        
        # 2. Adapter를 통한 차원 변환
        projected_audio = self.adapter(audio_features)  # (batch_size, audio_seq_len, text_hidden_size)
        
        # 3. 텍스트 입력과 오디오 특징 결합
        if input_ids is not None:
            # 텍스트 입력이 있는 경우: 텍스트를 embedding하고 오디오 특징과 결합
            text_embeddings = self.llm_decoder.get_input_embeddings()(input_ids)
            
            # 오디오 특징과 텍스트 embedding 결합
            inputs_embeds = torch.cat([text_embeddings, projected_audio], dim=1)
            
            # Attention mask 결합
            if attention_mask is not None and audio_attention_mask is not None:
                combined_attention_mask = torch.cat([attention_mask, audio_attention_mask], dim=1)
            elif attention_mask is not None:
                # 텍스트만 있는 경우
                audio_mask = torch.ones(
                    (audio_input.shape[0], projected_audio.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                combined_attention_mask = torch.cat([attention_mask, audio_mask], dim=1)
            elif audio_attention_mask is not None:
                # 오디오만 있는 경우
                text_mask = torch.ones(
                    (audio_input.shape[0], input_ids.shape[1]),
                    dtype=audio_attention_mask.dtype,
                    device=audio_attention_mask.device
                )
                combined_attention_mask = torch.cat([text_mask, audio_attention_mask], dim=1)
            else:
                combined_attention_mask = None
            
            # Labels 조정 (텍스트 부분만 loss 계산)
            if labels is not None:
                # 오디오 부분은 -100으로 설정하여 loss 계산에서 제외
                audio_labels = torch.full(
                    (labels.shape[0], projected_audio.shape[1]),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                combined_labels = torch.cat([labels, audio_labels], dim=1)
            else:
                combined_labels = None
            
            # LLM forward
            outputs = self.llm_decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention_mask,
                labels=combined_labels,
                **kwargs
            )
        else:
            # 텍스트 입력이 없는 경우: 오디오 특징만 사용
            if audio_attention_mask is not None:
                attention_mask = audio_attention_mask
            else:
                attention_mask = None
            
            # Labels 조정
            if labels is not None:
                # 오디오만 있는 경우 labels는 그대로 사용
                combined_labels = labels
            else:
                combined_labels = None
            
            # LLM forward
            outputs = self.llm_decoder(
                inputs_embeds=projected_audio,
                attention_mask=attention_mask,
                labels=combined_labels,
                **kwargs
            )
        
        return outputs
    
    def generate(
        self,
        audio_input,
        input_ids=None,
        attention_mask=None,
        audio_attention_mask=None,
        **generation_kwargs
    ):
        """
        Text generation
        
        Args:
            audio_input: 오디오 입력
            input_ids: 텍스트 입력 토큰 ID (optional, prompt 등)
            attention_mask: 텍스트 입력의 attention mask (optional)
            audio_attention_mask: 오디오 입력의 attention mask (optional)
            **generation_kwargs: generate 메서드에 전달할 추가 인자
                - max_new_tokens: 생성할 최대 토큰 수
                - do_sample: 샘플링 사용 여부
                - temperature: 샘플링 온도
        
        Returns:
            generated_ids: 생성된 토큰 ID
                Shape: (batch_size, generated_seq_len)
        """
        self.eval()
        
        # 1. Audio Encoder 통과
        audio_hidden_states = self.audio_encoder(audio_input)
        
        if isinstance(audio_hidden_states, tuple):
            audio_features = audio_hidden_states[-1]
        else:
            audio_features = audio_hidden_states
        
        # 2. Adapter를 통한 차원 변환
        projected_audio = self.adapter(audio_features)
        
        # 3. 텍스트 입력과 오디오 특징 결합
        if input_ids is not None:
            text_embeddings = self.llm_decoder.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([text_embeddings, projected_audio], dim=1)
            
            if attention_mask is not None and audio_attention_mask is not None:
                combined_attention_mask = torch.cat([attention_mask, audio_attention_mask], dim=1)
            elif attention_mask is not None:
                audio_mask = torch.ones(
                    (audio_input.shape[0], projected_audio.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                combined_attention_mask = torch.cat([attention_mask, audio_mask], dim=1)
            else:
                combined_attention_mask = None
        else:
            inputs_embeds = projected_audio
            combined_attention_mask = audio_attention_mask
        
        # 4. Generation
        with torch.no_grad():
            generated_ids = self.llm_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_attention_mask,
                **generation_kwargs
            )
        
        return generated_ids
    
    def get_audio_encoder(self):
        """Audio Encoder 반환"""
        return self.audio_encoder
    
    def get_llm_decoder(self):
        """LLM Decoder 반환"""
        return self.llm_decoder
    
    def get_adapter(self):
        """Adapter 반환"""
        return self.adapter

