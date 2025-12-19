import torch
import torch.nn as nn
from .model.audio_encoder import BaseAudioEncoder
from .model.adapter import MultiModalProjector
from .model.LLM import LLMDecoder


class LALMModel(nn.Module):
    """
    LALM Integrated Model
    Multi-modal model combining Audio Encoder + Adapter + LLM Decoder
    
    Data flow:
    [Audio] -> [Audio Encoder] -> [Adapter] -> [LLM Decoder] -> [Text]
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
            audio_encoder: Audio Encoder model (WhisperAudioEncoder, HubertAudioEncoder, Wav2Vec2AudioEncoder)
            llm_decoder: LLM Decoder model
            projector_hidden_act: Activation function for Adapter ("gelu" or "relu")
            projector_bias: Whether to use bias in Adapter
        """
        super().__init__()
        
        self.audio_encoder = audio_encoder
        self.llm_decoder = llm_decoder
        
        # Get hidden sizes from audio encoder and LLM
        audio_hidden_size = audio_encoder.get_hidden_size()
        text_hidden_size = llm_decoder.get_hidden_size()
        
        # Create Adapter
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
            audio_input: Audio input
                - Whisper: (batch_size, n_mels, seq_len)
                - HuBERT/Wav2Vec2: (batch_size, seq_len)
            input_ids: Text input token IDs (optional, for instruction, etc.)
                Shape: (batch_size, seq_len)
            attention_mask: Attention mask for text input (optional)
                Shape: (batch_size, seq_len)
            labels: Target text token IDs (used during training)
                Shape: (batch_size, seq_len)
            audio_attention_mask: Attention mask for audio input (optional)
                Shape: (batch_size, audio_seq_len)
            **kwargs: Additional arguments to pass to LLM forward
        
        Returns:
            outputs: Output from LLMDecoder
                - logits: (batch_size, seq_len, vocab_size)
                - loss: (computed if labels are provided during training)
        """
        # 1. Pass through Audio Encoder
        audio_hidden_states = self.audio_encoder(audio_input)
        
        # Use output from the last layer (final representation)
        # Layer can be specified (default: -1)
        if isinstance(audio_hidden_states, tuple):
            audio_features = audio_hidden_states[-1]  # (batch_size, audio_seq_len, audio_hidden_size)
        else:
            audio_features = audio_hidden_states
        
        # 2. Dimension transformation through Adapter
        projected_audio = self.adapter(audio_features)  # (batch_size, audio_seq_len, text_hidden_size)
        
        # 3. Combine text input and audio features
        if input_ids is not None:
            # If text input exists: embed text and combine with audio features
            text_embeddings = self.llm_decoder.get_input_embeddings()(input_ids)
            
            # Combine audio features and text embeddings
            inputs_embeds = torch.cat([text_embeddings, projected_audio], dim=1)
            
            # Combine attention masks
            if attention_mask is not None and audio_attention_mask is not None:
                combined_attention_mask = torch.cat([attention_mask, audio_attention_mask], dim=1)
            elif attention_mask is not None:
                # Text only
                audio_mask = torch.ones(
                    (audio_input.shape[0], projected_audio.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                combined_attention_mask = torch.cat([attention_mask, audio_mask], dim=1)
            elif audio_attention_mask is not None:
                # Audio only
                text_mask = torch.ones(
                    (audio_input.shape[0], input_ids.shape[1]),
                    dtype=audio_attention_mask.dtype,
                    device=audio_attention_mask.device
                )
                combined_attention_mask = torch.cat([text_mask, audio_attention_mask], dim=1)
            else:
                combined_attention_mask = None
            
            # Adjust labels (compute loss only for text portion)
            if labels is not None:
                # Set audio portion to -100 to exclude from loss calculation
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
            # If no text input: use audio features only
            if audio_attention_mask is not None:
                attention_mask = audio_attention_mask
            else:
                attention_mask = None
            
            # Adjust labels
            if labels is not None:
                # If audio only, use labels as is
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
            audio_input: Audio input
            input_ids: Text input token IDs (optional, for prompt, etc.)
            attention_mask: Attention mask for text input (optional)
            audio_attention_mask: Attention mask for audio input (optional)
            **generation_kwargs: Additional arguments to pass to generate method
                - max_new_tokens: Maximum number of tokens to generate
                - do_sample: Whether to use sampling
                - temperature: Sampling temperature
        
        Returns:
            generated_ids: Generated token IDs
                Shape: (batch_size, generated_seq_len)
        """
        self.eval()
        
        # 1. Pass through Audio Encoder
        audio_hidden_states = self.audio_encoder(audio_input)
        
        if isinstance(audio_hidden_states, tuple):
            audio_features = audio_hidden_states[-1]
        else:
            audio_features = audio_hidden_states
        
        # 2. Dimension transformation through Adapter
        projected_audio = self.adapter(audio_features)
        
        # 3. Combine text input and audio features
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
        """Return Audio Encoder"""
        return self.audio_encoder
    
    def get_llm_decoder(self):
        """Return LLM Decoder"""
        return self.llm_decoder
    
    def get_adapter(self):
        """Return Adapter"""
        return self.adapter

