import torch
from torch import nn

class MultiModalProjector(nn.Module):
    """
    Multi-modal projector that projects audio features to text model hidden size.
    """
    def __init__(
        self,
        audio_hidden_size,
        text_hidden_size,
        projector_hidden_act="gelu",
        bias=False

    ):
        super().__init__()
        self.linear_1 = nn.Linear(audio_hidden_size, text_hidden_size, bias=bias)
        self.act = nn.GELU() if projector_hidden_act == "gelu" else nn.ReLU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=bias)

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def set_inference_mode(self):
        """Set model to inference mode"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False