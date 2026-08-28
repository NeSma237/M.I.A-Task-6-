"""Combines the CNN encoder and attention-LSTM decoder into a single trainable
model."""

import torch
import torch.nn as nn

from src.models.encoder import EncoderCNN
from src.models.decoder import DecoderRNN


class CaptionModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, decoder_dim: int = 512,
                 attention_dim: int = 512, encoded_size: int = 7,
                 fine_tune_from_block: int = 6, dropout: float = 0.5,
                 pad_idx: int = 0):
        super().__init__()
        self.encoder = EncoderCNN(
            encoded_size=encoded_size,
            fine_tune_from_block=fine_tune_from_block,
        )
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            pad_idx=pad_idx,
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor,
                lengths: torch.Tensor):
        encoder_out = self.encoder(images)
        logits, alphas = self.decoder(encoder_out, captions, lengths)
        return logits, alphas

    def trainable_parameters(self):
        """Only parameters with requires_grad=True — i.e. the decoder plus the
        unfrozen tail of the encoder. Pass this to the optimizer instead of
        model.parameters() so frozen encoder blocks aren't wastefully tracked."""
        return filter(lambda p: p.requires_grad, self.parameters())