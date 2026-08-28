"""LSTM decoder with Bahdanau (additive) attention over the encoder's spatial
image features. At each timestep, the decoder decides which image regions to
focus on before predicting the next word."""

import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """Additive attention: scores each spatial location by combining the
    encoder feature at that location with the decoder's current hidden state,
    through a small feed-forward network. More expressive than dot-product
    attention when encoder and decoder dimensions differ (1536 vs. decoder
    hidden size here), which is the common case in image captioning.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int = 512):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        Args:
            encoder_out: (B, num_pixels, encoder_dim)
            decoder_hidden: (B, decoder_dim)
        Returns:
            context: (B, encoder_dim) — weighted sum of encoder features
            alpha:   (B, num_pixels)  — attention weights, for visualization
        """
        att1 = self.encoder_proj(encoder_out)                # (B, num_pixels, attn_dim)
        att2 = self.decoder_proj(decoder_hidden).unsqueeze(1)  # (B, 1, attn_dim)
        att = self.full_att(self.tanh(att1 + att2)).squeeze(2)  # (B, num_pixels)
        alpha = self.softmax(att)                              # (B, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


class DecoderRNN(nn.Module):
    """LSTMCell-based decoder, unrolled manually one timestep at a time so we
    can compute attention against the encoder features at every step."""

    def __init__(self, vocab_size: int, embed_dim: int, decoder_dim: int,
                 encoder_dim: int, attention_dim: int = 512, dropout: float = 0.5,
                 pad_idx: int = 0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

        # LSTMCell input = word embedding concatenated with attention context
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # Initialize the LSTM's initial (h0, c0) from the mean-pooled image
        # features, instead of zeros — gives the decoder a head start on
        # "what the image is about" before it has attended to anything.
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # A learned gate (Show, Attend & Tell-style) that scales the attention
        # context — lets the model down-weight the image signal for function
        # words (e.g. "a", "the") where visual grounding matters less.
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out: torch.Tensor):
        mean_features = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_features)
        c = self.init_c(mean_features)
        return h, c

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor,
                lengths: torch.Tensor):
        """Teacher-forced training forward pass.

        Args:
            encoder_out: (B, num_pixels, encoder_dim)
            captions:    (B, max_len) token ids, including <start>/<end>
            lengths:     (B,) actual caption lengths (before padding)
        Returns:
            logits: (B, max_len - 1, vocab_size) — predictions for each step
            alphas: (B, max_len - 1, num_pixels) — attention weights per step
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        decode_len = captions.size(1) - 1  # predict tokens[1:] from tokens[:-1]

        embeddings = self.embedding(captions)  # (B, max_len, embed_dim)
        h, c = self.init_hidden_state(encoder_out)

        logits = encoder_out.new_zeros(batch_size, decode_len, self.vocab_size)
        alphas = encoder_out.new_zeros(batch_size, decode_len, num_pixels)

        for t in range(decode_len):
            context, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context

            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            preds = self.fc(self.dropout(h))
            logits[:, t, :] = preds
            alphas[:, t, :] = alpha

        return logits, alphas