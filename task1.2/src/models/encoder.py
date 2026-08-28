"""CNN encoder: extracts spatial image features from a pretrained EfficientNet
for use with an attention-based decoder."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class EncoderCNN(nn.Module):
    """Pretrained EfficientNet-B3 backbone, with the classification head removed.

    Instead of pooling down to a single feature vector, we keep a *spatial* grid
    of features (via adaptive pooling to a fixed size), so the decoder's attention
    mechanism can later learn to look at different regions of the image for
    different words — e.g. focus on the dog while generating "dog", then on the
    ball while generating "ball".
    """

    def __init__(self, encoded_size: int = 7, fine_tune_from_block: int = 6):
        """
        Args:
            encoded_size: output feature grid is (encoded_size x encoded_size).
            fine_tune_from_block: EfficientNet-B3 has 9 top-level feature blocks
                (indices 0-8). Blocks before this index stay frozen (transfer
                learning); blocks from this index onward are fine-tuned. Set to
                9 to freeze the whole backbone.
        """
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        backbone = efficientnet_b3(weights=weights)

        # `.features` is the convolutional trunk (no avgpool / classifier) —
        # output is a (B, 1536, H, W) spatial feature map.
        self.backbone = backbone.features
        self.encoder_dim = 1536  # B3's final channel count (B0 would be 1280)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))
        self._set_fine_tuning(fine_tune_from_block)

    def _set_fine_tuning(self, from_block: int) -> None:
        """Freeze early blocks (generic features: edges, textures, colors — no
        need to relearn them on only 8k images) and leave later blocks trainable
        (higher-level features benefit from adapting to our domain)."""
        for idx, block in enumerate(self.backbone):
            requires_grad = idx >= from_block
            for param in block.parameters():
                param.requires_grad = requires_grad

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)  — note: B3 expects 300x300 input for best results
        Returns:
            features: (B, num_pixels, encoder_dim) — e.g. (B, 49, 1536) for a 7x7 grid
        """
        features = self.backbone(images)          # (B, 1536, H', W')
        features = self.adaptive_pool(features)    # (B, 1536, 7, 7)
        features = features.permute(0, 2, 3, 1)    # (B, 7, 7, 1536)
        features = features.flatten(1, 2)           # (B, 49, 1536)
        return features