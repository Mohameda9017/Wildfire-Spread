import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple fully convolutional CNN for wildfire prediction.

    Input:
        x -> [B, 12, H, W]

    Output:
        logits -> [B, 1, H, W]

    Notes:
    - All 3x3 convolutions use padding=1 so spatial size stays the same.
    - No pooling or stride > 1, so output remains aligned with input pixels.
    - Final layer outputs logits, not probabilities.
    """

    def __init__(self, in_channels: int = 12, dropout_p: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 12, H, W]
        returns logits: [B, 1, H, W]
        """
        return self.net(x)