import torch
import torch.nn as nn


class LogisticRegressionBaseline(nn.Module):
    """
    Pixel-wise logistic regression for wildfire prediction.

    Input:
        x -> [B, 12, H, W]
    Output:
        logits -> [B, 1, H, W], just the net inputs

    A 1x1 convolution looks at one pixel location at a time and combines
    the 12 channel values there using learned weights + a bias.
    That is exactly pixel wise logistic regression.
    """

    def __init__(self, in_channels: int = 12):
        super().__init__()

        # One weight per input channel, plus one bias, applied at every pixel.
        self.linear = nn.Conv2d(
            in_channels=in_channels, # takes 12 numbers at each location
            out_channels=1, # predicts 1 number fot that pixel
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function gets called when you pass an input through the model
        """
        logits = self.linear(x) # compute net input of each pixel 
        return logits
    
