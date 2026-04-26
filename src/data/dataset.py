from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from src.data.transforms import RandomCropPair
from src.data.stats import INPUT_FEATURES, DATA_STATS

def clip_and_normalize_channel(channel: torch.Tensor, key: str) -> torch.Tensor:
    """
    Takes one channel (elevation, wind speed, etc) and returns the processed tensor 
    """
    min_val, max_val, mean, std = DATA_STATS[key]
    channel = torch.clamp(channel, min=min_val, max=max_val) # clips extreme values
    channel = channel - mean # centers it relative to th mean
    channel = channel / std
    return channel

def clip_and_rescale_channel(channel: torch.Tensor, key: str) -> torch.Tensor:
    """
    Instead of clipping and normalizing with mean/std, this function takes a tensor and rescales values to the range [0,1]. 
    """
    min_val, max_val, _, _ = DATA_STATS[key]
    channel = torch.clamp(channel, min=min_val, max=max_val)
    denom = max_val - min_val
    if denom == 0:
        return torch.zeros_like(channel)
    channel = (channel - min_val) / denom # maps values to [0,1]
    return channel

'''
A PyTorch Dataset is a custom class that tells pytorch:
    - how many smaples you have 
    - and how to get one sample
So it knows how to load your data. 

The Dataloader sits on top of the dataset where: 
    - it pulls samples from Dataset
    - Gorups then into batches

We will define a WildfireDataset class that is a subclass of Dataset meaning it can be passed into Dataloader.
WildfireDataset will tell pytorch: 
    - where your wildfire .pt files are
    - how to load one
    - what to return for one sample
    - how to preprocess the wildfire channels
'''

class WildfireDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        clip_and_normalize: bool = False,
        clip_and_rescale: bool = False,
        transform: Optional[callable] = None,
    ):
        """
        root_dir example:
            data/processed_pt/train

        Returns:
            image: [C, H, W]
            label: [1, H, W]
            valid_mask: [1, H, W]
        """
        if clip_and_normalize and clip_and_rescale:
            raise ValueError("Cannot have both clip_and_normalize and clip_and_rescale be True.")

        self.root_dir = Path(root_dir)
        # stores a list of path of all samples in sorted order. 
        self.files = sorted(
            self.root_dir.glob("sample_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )

        self.clip_and_normalize = clip_and_normalize
        self.clip_and_rescale = clip_and_rescale
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        image shape: [C, H, W]
        Apply preprocessing channel by channel using INPUT_FEATURES order.
        """
        processed_channels = []

        for channel_idx, key in enumerate(INPUT_FEATURES):
            channel = image[channel_idx] # extracting one channel

            if self.clip_and_normalize:
                channel = clip_and_normalize_channel(channel, key)
            elif self.clip_and_rescale:
                channel = clip_and_rescale_channel(channel, key)

            processed_channels.append(channel)
        # stacks all processed channels 
        return torch.stack(processed_channels, dim=0)

    def __getitem__(self, idx: int):
        sample = torch.load(self.files[idx])

        image = sample["image"].float()         # [12, 64, 64]
        label = sample["label"].float()         # [1, 64, 64]
        valid_mask = sample["valid_mask"].float()  # [1, 64, 64]

        image = self.preprocess_image(image)

        if self.transform is not None:
            image, label, valid_mask = self.transform(image, label, valid_mask)

        return image, label, valid_mask
    

