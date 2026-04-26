import torch
import random


class RandomCropPair:
    """
    Randomly crop image, label, and valid_mask together so they stay aligned.

    Input shapes:
        image:      [C, H, W]
        label:      [1, H, W]
        valid_mask: [1, H, W]
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, image: torch.Tensor, label: torch.Tensor, valid_mask: torch.Tensor):
        _, h, w = image.shape
        crop_h = self.crop_size
        crop_w = self.crop_size

        if crop_h > h or crop_w > w:
            raise ValueError(
                f"Crop size {self.crop_size} is larger than input size {(h, w)}"
            )

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        image = image[:, top:top + crop_h, left:left + crop_w]
        label = label[:, top:top + crop_h, left:left + crop_w]
        valid_mask = valid_mask[:, top:top + crop_h, left:left + crop_w]

        return image, label, valid_mask