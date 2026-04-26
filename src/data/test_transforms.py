from pathlib import Path
import torch

from src.data.transforms import RandomCropPair


def main():
    project_root = Path(__file__).resolve().parents[2]
    sample_path = project_root / "data" / "processed_pt" / "train" / "sample_0.pt"

    sample = torch.load(sample_path)

    image = sample["image"]
    label = sample["label"]
    valid_mask = sample["valid_mask"]

    print("Before crop:")
    print("image shape:", image.shape)
    print("label shape:", label.shape)
    print("valid_mask shape:", valid_mask.shape)

    transform = RandomCropPair(crop_size=32)
    cropped_image, cropped_label, cropped_valid_mask = transform(image, label, valid_mask)

    print("\nAfter crop:")
    print("image shape:", cropped_image.shape)
    print("label shape:", cropped_label.shape)
    print("valid_mask shape:", cropped_valid_mask.shape)

    # simple checks
    assert cropped_image.shape == (12, 32, 32), "Image crop shape is wrong"
    assert cropped_label.shape == (1, 32, 32), "Label crop shape is wrong"
    assert cropped_valid_mask.shape == (1, 32, 32), "Valid mask crop shape is wrong"

    print("\nCrop test passed.")


if __name__ == "__main__":
    main()