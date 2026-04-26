from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.dataset import WildfireDataset
from src.models.logistic_baseline import LogisticRegressionBaseline
from src.training.losses import masked_bce_with_logits_loss


def main():
    project_root = Path(__file__).resolve().parents[2]
    train_dir = project_root / "data" / "processed_pt" / "train"

    dataset = WildfireDataset(
        root_dir=str(train_dir),
        clip_and_normalize=True
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    images, labels, valid_masks = next(iter(loader))

    print("images shape:", images.shape)
    print("labels shape:", labels.shape)
    print("valid_masks shape:", valid_masks.shape)

    model = LogisticRegressionBaseline(in_channels=12)

    logits = model(images)

    print("logits shape:", logits.shape)

    loss = masked_bce_with_logits_loss(logits, labels, valid_masks)

    print("loss:", loss.item())


if __name__ == "__main__":
    main()