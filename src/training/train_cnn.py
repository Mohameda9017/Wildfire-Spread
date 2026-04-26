from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import WildfireDataset
from src.models.simple_cnn import SimpleCNN
from src.training.losses import masked_bce_with_logits_loss, masked_weighted_bce_with_logits_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels, valid_masks in loader:
        images = images.to(device)
        labels = labels.to(device)
        valid_masks = valid_masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = masked_weighted_bce_with_logits_loss(
            logits,
            labels,
            valid_masks,
            pos_weight=5.0,
            neg_weight=1.0,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for images, labels, valid_masks in loader:
        images = images.to(device)
        labels = labels.to(device)
        valid_masks = valid_masks.to(device)

        logits = model(images)
        loss = masked_weighted_bce_with_logits_loss(
            logits,
            labels,
            valid_masks,
            pos_weight=5.0,
            neg_weight=1.0,
        )
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    project_root = Path(__file__).resolve().parents[2]

    train_dir = project_root / "data" / "processed_pt" / "train"
    eval_dir = project_root / "data" / "processed_pt" / "eval"
    checkpoint_dir = project_root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_dataset = WildfireDataset(
        root_dir=str(train_dir),
        clip_and_normalize=True,
    )

    eval_dataset = WildfireDataset(
        root_dir=str(eval_dir),
        clip_and_normalize=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = SimpleCNN(in_channels=12, dropout_p=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_eval_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        eval_loss = evaluate(model, eval_loader, device)

        print(f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.6f} | eval_loss={eval_loss:.6f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_path = checkpoint_dir / "simple_cnn_weighted_best.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_loss": eval_loss,
                },
                save_path,
            )
            print(f"Saved best model to: {save_path}")

    print("Training finished.")
    print("Best eval loss:", best_eval_loss)


if __name__ == "__main__":
    main()