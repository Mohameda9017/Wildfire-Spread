from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import WildfireDataset
from src.models.logistic_baseline import LogisticRegressionBaseline
from src.training.losses import masked_bce_with_logits_loss


def train_one_epoch(model, loader, optimizer, device):
    '''
    trains the model for one full epoch
    '''
    model.train()
    total_loss = 0.0

    # loops thorugh the dataloader batch by batch
    for images,labels, valid_masks in loader:
        images = images.to(device)
        labels =labels.to(device)
        valid_masks = valid_masks.to(device)

        optimizer.zero_grad() # resets gradients

        logits =model(images)
        loss =masked_bce_with_logits_loss(logits, labels, valid_masks)

        loss.backward()
        optimizer.step() # updates the weights
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    '''
    This function evaluates the model on the validation set.
    '''
    model.eval()
    total_loss =0.0

    for images,labels, valid_masks in loader:
        images = images.to(device)
        labels = labels.to(device)
        valid_masks = valid_masks.to(device)

        logits =model(images)
        loss =masked_bce_with_logits_loss(logits, labels, valid_masks)

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    project_root = Path(__file__).resolve().parents[2]
    train_dir = project_root / "data" / "processed_pt" / "train"
    eval_dir = project_root / "data" / "processed_pt" / "eval"
    checkpoint_dir = project_root / "outputs" / "checkpoints" # creates the path where the model will be saved at 
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # use mps else cpu 
    print("Using device:", device)

    train_dataset = WildfireDataset(root_dir=str(train_dir),clip_and_normalize=True,)

    eval_dataset = WildfireDataset(root_dir=str(eval_dir),clip_and_normalize=True,)

    # loads all the data into batches of 16 
    train_loader = DataLoader( train_dataset,batch_size=16,shuffle=True,num_workers=0,)

    eval_loader = DataLoader(eval_dataset,batch_size=16,shuffle=False,num_workers=0,)

    model = LogisticRegressionBaseline(in_channels=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # is what updates the weights

    num_epochs = 10
    best_eval_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device) # runs it on one. epoch, caluates training loss and updates weights
        eval_loss = evaluate(model, eval_loader, device)

        print(f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.6f} | eval_loss={eval_loss:.6f}")

        # if the eval_loss is better then the best one then save the model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_path = checkpoint_dir / "logistic_baseline_best.pt"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_loss": eval_loss,
                },
                save_path,)
            print(f"Saved best model to: {save_path}")

    print("Training finished.")
    print("Best eval loss:", best_eval_loss)


if __name__ == "__main__":
    main()