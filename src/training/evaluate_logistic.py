from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import WildfireDataset
from src.models.logistic_baseline import LogisticRegressionBaseline
from src.training.losses import masked_bce_with_logits_loss
from src.training.metrics import get_masked_confusion_counts


def evaluate_split(model, loader, device, threshold=0.25):
    '''
    Test how good the model is in the test or eval split 
    '''
    model.eval()

    total_loss_sum = 0.0
    total_valid_pixels = 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    # loops thorugh all the data one batch at a time
    with torch.no_grad():
        for images, labels, valid_masks in loader:
            images = images.to(device)
            labels = labels.to(device)
            valid_masks = valid_masks.to(device)

            logits = model(images)

            # masked loss returns average over valid pixels in this batch
            batch_loss = masked_bce_with_logits_loss(logits, labels, valid_masks)
            num_valid = valid_masks.sum().item()

            total_loss_sum += batch_loss.item() * num_valid
            total_valid_pixels += num_valid

            tp, fp, fn, tn = get_masked_confusion_counts(
                logits, labels, valid_masks, threshold=threshold
            )

            total_tp += tp.item()
            total_fp += fp.item()
            total_fn += fn.item()
            total_tn += tn.item()

    eps = 1e-8

    avg_loss = total_loss_sum / (total_valid_pixels + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "valid_pixels": total_valid_pixels,
    }


def print_metrics(split_name, metrics, threshold):
    print(f"\n--- {split_name.upper()} RESULTS @ threshold={threshold:.2f} ---")
    print(f"loss:      {metrics['loss']:.6f}")
    print(f"accuracy:  {metrics['accuracy']:.6f}")
    print(f"precision: {metrics['precision']:.6f}")
    print(f"recall:    {metrics['recall']:.6f}")
    print(f"f1:        {metrics['f1']:.6f}")
    print(f"iou:       {metrics['iou']:.6f}")
    print(f"tp: {int(metrics['tp'])} | fp: {int(metrics['fp'])} | fn: {int(metrics['fn'])} | tn: {int(metrics['tn'])}")
    print(f"valid_pixels: {int(metrics['valid_pixels'])}")


def main():
    project_root = Path(__file__).resolve().parents[2]

    eval_dir = project_root / "data" / "processed_pt" / "eval"
    test_dir = project_root / "data" / "processed_pt" / "test"
    checkpoint_path = project_root / "outputs" / "checkpoints" / "logistic_baseline_best.pt"

    threshold = 0.25
    batch_size = 16

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print("Loading checkpoint from:", checkpoint_path)

    eval_dataset = WildfireDataset(
        root_dir=str(eval_dir),
        clip_and_normalize=True,
    )

    test_dataset = WildfireDataset(
        root_dir=str(test_dir),
        clip_and_normalize=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = LogisticRegressionBaseline(in_channels=12).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"]) # loads the saved mocdel's weights to 'model'

    eval_metrics = evaluate_split(model, eval_loader, device, threshold=threshold)
    test_metrics = evaluate_split(model, test_loader, device, threshold=threshold)

    print_metrics("eval", eval_metrics, threshold)
    print_metrics("test", test_metrics, threshold)


if __name__ == "__main__":
    main()