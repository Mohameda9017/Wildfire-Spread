from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import WildfireDataset
from src.models.simple_cnn import SimpleCNN
from src.training.losses import masked_weighted_bce_with_logits_loss
from src.training.metrics import get_masked_confusion_counts


def evaluate_at_threshold(model, loader, device, threshold):
    model.eval()

    total_loss_sum = 0.0
    total_valid_pixels = 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    with torch.no_grad():
        for images, labels, valid_masks in loader:
            images = images.to(device)
            labels = labels.to(device)
            valid_masks = valid_masks.to(device)

            logits = model(images)

            batch_loss = masked_weighted_bce_with_logits_loss(
            logits,
            labels,
            valid_masks,
            pos_weight=5.0,
            neg_weight=1.0,
            )
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

    loss = total_loss_sum / (total_valid_pixels + eps)
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = total_tp / (total_tp + total_fp + total_fn + eps)

    return {
        "threshold": threshold,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


def main():
    project_root = Path(__file__).resolve().parents[2]

    eval_dir = project_root / "data" / "processed_pt" / "eval"
    checkpoint_path = project_root / "outputs" / "checkpoints" / "simple_cnn_weighted_crop_best.pt"

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

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = SimpleCNN(in_channels=12, dropout_p=0.1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    thresholds = [i / 100 for i in range(10, 95, 5)]

    best_result = None

    print("\nThreshold sweep on eval set")
    print("-" * 95)
    print(f"{'thr':<8}{'loss':<12}{'acc':<12}{'prec':<12}{'recall':<12}{'f1':<12}{'iou':<12}")
    print("-" * 95)

    for threshold in thresholds:
        metrics = evaluate_at_threshold(model, eval_loader, device, threshold)

        print(
            f"{metrics['threshold']:<8.2f}"
            f"{metrics['loss']:<12.6f}"
            f"{metrics['accuracy']:<12.6f}"
            f"{metrics['precision']:<12.6f}"
            f"{metrics['recall']:<12.6f}"
            f"{metrics['f1']:<12.6f}"
            f"{metrics['iou']:<12.6f}"
        )

        if best_result is None or metrics["f1"] > best_result["f1"]:
            best_result = metrics

    print("-" * 95)
    print("\nBest threshold by eval F1:")
    print(f"threshold: {best_result['threshold']:.2f}")
    print(f"loss:      {best_result['loss']:.6f}")
    print(f"accuracy:  {best_result['accuracy']:.6f}")
    print(f"precision: {best_result['precision']:.6f}")
    print(f"recall:    {best_result['recall']:.6f}")
    print(f"f1:        {best_result['f1']:.6f}")
    print(f"iou:       {best_result['iou']:.6f}")
    print(
        f"tp: {int(best_result['tp'])} | "
        f"fp: {int(best_result['fp'])} | "
        f"fn: {int(best_result['fn'])} | "
        f"tn: {int(best_result['tn'])}"
    )


if __name__ == "__main__":
    main()