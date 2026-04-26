import torch


def apply_sigmoid_and_threshold(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert logits into binary predictions.

    Args:
        logits: [B, 1, H, W]
        threshold: probability cutoff

    Returns:
        preds: [B, 1, H, W] with values 0 or 1
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return preds


def get_masked_confusion_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Compute TP, FP, FN, TN using only valid pixels.

    Args:
        logits: [B, 1, H, W]
        targets: [B, 1, H, W]
        valid_mask: [B, 1, H, W]
        threshold: probability threshold for predicting fire

    Returns:
        tp, fp, fn, tn as scalar tensors
    """
    preds = apply_sigmoid_and_threshold(logits, threshold)

    # Keep only valid pixels
    preds = preds[valid_mask == 1]
    targets = targets[valid_mask == 1]

    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    tn = ((preds == 0) & (targets == 0)).sum().float()

    return tp, fp, fn, tn


def masked_precision(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    tp, fp, _, _ = get_masked_confusion_counts(logits, targets, valid_mask, threshold)
    return tp / (tp + fp + eps)


def masked_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    tp, _, fn, _ = get_masked_confusion_counts(logits, targets, valid_mask, threshold)
    return tp / (tp + fn + eps)


def masked_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    precision = masked_precision(logits, targets, valid_mask, threshold, eps)
    recall = masked_recall(logits, targets, valid_mask, threshold, eps)
    return 2 * precision * recall / (precision + recall + eps)


def masked_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    tp, fp, fn, _ = get_masked_confusion_counts(logits, targets, valid_mask, threshold)
    return tp / (tp + fp + fn + eps)