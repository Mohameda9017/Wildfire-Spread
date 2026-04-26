import torch
import torch.nn.functional as F


def masked_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    BCE with logits, averaged only over valid pixels.
    """
    per_pixel_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    masked_loss = per_pixel_loss * valid_mask
    num_valid = valid_mask.sum()

    if num_valid == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return masked_loss.sum() / num_valid


def masked_weighted_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: float = 5.0,
    neg_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted BCE with logits, averaged only over valid pixels.

    Args:
        logits:     [B, 1, H, W]
        targets:    [B, 1, H, W] with values 0 or 1
        valid_mask: [B, 1, H, W] with values 0 or 1
        pos_weight: weight for fire pixels (target == 1)
        neg_weight: weight for non-fire pixels (target == 0)
    """
    per_pixel_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    class_weights = torch.where(
        targets == 1,
        torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype),
        torch.tensor(neg_weight, device=logits.device, dtype=logits.dtype),
    )

    weighted_loss = per_pixel_loss * class_weights
    masked_weighted_loss = weighted_loss * valid_mask

    num_valid = valid_mask.sum()

    if num_valid == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return masked_weighted_loss.sum() / num_valid

def main():
    # small test 
    logits = torch.randn(2, 1, 4, 4)
    targets = torch.randint(0, 2, (2, 1, 4, 4)).float()
    valid_mask = torch.ones(2, 1, 4, 4)

    loss = masked_bce_with_logits_loss(logits, targets, valid_mask)
    print(loss)
    print(loss.shape)


if __name__ == "__main__":
    main()