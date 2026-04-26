from pathlib import Path
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib import colors

INPUT_FEATURES = [
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
    "PrevFireMask",
]


def main():
    parser = argparse.ArgumentParser(description="Visualize one wildfire sample")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval", "test"])
    parser.add_argument("--sample_id", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    sample_path = project_root / "data" / "processed_pt" / args.split / f"sample_{args.sample_id}.pt"

    if not sample_path.exists():
        print(f"Sample not found: {sample_path}")
        return

    sample = torch.load(sample_path)

    image = sample["image"]            # [12, 64, 64]
    label = sample["label"]            # [1, 64, 64]
    valid_mask = sample["valid_mask"]  # [1, 64, 64]

    print("Loaded:", sample_path)
    print("image shape:", image.shape)
    print("label shape:", label.shape)
    print("valid_mask shape:", valid_mask.shape)
    print("label unique values:", torch.unique(label))
    print("valid_mask unique values:", torch.unique(valid_mask))
    print("num fire pixels:", ((label == 1) & (valid_mask == 1)).sum().item())
    print("num valid pixels:", (valid_mask == 1).sum().item())

    # Convert to numpy for plotting
    image_np = image.numpy()
    label_np = label[0].numpy()
    valid_mask_np = valid_mask[0].numpy()

    # Reconstruct label display so it uses the SAME meaning/colors as PrevFireMask:
    # -1 = invalid, 0 = no fire, 1 = fire
    display_label = label_np.copy()
    display_label[valid_mask_np == 0] = -1

    # Colormap for fire masks:
    # black = invalid, silver = no fire, orangered = fire
    fire_cmap = colors.ListedColormap(["black", "silver", "orangered"])
    fire_bounds = [-1, -0.1, 0.001, 1]
    fire_norm = colors.BoundaryNorm(fire_bounds, fire_cmap.N)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Top row: continuous input channels
    axes[0, 0].imshow(image_np[0], cmap="terrain")
    axes[0, 0].set_title("Elevation")

    axes[0, 1].imshow(image_np[1], cmap="viridis")
    axes[0, 1].set_title("Wind Direction (th)")

    axes[0, 2].imshow(image_np[2], cmap="viridis")
    axes[0, 2].set_title("Wind Speed (vs)")

    axes[0, 3].imshow(image_np[8], cmap="Greens")
    axes[0, 3].set_title("NDVI")

    # Bottom row
    axes[1, 0].imshow(image_np[9], cmap="magma")
    axes[1, 0].set_title("Population")

    axes[1, 1].imshow(image_np[11], cmap=fire_cmap, norm=fire_norm)
    axes[1, 1].set_title("PrevFireMask")

    axes[1, 2].imshow(display_label, cmap=fire_cmap, norm=fire_norm)
    axes[1, 2].set_title("FireMask (Label)")

    axes[1, 3].imshow(valid_mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 3].set_title("Valid Mask")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()