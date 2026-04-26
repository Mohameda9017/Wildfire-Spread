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

project_root = Path(__file__).resolve().parents[2]
sample_path = project_root / "data" / "processed_pt" / "eval" / "sample_1.pt"

sample = torch.load(sample_path)

print("Keys:", sample.keys())
print("image shape:", sample["image"].shape)
print("label shape:", sample["label"].shape)
print("valid_mask shape:", sample["valid_mask"].shape)

print("image dtype:", sample["image"].dtype)
print("label unique values:", sample["label"].unique())
print("valid_mask unique values:", sample["valid_mask"].unique())

print("num fire pixels:", (sample["label"] == 1).sum().item())
print("num valid pixels:", (sample["valid_mask"] == 1).sum().item())

