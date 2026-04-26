from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import WildfireDataset

def main():
    project_root = Path(__file__).resolve().parents[2]
    train_dir = project_root / "data" / "processed_pt" / "train"

    dataset = WildfireDataset(
        root_dir=str(train_dir),
        clip_and_normalize=True
    )

    print("dataset size:", len(dataset))

    image, label, valid_mask = dataset[0] # allipes clip and normalizing to the 0th sample
    print("single sample shapes:")
    print("image:", image.shape)
    print("label:", label.shape)
    print("valid_mask:", valid_mask.shape)

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    images, labels, masks = next(iter(loader))

    print("\nbatch shapes:")
    print("images:", images.shape)
    print("labels:", labels.shape)
    print("masks:", masks.shape)

if __name__ == "__main__":
    main()