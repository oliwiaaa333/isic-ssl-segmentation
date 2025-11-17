import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from pathlib import Path
from src.models.maau import MAAU
from src.data.augmentations import get_augmentations_maau
from src.training.data import SegDataset


def sanity_check():
    csv_path = Path("data/metadata/isic2018_task1_train_processed_test_hair_removal_small.csv")
    augment = get_augmentations_maau()

    dataset = SegDataset(csv_path, transform=augment)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # inicjalizacja modelu
    model = MAAU(in_channels=3, out_channels=1, final_activation="sigmoid").to(device)
    model.eval()  # tryb ewaluacji (bez dropout, BN w trybie eval)

    # próbny batch
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        print(f"[INFO] Wejście: {images.shape} | Maski: {masks.shape}")
        with torch.no_grad():
            preds = model(images)
        print(f"[INFO] Wyjście modelu: {preds.shape} | min={preds.min():.3f}, max={preds.max():.3f}")
        break


if __name__ == "__main__":
    sanity_check()
