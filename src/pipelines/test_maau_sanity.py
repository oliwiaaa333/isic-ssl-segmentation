import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from pathlib import Path
from src.models.maau import MAAU
from src.data.augmentations import get_augmentations_maau


# --- prosty dataset dla sanity-checka ---
class SmallISICDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row['image_url']))
        mask = cv2.imread(str(row['mask_url']), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Nie udało się wczytać: {row['image_url']} lub {row['mask_url']}")

        # Albumentations: augmentacja + normalizacja + konwersja do tensorów
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()  # [1, H, W]

        return img, mask


# --- główny sanity-check ---
def sanity_check():
    csv_path = Path("data/metadata/isic2018_task1_train_processed_test_hair_removal_small.csv")
    augment = get_augmentations_maau()

    dataset = SmallISICDataset(csv_path, transform=augment)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # inicjalizacja modelu
    model = MAAU(in_channels=3, out_channels=1, final_activation="sigmoid").to(device)
    model.eval()  # tryb ewaluacji (bez dropout, BN w trybie eval)

    # próbny batch
    for images, masks in loader:
        print(f"[INFO] Wejście: {images.shape} | Maski: {masks.shape}")
        with torch.no_grad():
            preds = model(images)
        print(f"[INFO] Wyjście modelu: {preds.shape} | min={preds.min():.3f}, max={preds.max():.3f}")
        break  # wystarczy 1 batch


if __name__ == "__main__":
    sanity_check()
