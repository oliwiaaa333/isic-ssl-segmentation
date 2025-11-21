from pathlib import Path
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class SegDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(Path(row["image_url"])))
        mask = cv2.imread(str(Path(row["mask_url"])), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Bad path: {row['image_url']} / {row['mask_url']}")
        out = self.transform(image=img, mask=mask) if self.transform else {"image": img, "mask": mask}
        x = out["image"]
        y = (out["mask"].float() / 255.0).unsqueeze(0)
        return x, y


def make_loaders(train_csv, val_csv, train_tf, val_tf, batch_size=8, num_workers=2):
    train_ds = SegDataset(train_csv, transform=train_tf)
    val_ds   = SegDataset(val_csv,   transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl

def make_loader_eval(test_csv, test_tf, num_workers=2):
    test_ds = SegDataset(test_csv, transform=test_tf)
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_dl