from pathlib import Path
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# dla nadzorowanego i czesci labeled w noisy student
class LabeledSegDataset(Dataset):
    def __init__(self, csv_path, transform, return_path=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.return_path = return_path

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(Path(row["image_path"])))
        mask = cv2.imread(str(Path(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Bad path: {row['image_path']} / {row['mask_path']}")
        out = self.transform(image=img, mask=mask) if self.transform else {"image": img, "mask": mask}
        x = out["image"]
        y = (out["mask"].float() / 255.0).unsqueeze(0)

        if self.return_path:
            return x, y, row["image_path"]
        else:
            return x, y


# dla noisy student
class UnlabeledSegDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(Path(row["image_path"])))

        if img is None:
            raise FileNotFoundError(f"Bad image path: {row['image_path']}")

        out = self.transform(image=img) if self.transform else {"image": img}

        x = out["image"]

        return x, row["image_path"]


class PseudoLabeledSegDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(str(Path(row["image_path"])))
        mask = cv2.imread(str(Path(row["pseudo_mask_path"])), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"Bad path: {row['image_path']} or {row['pseudo_mask_path']}")

        out = self.transform(image=img, mask=mask) if self.transform else {"image": img, "mask": mask}

        x = out["image"]
        y = (out["mask"].float() / 255.0).unsqueeze(0)

        return x, y, row["image_path"]


# nadzorowane
def make_loaders_supervised(train_csv, val_csv, train_tf, val_tf, batch_size=8, num_workers=2):
    train_ds = LabeledSegDataset(train_csv, transform=train_tf)
    val_ds   = LabeledSegDataset(val_csv, transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


# noisy student
def make_loader_labeled(csv_path, transform, batch_size=8, num_workers=2):
    ds = LabeledSegDataset(csv_path, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dl


def make_loader_unlabeled(csv_path, transform, batch_size=8, num_workers=2):
    ds = UnlabeledSegDataset(csv_path, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl


def make_loader_pseudo(csv_path, transform, batch_size=8, num_workers=2):
    ds = PseudoLabeledSegDataset(csv_path, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dl


# ewaluacja
def make_loader_eval(test_csv, test_tf, num_workers=2):
    test_ds = LabeledSegDataset(test_csv, transform=test_tf)
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_dl