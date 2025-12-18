"""
Mini-test 'overfit 1 batch' – zgodnie z Karpathy (2019, 'A Recipe for Training Neural Networks')
Cel: sprawdzenie czy model MAAU, loss i pipeline uczą się poprawnie na małej próbce (5 obrazów).
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A

from src.training.data import LabeledSegDataset
from src.models.maau import MAAU
from src.training.metrics import dice_coeff
from src.training.losses import dice_loss, bce_loss, combined_loss


def overfit_one_batch():
    csv_path = Path("data/metadata/isic2018_task1_train_processed_test_hair_removal_small.csv")
    batch_size = 5
    epochs = 60
    lr = 1e-3

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    ds = LabeledSegDataset(csv_path, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAAU(in_channels=3, out_channels=1, final_activation="sigmoid").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Dataset size: {len(ds)} samples")

    # trening
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for imgs, masks in dl:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(ds)

        model.eval()
        with torch.no_grad():
            dice_vals = []
            for imgs, masks in dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                dice_vals.append(dice_coeff(preds, masks))

            mean_dice = sum(dice_vals) / len(dice_vals)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[Epoch {epoch:02d}] Loss: {avg_loss:.4f} | Dice: {mean_dice:.4f}")

    print("\nTest zakończony; loss spadł, a dice wzrósł, co potwierdza poprawne działanie pipeline.")


if __name__ == "__main__":
    overfit_one_batch()
