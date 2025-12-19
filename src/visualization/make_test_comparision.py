import argparse
from pathlib import Path
import yaml
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd

from src.models.maau import MAAU
from src.training.data import make_loader_eval
from src.visualization.visuals import overlay_mask_on_image


@torch.no_grad()
def load_model(cfg, checkpoint_path, device):
    model = MAAU(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        final_activation=None,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def make_test_comparison(
        cfg_path,
        checkpoints: dict,
        out_dir,
        samples_list=None,
        thr=0.5,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"])

    test_df = pd.read_csv(cfg["data"]["test_csv"]).reset_index(drop=True)

    test_tf = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_dl = make_loader_eval(
        cfg["data"]["test_csv"],
        test_tf,
        num_workers=0,
        shuffle=False,
    )

    selected = None
    if samples_list is not None:
        with open(samples_list) as f:
            selected = set(line.strip() for line in f if line.strip())

    models = {
        name: load_model(cfg, ckpt, device)
        for name, ckpt in checkpoints.items()
    }

    for idx, ((x, y), row) in enumerate(zip(test_dl, test_df.itertuples())):
        img_path = Path(row.image_path)

        if selected and str(img_path) not in selected:
            continue

        x = x.to(device)
        y = y[0, 0].cpu().numpy() * 255

        # de-normalize image
        image = x[0].cpu().numpy().transpose(1, 2, 0)
        image = ((image * np.array([0.229, 0.224, 0.225]))
                 + np.array([0.485, 0.456, 0.406]))
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        panels = [image, y.astype(np.uint8)]

        for model in models.values():
            probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            pred = (probs >= thr).astype(np.uint8) * 255
            overlay = overlay_mask_on_image(image, pred)
            panels.append(overlay)

        row_img = np.concatenate(panels, axis=1)
        out_path = out_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), row_img)

        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--samples_list", default=None)
    p.add_argument("--thr", type=float, default=0.5)

    p.add_argument("--baseline_ckpt", required=True)
    p.add_argument("--noisy_ckpt", required=True)
    p.add_argument("--dumm_ckpt", required=True)

    args = p.parse_args()

    make_test_comparison(
        cfg_path=args.config,
        checkpoints={
            "baseline": args.baseline_ckpt,
            "noisy_student": args.noisy_ckpt,
            "dumm": args.dumm_ckpt,
        },
        out_dir=args.out_dir,
        samples_list=args.samples_list,
        thr=args.thr,
    )
