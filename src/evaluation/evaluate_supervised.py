import torch
import yaml
from pathlib import Path
import argparse
import pandas as pd
import json

from src.models.maau import MAAU
from src.training.data import make_loader_eval
from src.training.metrics import dice_coeff, iou_score, precision_score, recall_score, specificity_score

import albumentations as A
from albumentations.pytorch import ToTensorV2


@torch.no_grad()
def evaluate_supervised(cfg_path: str, checkpoint_path: str):

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"])
    root = Path(cfg["paths"]["root"])

    test_tf = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_dl = make_loader_eval(root / cfg["data"]["test_csv"], test_tf, num_workers=cfg["data"]["num_workers"])

    model = MAAU(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        final_activation=cfg["model"]["final_activation"]
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    thr = float(cfg["metrics"]["threshold"])
    eps = float(cfg["metrics"]["eps"])

    dices, ious, precs, recs, specs = [], [], [], [], []

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        dices.append(dice_coeff(pred, y, thr=thr, eps=eps))
        ious.append(iou_score(pred, y, thr=thr, eps=eps))
        precs.append(precision_score(pred, y, thr=thr, eps=eps))
        recs.append(recall_score(pred, y, thr=thr, eps=eps))
        specs.append(specificity_score(pred, y, thr=thr, eps=eps))

    metrics = {
        "method": "supervised",
        "dice": float(torch.tensor(dices).mean()),
        "iou": float(torch.tensor(ious).mean()),
        "precision": float(torch.tensor(precs).mean()),
        "recall": float(torch.tensor(recs).mean()),
        "specificity": float(torch.tensor(specs).mean()),
    }

    exp_dir = Path(checkpoint_path).parents[1]
    test_dir = exp_dir / "test"
    test_dir.mkdir(exist_ok=True)

    with open(test_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(test_dir / "metrics.csv", index=False)

    print("[INFO] Metryki dla supervised zapisane w: ", test_dir)
    for k,v in metrics.items():
        if k != "method":
            print(f"{k}: {v:.4f}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    evaluate_supervised(args.config, args.checkpoint)