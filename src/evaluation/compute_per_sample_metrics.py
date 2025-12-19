import argparse
from pathlib import Path
import yaml
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.maau import MAAU
from src.training.data import make_loader_eval
from src.training.metrics import dice_coeff


@torch.no_grad()
def compute_per_sample_metrics(
        cfg_path: str,
        checkpoint_path: str,
        out_dir: str,
        easy_q: float = 0.85,
        hard_q: float = 0.15,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"])

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

    model = MAAU(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        final_activation=None,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    thr = float(cfg["metrics"]["threshold"])
    eps = float(cfg["metrics"]["eps"])

    dices = []

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        probs = torch.sigmoid(model(x))
        d = dice_coeff(probs, y, thr=thr, eps=eps)
        dices.append(float(d))

    df = pd.read_csv(cfg["data"]["test_csv"])
    df = df.reset_index(drop=True)
    df["dice"] = dices

    per_sample_csv = out_dir / "per_sample_metrics.csv"
    df.to_csv(per_sample_csv, index=False)

    easy_thr = df["dice"].quantile(easy_q)
    hard_thr = df["dice"].quantile(hard_q)

    easy_df = df[df["dice"] >= easy_thr]
    hard_df = df[df["dice"] <= hard_thr]

    easy_txt = out_dir / "easy_samples.txt"
    hard_txt = out_dir / "hard_samples.txt"

    easy_df["image_path"].to_csv(easy_txt, index=False, header=False)
    hard_df["image_path"].to_csv(hard_txt, index=False, header=False)

    print("[INFO] Saved per-sample metrics to:", per_sample_csv)
    print(f"[INFO] Easy samples ({len(easy_df)}) -> {easy_txt}")
    print(f"[INFO] Hard samples ({len(hard_df)}) -> {hard_txt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--easy_q", type=float, default=0.85)
    p.add_argument("--hard_q", type=float, default=0.15)
    args = p.parse_args()

    compute_per_sample_metrics(
        cfg_path=args.config,
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        easy_q=args.easy_q,
        hard_q=args.hard_q,
    )
