import argparse
from pathlib import Path

import yaml
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.models.maau import MAAU
from src.visualization.visuals import save_teacher_student_example


NUM_EXAMPLES = 3


@torch.no_grad()
def make_noisy_student_examples(experiment_dir: str, thr: float = 0.5):

    exp = Path(experiment_dir)
    if not exp.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu eksperymentu: {exp}")

    cfg_path = exp / "config_used.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Brak config_used.yaml w {exp}")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    ckpt_path = exp / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Nie ma best_model.pt w {exp}/checkpoints")

    pseudo_root = Path(cfg["data"]["pseudo_labels_root"])
    df = pd.read_csv(pseudo_root / "pseudo_labels_filtered.csv")
    df = df.dropna(subset=["image_path", "pseudo_mask_path"])

    if len(df) == 0:
        raise RuntimeError("pseudo_labels_filtered.csv jest puste.")

    df_sample = df.sample(NUM_EXAMPLES, random_state=0)

    model = MAAU(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        final_activation=None,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    eval_tf = A.Compose([
        A.Resize(cfg["data"]["image_size"], cfg["data"]["image_size"]),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    out_dir = exp / "visual_examples"
    out_dir.mkdir(exist_ok=True)

    for i, row in enumerate(df_sample.itertuples(), start=1):

        img_path = Path(row.image_path)
        teacher_mask_path = Path(row.pseudo_mask_path)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Nie mozna wczytac obrazu {img_path}")
            continue

        augmented = eval_tf(image=img)
        x = augmented["image"].unsqueeze(0).to(device)

        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0]

        prefix = f"example_{i:02d}"
        save_teacher_student_example(
            image_path=img_path,
            teacher_mask_path=teacher_mask_path,
            student_probs=probs,
            out_dir=out_dir,
            prefix=prefix,
            thr=thr
        )

        print(f"[INFO] Zapisano wizualizacje jako {prefix}_*.png w {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_dir", required=True, help="Ścieżka do katalogu eksperymentu Noisy Studenta")
    p.add_argument("--thr", type=float, default=0.5)
    args = p.parse_args()

    make_noisy_student_examples(args.experiment_dir, thr=args.thr)
