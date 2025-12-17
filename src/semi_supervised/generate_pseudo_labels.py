import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2

from src.models.maau import MAAU
from src.training.data import make_loader_unlabeled
from src.data.augmentations import get_augmentations_teacher


def save_mask(mask_tensor, save_path):
    mask_np = mask_tensor.cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_np = (mask_np * 255).astype("uint8")
    cv2.imwrite(str(save_path), mask_np)


def generate_pseudo_labels(cfg, experiment_dir: Path, checkpoint_path=None, round_id: int =1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MAAU(
        in_channels=3,
        out_channels=1,
        final_activation=None
    ).to(device)

    if checkpoint_path is None:
        ckpt_path = cfg["semi_supervised"]["teacher_checkpoint"]
    else:
        ckpt_path = checkpoint_path

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    aug = get_augmentations_teacher()
    unlabeled_csv = cfg["data"]["unlabeled_csv"]

    unl_dl = make_loader_unlabeled(
        unlabeled_csv,
        aug,
        batch_size=cfg["training"]["batch_size"],
        num_workers=2
    )

    pseudo_root = Path(experiment_dir) / "pseudo_labels"
    masks_dir = pseudo_root / f"masks_r{round_id}"
    masks_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []

    thr = cfg["semi_supervised"]["confidence_thr"]

    meta_csv_path = pseudo_root / f"pseudo_labels_r{round_id}.csv"

    print(f"[INFO] Generowanie pseudo-masek (runda {round_id}) z ckpt: {ckpt_path}")
    with torch.no_grad():
        for imgs, img_paths in tqdm(unl_dl):
            imgs = imgs.to(device)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            binary = (probs >= thr).float()

            for i in range(len(img_paths)):
                img_path = img_paths[i]
                filename = Path(img_path).stem + f"_pseudo_r{round_id}.png"
                save_path = masks_dir / filename

                save_mask(binary[i], save_path)

                meta_rows.append({
                    "image_path": img_path,
                    "pseudo_mask_path": str(save_path),
                    "mean_conf": float(probs[i].mean().item())
                })

    df = pd.DataFrame(meta_rows)
    df.to_csv(meta_csv_path, index=False)

    print(f"[INFO] Zapisanie pseudo-masek do:   {pseudo_root}")
    print(f"[INFO] Meta CSV (runda {round_id}): {meta_csv_path}")
    print(f"[INFO] Liczba wygenerowanych masek: {len(meta_rows)}")

    return meta_csv_path
