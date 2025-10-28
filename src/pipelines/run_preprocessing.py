import argparse
import yaml
import csv
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preproc_utils.preprocessing import (
    load_image, load_mask,
    remove_hairs, resize_image,
    normalize_image, save_image
)

def main(cfg):
    csv_path = Path(cfg["input_csv"])
    out_dir = Path(cfg["output_dir"])
    resize_size = tuple(cfg.get("resize", [256, 256]))
    normalize = cfg.get("normalize", True)
    do_hair_removal = cfg.get("hair_removal", {}).get("enabled", False)
    save_format = cfg.get("save_format", "png")
    max_examples = cfg.get("artifacts", {}).get("save_examples", 0)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows[:5]):  # TODO: na razie tylko 5 obrazów jako placeholder
        img_path = Path(row["image_url"])
        mask_path = Path(row["mask_url"]) if row.get("mask_url") else None

        img = load_image(img_path)

        if do_hair_removal:
            img = remove_hairs(img)

        img = resize_image(img, resize_size)
        img_arr = normalize_image(img) if normalize else np.asarray(img)

        save_path = out_dir / "images" / img_path.name
        save_image(img_arr, save_path, fmt=save_format)

        if mask_path:
            mask = load_mask(mask_path)
            mask = resize_image(mask, resize_size, is_mask=True)
            mask_arr = normalize_image(mask)
            save_mask_path = out_dir / "masks" / mask_path.name
            save_image(mask_arr, save_mask_path, fmt=save_format)

        if i < max_examples:
            print(f"[INFO] Example {i} saved: {save_path.name}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)