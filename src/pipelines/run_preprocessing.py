import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preproc_utils.preprocessing import (
    load_image, load_mask,
    remove_hairs, resize_image,
    normalize_image, save_image
)

OUT_META_DIR = Path("data/metadata")
OUT_META_DIR.mkdir(parents=True, exist_ok=True)


def main(cfg):
    df = pd.read_csv(cfg["input_csv"])
    out_dir = Path(cfg["output_dir"])
    resize_size = tuple(cfg.get("resize", [256, 256]))
    normalize = cfg.get("normalize", True)
    do_hair_removal = cfg.get("hair_removal", {}).get("enabled", False)
    save_format = cfg.get("save_format", "png")
    max_examples = cfg.get("artifacts", {}).get("save_examples", 0)

    records = []

    for i, row in df.iterrows():
        in_img_path = Path(row["image_path"])
        in_mask_path = Path(row["mask_path"]) if isinstance(row.get("mask_path"), str) and row["mask_path"] else None

        img_pil = load_image(in_img_path)

        if do_hair_removal:
            img_pil = remove_hairs(img_pil)

        img_pil = resize_image(img_pil, resize_size)
        img_np = normalize_image(img_pil) if normalize else np.asarray(img_pil)

        out_img_path = (out_dir / "images" / in_img_path.stem).with_suffix(f".{save_format}")
        save_image(img_np, out_img_path, fmt=save_format)

        out_mask_path = ""
        if in_mask_path:
            mask_pil = load_mask(in_mask_path)
            mask_pil = resize_image(mask_pil, resize_size, is_mask=True)
            mask_np = normalize_image(mask_pil)
            out_mask_path = out_dir / "masks" / in_mask_path.name
            save_image(mask_np, out_mask_path, fmt=save_format)

        if i < max_examples:
            print(f"[INFO] Example {i} processed: {out_img_path.name}")

        records.append({
            "filename": row["filename"],
            "image_path": out_img_path.as_posix(),
            "mask_path": out_mask_path if in_mask_path else ""
        })

    out_csv = OUT_META_DIR / "isic2018_processed.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"[INFO] Preprocessing completed, saved manifest: {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)