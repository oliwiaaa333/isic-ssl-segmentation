import shutil
from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED_CSV = Path("data/metadata/isic2018_processed.csv")
BASE_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = {
    "train_labeled": 50,
    "train_unlabeled": 500,
    "val": 100,
    "test": 1000,
}


def prepare_dirs():
    for split in SIZES.keys():
        split_dir = BASE_DIR / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        if split != "train_unlabeled":
            (split_dir / "masks").mkdir(parents=True, exist_ok=True)


def copy_file(src: str, dst: Path):
    if src:
        shutil.copy(src, dst)


def main():
    df = pd.read_csv(PROCESSED_CSV)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    prepare_dirs()

    idx = 0
    manifests = {}

    for split, size in SIZES.items():
        subset = df.iloc[idx:idx + size].copy()
        idx += size

        split_dir = BASE_DIR / split

        has_masks = split != "train_unlabeled"

        for _, row in subset.iterrows():
            src_img = row["image_path"]
            dst_img = split_dir / "images" / Path(src_img).name
            copy_file(src_img, dst_img)

            if has_masks and isinstance(row["mask_path"], str) and row["mask_path"]:
                src_mask = row["mask_path"]
                dst_mask = split_dir / "masks" / Path(src_mask).name
                copy_file(src_mask, dst_mask)
            else:
                row["mask_path"] = ""

            row["image_path"] = dst_img.as_posix()
            if has_masks:
                if row["mask_path"]:
                    row["mask_path"] = (split_dir / "masks" / Path(row["mask_path"]).name).as_posix()

        manifests[split] = subset

    for split, df_split in manifests.items():
        out_csv = OUTPUT_DIR / f"{split}.csv"
        df_split.to_csv(out_csv, index=False)
        print(f"[INFO] Saved {split} → {out_csv} ({len(df_split)} rows)")


if __name__ == "__main__":
    main()
