from pathlib import Path
import pandas as pd

# Lokalizacja danych
RAW_IMG_DIR = Path("data/raw/images/train")
RAW_MASK_DIR = Path("data/raw/masks/train")
OUT_DIR = Path("data/metadata")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    image_files = sorted(RAW_IMG_DIR.glob("*.jpg"))

    records = []
    for img_path in image_files:
        stem = img_path.stem
        mask_candidates = list(RAW_MASK_DIR.glob(f"{stem}*_segmentation.*"))

        mask_path = mask_candidates[0] if mask_candidates else None

        records.append({
            "filename": stem,
            "image_path": img_path.as_posix(),
            "mask_path": mask_path.as_posix() if mask_path else ""
        })

    df = pd.DataFrame(records)
    out_path = OUT_DIR / "isic2018_raw_train.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved raw manifest {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
