"""
Generuje pliki CSV (manifesty) dla ISIC 2018 Task 1 na podstawie lokalnych folderów:
 - images/train, val, test
 - masks/train, val

Każdy wiersz zawiera:
- filename (bez rozszerzenia)
- image_url (ścieżka względna do obrazu)
- mask_url (ścieżka względna do maski lub pusta dla testu)

Pliki wyjściowe zapisywane są do `data/metadata/`:
 - isic2018_task1_train.csv
 - isic2018_task1_val.csv
 - isic2018_task1_test.csv
"""

from pathlib import Path
import csv

# Lokalizacja danych
RAW_DIR = Path("data/raw")
IMG_DIR = RAW_DIR / "images"
MASK_DIR = RAW_DIR / "masks"
OUT_DIR = Path("data/metadata")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def collect_files(images_path: Path, masks_path: Path | None = None):
    image_files = sorted(images_path.glob("*.jpg"))
    data = []

    for img_path in image_files:
        stem = img_path.stem  # np. ISIC_0000000
        image_rel = img_path.as_posix()

        mask_rel = ""
        if masks_path:
            candidates = list(masks_path.glob(f"{stem}*_segmentation.*"))
            if candidates:
                mask_rel = candidates[0].as_posix()

        data.append({
            "filename": stem,
            "image_url": image_rel,
            "mask_url": mask_rel
        })

    return data

def write_csv(data: list[dict], out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "image_url", "mask_url"])
        writer.writeheader()
        writer.writerows(data)

def main():
    sets = [
        ("train", IMG_DIR / "train", MASK_DIR / "train"),
        ("val", IMG_DIR / "val", MASK_DIR / "val"),
        ("test", IMG_DIR / "test", None),  # brak masek
    ]

    for name, img_p, mask_p in sets:
        print(f"[INFO] Generating manifest for: {name}")
        rows = collect_files(img_p, mask_p)
        out_file = OUT_DIR / f"isic2018_task1_{name}.csv"
        write_csv(rows, out_file)
        print(f" -> Saved {len(rows)} rows to {out_file}")

if __name__ == "__main__":
    main()
