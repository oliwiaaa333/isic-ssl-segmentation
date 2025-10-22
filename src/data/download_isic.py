"""
download_isic.py — miejsce na pobieranie/organizację zbioru ISIC 2018.

Warianty użycia (docelowo):
1) Ręczne pobranie ZIP-ów/plików -> rozpakuj do data/raw i uruchom preprocess.
2) Automatyczne pobranie:
   - przygotuj CSV z kolumnami: image_url, mask_url
   - python -m src.data.download_isic --csv urls.csv --out data/raw
"""
import argparse, csv, os
from pathlib import Path
from urllib.request import urlretrieve

def download_from_csv(csv_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            for key in ("image_url", "mask_url"):
                if row.get(key):
                    fname = os.path.join(out, os.path.basename(row[key]))
                    if not os.path.exists(fname):
                        print(f"[{i}] downloading {row[key]}")
                        urlretrieve(row[key], fname)
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="CSV with image_url,mask_url")
    ap.add_argument("--out", type=str, default="data/raw")
    args = ap.parse_args()

    if args.csv:
        download_from_csv(args.csv, args.out)
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        print(f"Stworzono folder {args.out}.")
        print("Na tym etapie możesz ręcznie wgrać ISIC do data/raw, albo przygotować CSV i użyć --csv.")