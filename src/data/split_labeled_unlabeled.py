"""
Dzieli zbiór treningowy (train.csv) na:
- labeled: podany procent próbek z maskami
- unlabeled: reszta, ale z pustym polem mask_url


Dodatkowo tworzy train_small.csv (100 par obraz–maska).
"""
import csv
import random
from pathlib import Path
from typing import List, Dict


random.seed(42) # powtarzalność


INPUT_CSV = Path("data/metadata/isic2018_task1_train.csv")
OUTPUT_DIR = Path("data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LABELED_PCT = 0.10 # 10%
SMALL_SIZE = 100


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(rows: List[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "image_url", "mask_url"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read_csv(INPUT_CSV)
    total = len(rows)
    n_labeled = int(total * LABELED_PCT)


    shuffled = rows.copy()
    random.shuffle(shuffled)


    labeled = shuffled[:n_labeled]
    unlabeled = shuffled[n_labeled:]


    for r in unlabeled:
        r["mask_url"] = ""


    write_csv(labeled, OUTPUT_DIR / f"isic2018_task1_train_labeled_p10.csv")
    write_csv(unlabeled, OUTPUT_DIR / f"isic2018_task1_train_unlabeled_p90.csv")


    # train_small: 100 pierwszych z pełnego zbioru
    small = shuffled[:SMALL_SIZE]
    write_csv(small, OUTPUT_DIR / "isic2018_task1_train_small.csv")


    print(f"Zapisano: {n_labeled} labeled, {total - n_labeled} unlabeled, {SMALL_SIZE} small")


if __name__ == "__main__":
    main()