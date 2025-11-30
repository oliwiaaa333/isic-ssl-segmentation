import pandas as pd
from pathlib import Path


def filter_pseudo_labels(cfg):
    pseudo_root = Path(cfg["data"]["pseudo_labels_root"])
    meta_path = pseudo_root / "pseudo_labels.csv"
    out_path = pseudo_root / "pseudo_labels_filtered.csv"

    min_conf = cfg["semi_supervised"]["min_conf"]
    use_filter = cfg["semi_supervised"]["use_confidence_filter"]

    if not meta_path.exists():
        raise FileNotFoundError(f"[ERROR] Nie znaleziono {meta_path}")

    df = pd.read_csv(meta_path)

    print(f"[INFO] Wczytano {len(df)} pseudo-etykiet.")

    # sanity check — usuwamy wiersze z brakującymi ścieżkami
    df = df.dropna(subset=["image_path", "pseudo_mask_path"])

    before = len(df)

    if use_filter:
        df = df[df["mean_conf"] >= min_conf]
        print(f"[INFO] Po filtracji confidence >= {min_conf}: {len(df)} przykładów.")
    else:
        print("[INFO] Filtracja confidence wyłączona.")

    after = len(df)

    df.to_csv(out_path, index=False)

    print(f"[INFO] Zapisano przefiltrowany zbiór do: {out_path}")
    print(f"[INFO] Zachowano {after}/{before} pseudo-etykiet.")

    return df
