import pandas as pd
from pathlib import Path


def filter_pseudo_labels(cfg, experiment_dir: Path, round_id: int=1):
    pseudo_root = Path(experiment_dir) / "pseudo_labels"
    meta_path = pseudo_root / f"pseudo_labels_r{round_id}.csv"
    out_path = pseudo_root / f"pseudo_labels_r{round_id}_filtered.csv"

    min_conf = cfg["semi_supervised"]["min_conf"]
    use_filter = cfg["semi_supervised"]["use_confidence_filter"]

    if not meta_path.exists():
        raise FileNotFoundError(f"[ERROR] Nie znaleziono {meta_path}")

    df = pd.read_csv(meta_path)

    print(f"[INFO] Wczytano {len(df)} pseudo-etykiet (runda {round_id}).")

    # sanity check — usuwamy wiersze z brakującymi ścieżkami
    df = df.dropna(subset=["image_path", "pseudo_mask_path"])

    before = len(df)

    if use_filter:
        df = df[df["mean_conf"] >= min_conf]
        print(f"[INFO] Po filtracji confidence >= {min_conf}: {len(df)} przykładów (runda {round_id}).")
    else:
        print("[INFO] Filtracja confidence wyłączona.")

    after = len(df)

    df.to_csv(out_path, index=False)

    print(f"[INFO] (runda {round_id}) Zapisano przefiltrowany zbiór do: {out_path}")
    print(f"[INFO] Zachowano {after}/{before} pseudo-etykiet.")

    return out_path
