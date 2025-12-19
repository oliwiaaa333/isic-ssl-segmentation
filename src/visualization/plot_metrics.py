import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def collect_metrics(experiments_root: Path):
    rows = []

    for metrics_path in experiments_root.glob("**/test/metrics.json"):
        with open(metrics_path) as f:
            metrics = json.load(f)

        exp_name = metrics_path.parents[1].name
        metrics["experiment"] = exp_name
        rows.append(metrics)

    if not rows:
        raise RuntimeError("Nie znaleziono żadnych plików metrics.json")

    return pd.DataFrame(rows)


def plot_metric(df, metric: str, out_dir: Path):
    plt.figure(figsize=(6, 4))

    df_sorted = df.sort_values(metric, ascending=False)

    plt.bar(df_sorted["method"], df_sorted[metric])
    plt.ylabel(metric.capitalize())
    plt.title(f"Porównanie metod – {metric}")
    plt.ylim(0, 1)

    for i, v in enumerate(df_sorted[metric]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{metric}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Zapisano wykres: {out_path}")


def main(experiments_root: str, out_dir: str):

    experiments_root = Path(experiments_root)
    out_dir = Path(out_dir)

    df = collect_metrics(experiments_root)

    print("\nZnalezione wyniki:")
    print(df[["experiment", "method", "dice", "iou"]])

    metrics = ["dice", "iou", "precision", "recall", "specificity"]

    for m in metrics:
        plot_metric(df, m, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiments_root",
        default="experiments",
        help="Katalog z eksperymentami",
    )
    p.add_argument(
        "--out_dir",
        default="experiments/plots",
        help="Gdzie zapisać wykresy",
    )
    args = p.parse_args()

    main(args.experiments_root, args.out_dir)
