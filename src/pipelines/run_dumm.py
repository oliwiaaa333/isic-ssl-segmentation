import argparse
import yaml
from pathlib import Path
from datetime import datetime
import shutil
import csv

from src.training.train_dumm import train_dumm


def prepare_experiment_dir(cfg):
    root = Path(cfg["experiment"]["output_dir"])
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{cfg['experiment']['name']}_seed{cfg['experiment']['seed']}_{timestamp}"
    exp_dir = root / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "splits").mkdir(exist_ok=True)

    shutil.copy(cfg["_config_path"], exp_dir / "config_used.yaml")

    print(f"[INFO] Utworzono katalog eksperymentu: {exp_dir}")
    return exp_dir


def train(cfg, experiment_dir):
    print("Training DUMM\n")
    print("Config:", cfg["_config_path"])
    print(f"Wyniki i logi będą zapisane w: {experiment_dir}")

    metrics_rows, best_model_path = train_dumm(cfg, experiment_dir)

    # zapis CSV
    csv_path = Path(experiment_dir) / "logs" / "metrics_dumm.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "epoch",
            "train_loss", "sup_loss", "unsup_loss",
            "dice", "iou", "precision", "recall", "specificity",
            "time_sec"
        ])
        for row in metrics_rows:
            writer.writerow([
                row["phase"],
                row["epoch"],
                row["loss"],
                row["sup_loss"],
                row["unsup_loss"],
                row["dice"],
                row["iou"],
                row["precision"],
                row["recall"],
                row["specificity"],
                row["time_sec"],
            ])

    print(f"[INFO] Trening zakończony.")
    print(f"[INFO] Najlepszy model zapisany w {best_model_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config

    experiment_dir = prepare_experiment_dir(cfg)
    train(cfg, experiment_dir)
