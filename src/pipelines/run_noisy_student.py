import argparse
import yaml
from pathlib import Path
from datetime import datetime
import shutil

from src.semi_supervised.generate_pseudo_labels import generate_pseudo_labels
from src.semi_supervised.filter_pseudo_labels import filter_pseudo_labels
from src.training.train_noisy_student import train_noisy_student


def prepare_experiment_dir(cfg):
    experiment_root = Path(cfg["experiment"]["output_dir"])
    experiment_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{cfg['experiment']['name']}_seed{cfg['experiment']['seed']}_{timestamp}"
    experiment_dir = experiment_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)

    config_copy = experiment_dir / "config_used.yaml"
    shutil.copy(cfg["_config_path"], config_copy)

    print(f"[INFO] Utworzono eksperyment: {experiment_dir}")
    return experiment_dir


def train(cfg, experiment_dir):
    print("Training Noisy Student\n")
    print("Config:", cfg["_config_path"])
    print(f"Wyniki i logi będą zapisane w: {experiment_dir}")

    best_model_path = train_noisy_student(cfg, experiment_dir)

    print(f"[INFO] Trening zakonczony.")
    print(f"[INFO] Najlepszy model zapisany w {best_model_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config

    experiment_dir = prepare_experiment_dir(cfg)

    pseudo_root = Path(cfg["data"]["pseudo_labels_root"])
    meta_csv = pseudo_root / "pseudo_labels.csv"
    filtered_csv = pseudo_root / "pseudo_labels_filtered.csv"

    if meta_csv.exists():
        print("[INFO] Wykryto istniejący plik CSV. Pomijam generowanie pseudo-etykiet.")
    else:
        generate_pseudo_labels(cfg)

    filter_pseudo_labels(cfg)

    train(cfg, experiment_dir)
