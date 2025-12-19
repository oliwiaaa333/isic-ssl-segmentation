import argparse
import yaml
from pathlib import Path
from datetime import datetime
import shutil

from src.training.train_baseline import fit


def prepare_experiment_dir(cfg):
    root = Path(cfg["paths"]["root"])
    experiment_root = root / cfg["experiment"]["output_dir"]
    experiment_root.mkdir(parents=True, exist_ok=True)

    # pobieranie daty
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{cfg['experiment']['name']}_seed{cfg['experiment']['seed']}_{timestamp}"
    experiment_dir = experiment_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # podkatalogi do zapisania wag modelu i logow
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)

    # kopiowanie parametrow z jakimi trenowany byl model
    config_copy = experiment_dir / "config_used.yaml"
    shutil.copy(cfg["_config_path"], config_copy)

    print(f"[INFO] Utworzono katalog eksperymentu: {experiment_dir}")
    return experiment_dir


def train(cfg, experiment_dir):
    print("Training baseline MAAU\n")
    print("Config:", cfg["_config_path"])
    print(f"Wyniki i logi będą zapisane w: {experiment_dir}")
    best_model_path = fit(cfg, experiment_dir)
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

    train(cfg, experiment_dir)