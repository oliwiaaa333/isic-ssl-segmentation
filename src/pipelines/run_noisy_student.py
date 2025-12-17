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
    (experiment_dir / "pseudo_labels").mkdir(exist_ok=True)

    config_copy = experiment_dir / "config_used.yaml"
    shutil.copy(cfg["_config_path"], config_copy)

    print(f"[INFO] Utworzono eksperyment: {experiment_dir}")
    return experiment_dir


def train_rounds(cfg, experiment_dir):
    print("Training Noisy Student\n")
    print("Config:", cfg["_config_path"])
    print(f"Wyniki i logi będą zapisane w: {experiment_dir}")

    rounds = cfg["semi_supervised"]["rounds"]

    print("[INFO] Runda 1: teacher -> student")

    generate_pseudo_labels(cfg,experiment_dir=experiment_dir, checkpoint_path=None, round_id=1)

    filtered_r1 = filter_pseudo_labels(cfg,experiment_dir=experiment_dir, round_id=1)

    print("[INFO] Trening studenta (runda 1)")
    best_student_r1 = train_noisy_student(cfg, experiment_dir, pseudo_csv_path=filtered_r1, init_checkpoint=None, tag="r1")

    if rounds == 1:
        print("[INFO] Trening zakończono po 1 rundzie.")
        print(f"[INFO] Najlepszy model studenta zapisany w: {best_student_r1}")
        return

    print("[INFO] Runda 2: student -> student*")
    generate_pseudo_labels(cfg, experiment_dir=experiment_dir, checkpoint_path=best_student_r1, round_id=2)
    filtered_r2 = filter_pseudo_labels(cfg, experiment_dir=experiment_dir, round_id=2)

    print("[INFO] Trening studenta* (runda 2)")
    best_student_r2 = train_noisy_student(cfg, experiment_dir, pseudo_csv_path=filtered_r2, init_checkpoint=best_student_r1, tag="r2")

    print(f"[INFO] Trening zakończono po 2 rundzie.")
    print(f"[INFO] Najlepszy model studenta* (runda 2) zapisany w {best_student_r2}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config

    experiment_dir = prepare_experiment_dir(cfg)

    train_rounds(cfg, experiment_dir)
