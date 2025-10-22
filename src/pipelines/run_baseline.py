import argparse, yaml, pathlib
from pathlib import Path

def train(cfg):
    print("Training baseline with config:", cfg)
    # TODO: dataloaders z data/processed, model MAAU, pętla trenowania itd.
    Path("experiments/2025-01-01_02_maau_seed0").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)