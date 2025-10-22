import argparse, yaml, pathlib
from pathlib import Path

def main(cfg):
    print("Preprocessing config:", cfg)
    # tu w przyszłości: wczytaj obrazy z data/raw i zapisz do data/processed
    Path("data/processed").mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)