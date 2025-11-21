import torch
import yaml
from pathlib import Path
import argparse

from src.models.maau import MAAU
from src.training.data import make_loader_test
from src.training.metrics import dice_coeff, iou_score, precision_score, recall_score, specificity_score

import albumentations as A
from albumentations.pytorch import ToTensorV2


@torch.no_grad()
def evaluate_model(cfg_path, checkpoint_path):

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"])

    test_tf = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_csv = cfg["data"]["val_csv"]

    test_dl = make_loader_test(test_csv, test_tf, num_workers=cfg["data"]["num_workers"])

    model = MAAU(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        final_activation=cfg["model"]["final_activation"]
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dices, ious, precs, recs, specs = [], [], [], [], []

    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        dices.append(dice_coeff(pred, y))
        ious.append(iou_score(pred, y))
        precs.append(precision_score(pred, y))
        recs.append(recall_score(pred, y))
        specs.append(specificity_score(pred, y))

    print("\n Metryki: ")
    print("Dice:", sum(dices)/len(dices))
    print("IoU:", sum(ious)/len(ious))
    print("Precision:", sum(precs)/len(precs))
    print("Recall:", sum(recs)/len(recs))
    print("Specificity:", sum(specs)/len(specs))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    evaluate_model(args.config, args.checkpoint)