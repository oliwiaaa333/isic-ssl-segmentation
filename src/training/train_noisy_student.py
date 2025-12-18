import torch
import csv
import time
from itertools import cycle
from pathlib import Path
import numpy as np

from src.models.maau import MAAU
from src.training.data import (
    make_loader_labeled,
    make_loader_pseudo,
    make_loader_eval,
)
from src.data.augmentations import get_augmentations
from src.training.losses import combined_loss_logits
from src.training.metrics import (
    dice_coeff,
    iou_score,
    precision_score,
    recall_score,
    specificity_score,
    evaluate_logits,
)


@torch.no_grad()
def evaluate_logits_full(model, loader, device, metric_fns, thr, eps):
    model.eval()

    dices, ious, precs, recs, specs = [], [], [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)

        dices.append(metric_fns["dice"](probs, y, thr, eps))
        ious.append(metric_fns["iou"](probs, y, thr, eps))
        precs.append(metric_fns["precision"](probs, y, thr, eps))
        recs.append(metric_fns["recall"](probs, y, thr, eps))
        specs.append(metric_fns["specificity"](probs, y, thr, eps))

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "specificity": float(np.mean(specs)),
    }


def train_noisy_student(cfg, experiment_dir,
                        pseudo_csv_path: str,
                        init_checkpoint: str | None = None,
                        tag: str = "r1"):

    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    print("[INFO] Start treningu Noisy Student…")

    sup_tf     = get_augmentations(cfg["augmentations"]["supervised"])
    student_tf = get_augmentations(cfg["augmentations"]["student"])

    labeled_csv = cfg["data"]["supervised_train_csv"]
    val_csv     = cfg["data"]["supervised_val_csv"]

    pseudo_csv = Path(pseudo_csv_path).resolve()

    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    labeled_dl = make_loader_labeled(labeled_csv, student_tf, batch_size, num_workers)
    pseudo_dl  = make_loader_pseudo(pseudo_csv, student_tf, batch_size, num_workers)
    val_dl     = make_loader_eval(val_csv, sup_tf, num_workers)

    model = MAAU(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        final_activation=cfg["model"]["final_activation"],
    ).to(device)

    if init_checkpoint is None:
        ckpt_path = cfg["semi_supervised"]["teacher_checkpoint"]
    else:
        ckpt_path = init_checkpoint

    teacher_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(teacher_ckpt["state_dict"])

    loss_fn = combined_loss_logits(
        alpha=cfg["loss"]["dice_weight"],
        beta=cfg["loss"]["bce_weight"],
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    if tag == "r1":
        T_max = cfg["scheduler"]["t_max"]
    else:
        T_max = cfg["scheduler"].get("t_max_star", cfg["scheduler"]["t_max"])

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=T_max,
        eta_min=cfg["scheduler"]["min_lr"],
    )

    thr = cfg["metrics"]["threshold"]
    eps = cfg["metrics"]["eps"]

    metric_fns = {
        "dice": dice_coeff,
        "iou": iou_score,
        "precision": precision_score,
        "recall": recall_score,
        "specificity": specificity_score,
    }

    exp_dir = Path(experiment_dir)
    ckpt_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"

    csv_path = logs_dir / f"metrics_student_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_dice",
            "val_iou",
            "val_precision",
            "val_recall",
            "val_specificity",
            "lr",
            "time_sec"
        ])

    best_val = -1
    patience = cfg["training"]["early_stopping_patience"]

    best_round_path = ckpt_dir / f"best_model_{tag}.pt"
    last_round_path = ckpt_dir / f"last_model_{tag}.pt"
    best_model_path = ckpt_dir / "best_model.pt"

    if tag == "r1":
        num_epochs = cfg["training"]["epochs_student"]
    else:
        num_epochs = cfg["training"].get("epochs_student_star", cfg["training"]["epochs_student"])

    for epoch in range(1, num_epochs + 1):

        model.train()
        t0 = time.time()
        total_loss = 0.0

        for (x_l, y_l), (x_p, y_p, _) in zip(cycle(labeled_dl), pseudo_dl):
            x = torch.cat([x_l, x_p]).to(device)
            y = torch.cat([y_l, y_p]).to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        sched.step()

        val_metrics = evaluate_logits_full(model, val_dl, device, metric_fns, thr, eps)

        elapsed = time.time() - t0

        print(
            f"[E{epoch:03d}] "
            f"train={total_loss:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                total_loss,
                val_metrics["dice"],
                val_metrics["iou"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["specificity"],
                opt.param_groups[0]["lr"],
                elapsed,
            ])

        if val_metrics["dice"] > best_val:
            best_val = val_metrics["dice"]
            patience = cfg["training"]["early_stopping_patience"]
            torch.save({"state_dict": model.state_dict()}, best_round_path)
            torch.save({"state_dict": model.state_dict()}, best_model_path)
        else:
            patience -= 1
            if patience == 0:
                print(f"[INFO] Early stopping na epoce {epoch} (best dice={best_val:.4f})")
                break

    torch.save({"state_dict": model.state_dict()}, last_round_path)
    print("[INFO] Zapisano last model.")

    return str(best_round_path)
