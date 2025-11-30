import csv, time
from pathlib import Path
import torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from src.models.maau import MAAU
from src.training.metrics import dice_coeff, iou_score, precision_score, recall_score, specificity_score
from src.training.data import make_loaders_supervised
from src.data.augmentations import get_augmentations_supervised
from src.training.losses import dice_loss, bce_loss, combined_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2


@torch.no_grad()
def evaluate(model, loader, device, thr, eps, loss_fn=None):
    model.eval()
    total_loss = 0.0
    dices, ious, precs, recs, specs = [], [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        if loss_fn is not None:
            total_loss += loss_fn(pred, y).item() * x.size(0)

        dices.append(dice_coeff(pred, y, thr=thr, eps=eps))
        ious.append(iou_score(pred, y, thr=thr, eps=eps))
        precs.append(precision_score(pred, y, thr=thr, eps=eps))
        recs.append(recall_score(pred, y, thr=thr, eps=eps))
        specs.append(specificity_score(pred, y, thr=thr, eps=eps))

        n = len(loader.dataset)
    return {
        "loss": total_loss / n if loss_fn is not None else None,
        "dice": sum(dices)/len(dices),
        "iou": sum(ious)/len(ious),
        "precision": sum(precs)/len(precs),
        "recall": sum(recs)/len(recs),
        "specificity": sum(specs)/len(specs),
    }


def train_one_epoch(model, loader, opt, scaler, loss_fn, device, use_amp=True):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def fit(cfg, experiment_dir: Path):
    device = torch.device(cfg["training"]["device"])

    train_tf = get_augmentations_supervised()
    val_tf = A.Compose([A.Resize(256,256),A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])

    train_dl, val_dl = make_loaders_supervised(
        cfg["data"]["train_csv"],
        cfg["data"]["val_csv"],
        train_tf,
        val_tf,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = MAAU(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        final_activation=cfg["model"]["final_activation"]
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"]
    )

    sched_cfg = cfg["scheduler"]
    if sched_cfg["type"] == "cosine_annealing":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=sched_cfg["t_max"],
            eta_min=sched_cfg["min_lr"]
        )
    else:
        sched = None

    loss_cfg = cfg["loss"]
    if loss_cfg["type"] == "dice_bce":
        loss_fn = combined_loss(
            alpha=loss_cfg["dice_weight"],
            beta=loss_cfg["bce_weight"]
        )
    elif loss_cfg["type"] == "dice":
        loss_fn = dice_loss
    elif loss_cfg["type"] == "bce":
        loss_fn = bce_loss
    else:
        raise ValueError("Unknown loss function")

    use_amp = cfg["training"]["mixed_precision"]
    scaler = GradScaler(enabled=use_amp)

    logs_path = experiment_dir / "logs" / "metrics.csv"
    with open(logs_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "dice", "iou", "precision", "recall", "specificity",
            "lr"
        ])

    patience = cfg["training"]["early_stopping_patience"]
    best_dice = -1
    wait = 0

    ckpt_dir = experiment_dir / "checkpoints"
    best_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_dl, opt, scaler, loss_fn, device, use_amp
        )

        val_metrics = evaluate(
            model, val_dl, device,
            thr=float(cfg["metrics"]["threshold"]),
            eps=float(cfg["metrics"]["eps"]),
            loss_fn=loss_fn
        )

        if sched:
            sched.step()

        # zapis do CSV
        with open(logs_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['dice']:.4f}",
                f"{val_metrics['iou']:.4f}",
                f"{val_metrics['precision']:.4f}",
                f"{val_metrics['recall']:.4f}",
                f"{val_metrics['specificity']:.4f}",
                f"{opt.param_groups[0]['lr']:.6f}",
            ])

        print(
            f"[E{epoch:03d}] "
            f"train={train_loss:.4f} | "
            f"val_dice={val_metrics['dice']:.4f}, val_loss={val_metrics['loss']:.4f} | "
            f"time={time.time()-t0:.1f}s"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            wait = 0
            torch.save(
                {"state_dict": model.state_dict(), "val_dice": best_dice},
                best_path
            )
        else:
            wait += 1

        if wait >= patience:
            print(
                f"[INFO] Early stopping na epoce {epoch} "
                f"(best dice={best_dice:.4f})"
            )
            break

    torch.save(
        {"state_dict": model.state_dict(), "val_dice": best_dice},
        ckpt_dir / "last_model.pt"
    )

    return str(best_path)


