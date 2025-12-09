import time
import copy
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import (
    combined_loss_logits,
    dumm_unsup_loss_logits,
)
from src.semi_supervised.dumm.sus import compute_sus_split
from src.training.data import (
    make_loader_labeled,
    make_loader_unlabeled,
    make_loader_eval,
)
from src.data.augmentations import get_augmentations
from src.training.metrics import (
    dice_coeff,
    iou_score,
    precision_score,
    recall_score,
    specificity_score,
)
from src.models.maau import MAAU


def ema_update(teacher, student, decay):
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data = decay * tp.data + (1 - decay) * sp.data


@torch.no_grad()
def evaluate_logits_full(model, loader, device, thr, eps):
    model.eval()

    dices, ious, precs, recs, specs = [], [], [], [], []

    metric_fns = {
        "dice": dice_coeff,
        "iou": iou_score,
        "precision": precision_score,
        "recall": recall_score,
        "specificity": specificity_score,
    }

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
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


@torch.no_grad()
def generate_pseudo_online(t1, t2, x):
    t1.eval()
    t2.eval()
    p1 = torch.sigmoid(t1(x))
    p2 = torch.sigmoid(t2(x))
    return 0.5 * (p1 + p2)


def compute_pixel_kl(student_logits, teacher_probs, eps=1e-7):
    s = torch.sigmoid(student_logits)
    t = torch.clamp(teacher_probs, eps, 1 - eps)

    kl = s * torch.log((s + eps) / (t + eps)) + \
         (1 - s) * torch.log(((1 - s) + eps) / ((1 - t) + eps))
    return kl


def train_epoch_dumm(
        student,
        teacher1,
        teacher2,
        labeled_dl,
        unlabeled_dl,
        optimizer,
        device,
        tau_p,
        unsup_weight,
        sus_mask_set,
):
    student.train()

    sup_loss_fn = combined_loss_logits()
    t0 = time.time()

    total_loss = 0.0
    total_sup = 0.0
    total_unsup = 0.0
    steps = 0

    unl_iter = iter(unlabeled_dl)

    for x_l, y_l in tqdm(labeled_dl, desc="Train epoch"):
        x_l, y_l = x_l.to(device), y_l.to(device)

        logits_l = student(x_l)
        loss_sup = sup_loss_fn(logits_l, y_l)

        try:
            x_u, paths = next(unl_iter)
        except StopIteration:
            unl_iter = iter(unlabeled_dl)
            x_u, paths = next(unl_iter)

        x_u = x_u.to(device)
        p_avg = generate_pseudo_online(teacher1, teacher2, x_u)
        logits_u = student(x_u)
        pixel_unc = compute_pixel_kl(logits_u, p_avg)

        sus_mask = None
        if sus_mask_set is not None:
            sus_mask = torch.tensor(
                [p in sus_mask_set for p in paths],
                dtype=torch.bool,
                device=device
            )

        loss_unsup = dumm_unsup_loss_logits(
            logits_u,
            p_avg,
            pixel_unc,
            tau_p=tau_p,
            sus_mask=sus_mask
        )

        loss = loss_sup + unsup_weight * loss_unsup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema_update(teacher1, student, decay=0.999)
        ema_update(teacher2, student, decay=0.99)

        total_loss += float(loss.item())
        total_sup += float(loss_sup.item())
        total_unsup += float(loss_unsup.item())
        steps += 1

    return {
        "loss": total_loss / max(1, steps),
        "sup_loss": total_sup / max(1, steps),
        "unsup_loss": total_unsup / max(1, steps),
        "time_sec": time.time() - t0,
    }


def train_dumm(cfg, experiment_dir):

    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    sup_tf = get_augmentations(cfg["augmentations"]["supervised"])
    student_tf = get_augmentations(cfg["augmentations"]["student_dumm"])
    teacher_tf = get_augmentations(cfg["augmentations"]["teacher_dumm"])

    labeled_dl = make_loader_labeled(
        cfg["data"]["labeled_csv"],
        student_tf,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    unlabeled_dl_sus = make_loader_unlabeled(
        cfg["data"]["unlabeled_csv"],
        teacher_tf,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    unlabeled_dl_train = make_loader_unlabeled(
        cfg["data"]["unlabeled_csv"],
        student_tf,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"]
    )

    val_dl = make_loader_eval(
        cfg["data"]["val_csv"],
        sup_tf,
        num_workers=cfg["data"]["num_workers"]
    )

    student = MAAU(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        final_activation=None,
    ).to(device)

    teacher1 = copy.deepcopy(student).to(device)
    teacher2 = copy.deepcopy(student).to(device)

    metrics_rows = []

    # PHASE 1: PRETRAIN
    print("\n[PHASE 1] Pretraining student…")

    sup_loss_fn = combined_loss_logits()
    opt_pre = torch.optim.Adam(student.parameters(), lr=0.001)

    es_patience = cfg["training"]["early_stopping_patience"]
    best_val = -1.0

    for epoch in range(1, cfg["training"]["pretrain_epochs"] + 1):
        student.train()
        t0 = time.time()

        total, steps = 0.0, 0

        for x, y in tqdm(labeled_dl, desc=f"Pretrain epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            logits = student(x)
            loss = sup_loss_fn(logits, y)
            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()

            total += float(loss.item())
            steps += 1

        train_loss = total / max(1, steps)

        val_metrics = evaluate_logits_full(
            model=student,
            loader=val_dl,
            device=device,
            thr=cfg["metrics"]["threshold"],
            eps=cfg["metrics"]["eps"],
        )

        if val_metrics["dice"] > best_val:
            best_val = val_metrics["dice"]
            es_patience = cfg["training"]["early_stopping_patience"]
            torch.save({"state_dict": student.state_dict()},
                       Path(experiment_dir) / "checkpoints" / "best_pretrain.pt")
        else:
            es_patience -= 1
            if es_patience == 0:
                print("[INFO] Early stopping in pretraining")
                break

        metrics_rows.append({
            "phase": "pretrain",
            "epoch": epoch,
            "loss": train_loss,
            "sup_loss": train_loss,
            "unsup_loss": 0.0,
            "time_sec": time.time() - t0,
            **val_metrics
        })

    pretrained_state = copy.deepcopy(student.state_dict())

    # PHASE 2: SUS
    print("\n[PHASE 2] Running SUS prioritization…")

    du_dir = Path(cfg["sus"]["du1_csv"]).parent
    du_dir.mkdir(parents=True, exist_ok=True)

    D_u1, D_u2 = compute_sus_split(
        model=student,
        unlabeled_loader=unlabeled_dl_sus,
        device=device,
        K=cfg["sus"]["dropout_iterations"],
        rho=cfg["sus"]["softmax_threshold"],
        save_csv=True,
        out_dir=du_dir
    )

    student.load_state_dict(pretrained_state)
    teacher1.load_state_dict(pretrained_state)
    teacher2.load_state_dict(pretrained_state)

    # PHASE 3: ROUND 1
    print("\n[PHASE 3] Training on D_u1…")

    opt = torch.optim.Adam(student.parameters(), lr=cfg["optimizer"]["lr"])

    sus_set = set(D_u1)
    best_val = -1.0
    es_patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(1, cfg["training"]["round1_epochs"] + 1):
        stats = train_epoch_dumm(
            student,
            teacher1,
            teacher2,
            labeled_dl,
            unlabeled_dl_train,
            opt,
            device,
            tau_p=cfg["pum"]["kl_temperature"],
            unsup_weight=cfg["pum"]["unsupervised_weight"],
            sus_mask_set=sus_set
        )

        val_metrics = evaluate_logits_full(
            model=student,
            loader=val_dl,
            device=device,
            thr=cfg["metrics"]["threshold"],
            eps=cfg["metrics"]["eps"],
        )

        metrics_rows.append({
            "phase": "round1",
            "epoch": epoch,
            **stats,
            **val_metrics
        })

        print(f"[Round1 E{epoch}] train={stats['loss']:.4f} | "
              f"sup={stats['sup_loss']:.4f} | unsup={stats['unsup_loss']:.4f} | "
              f"val_dice={val_metrics['dice']:.4f}")

        if val_metrics["dice"] > best_val:
            best_val = val_metrics["dice"]
            es_patience = cfg["training"]["early_stopping_patience"]
            torch.save({"state_dict": student.state_dict()},
                       Path(experiment_dir) / "checkpoints" / "best_round1.pt")
        else:
            es_patience -= 1
            if es_patience == 0:
                print("[INFO] Early stopping in Round1")
                break

    # PHASE 4: ROUND 2
    print("\n[PHASE 4] Training on D_u1 ∪ D_u2…")

    sus_set = set(D_u1 + D_u2)
    best_val = -1.0
    es_patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(1, cfg["training"]["round2_epochs"] + 1):
        stats = train_epoch_dumm(
            student,
            teacher1,
            teacher2,
            labeled_dl,
            unlabeled_dl_train,
            opt,
            device,
            tau_p=cfg["pum"]["kl_temperature"],
            unsup_weight=cfg["pum"]["unsupervised_weight"],
            sus_mask_set=sus_set
        )

        val_metrics = evaluate_logits_full(
            model=student,
            loader=val_dl,
            device=device,
            thr=cfg["metrics"]["threshold"],
            eps=cfg["metrics"]["eps"],
        )

        metrics_rows.append({
            "phase": "round2",
            "epoch": epoch,
            **stats,
            **val_metrics
        })

        print(f"[Round2 E{epoch}] train={stats['loss']:.4f} | "
              f"val_dice={val_metrics['dice']:.4f}")

        if val_metrics["dice"] > best_val:
            best_val = val_metrics["dice"]
            es_patience = cfg["training"]["early_stopping_patience"]
            torch.save({"state_dict": student.state_dict()},
                       Path(experiment_dir) / "checkpoints" / "best_round2.pt")
        else:
            es_patience -= 1
            if es_patience == 0:
                print("[INFO] Early stopping in Round2")
                break

    # save last
    last_path = Path(experiment_dir) / "checkpoints" / "last_model.pt"
    torch.save({"state_dict": student.state_dict()}, last_path)

    best_model_path = Path(experiment_dir) / "checkpoints" / "best_round2.pt"

    return metrics_rows, str(best_model_path)
