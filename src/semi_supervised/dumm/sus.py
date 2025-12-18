import torch
import pandas as pd
from pathlib import Path
import numpy as np


@torch.no_grad()
def mc_dropout_predict(model, x, K=5):

    model.train()
    preds = []

    for _ in range(K):
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds.append(probs.detach())

    return torch.stack(preds, dim=0)  # [K,B,1,H,W]


def compute_sample_uncertainty(mc_preds):
    var_map = mc_preds.var(dim=0)

    sample_unc = var_map.mean(dim=[1,2,3])

    return sample_unc, var_map


def compute_softmax_confidence(mc_preds, rho=0.9):
    mean_pred = mc_preds.mean(dim=0)

    conf = (mean_pred > rho).float().mean(dim=[1,2,3])  # [B]

    return conf


def compute_sus_split(
        model,
        unlabeled_loader,
        device,
        experiment_dir: Path,
        K=5,
        rho=0.9
):

    model = model.to(device)

    all_paths = []
    all_unc = []
    all_conf = []

    # 1. Compute uncertainty + confidence for all samples
    for x, path in unlabeled_loader:
        x = x.to(device)

        # MC dropout predictions
        mc_preds = mc_dropout_predict(model, x, K=K)  # [K,B,1,H,W]

        # sample uncertainty + pixel uncertainty map
        sample_unc, _ = compute_sample_uncertainty(mc_preds)

        # softmax confidence score
        conf = compute_softmax_confidence(mc_preds, rho=rho)

        # detach as python scalars
        for i in range(len(path)):
            all_paths.append(path[i])
            all_unc.append(float(sample_unc[i].cpu()))
            all_conf.append(float(conf[i].cpu()))

    # Convert to numpy for sorting
    all_unc = np.array(all_unc)
    all_conf = np.array(all_conf)
    all_paths = np.array(all_paths)

    # 2. Initial split by uncertainty (50% lowest = D_u1)
    sorted_idx = np.argsort(all_unc)   # ascending → small=stable

    N = len(sorted_idx)
    half = N // 2

    idx_u1 = sorted_idx[:half]
    idx_u2 = sorted_idx[half:]

    # 3. Softmax filtering / swap phase
    # threshold = mean confidence
    mean_conf = all_conf.mean()

    # candidates for swap:
    # u1: low confidence
    swap_from_u1 = idx_u1[all_conf[idx_u1] < mean_conf]

    # u2: high confidence
    swap_from_u2 = idx_u2[all_conf[idx_u2] > mean_conf]

    # perform swaps
    k = min(len(swap_from_u1), len(swap_from_u2))
    if k > 0:
        idx_u1 = np.setdiff1d(idx_u1, swap_from_u1[:k])
        idx_u2 = np.setdiff1d(idx_u2, swap_from_u2[:k])

        idx_u1 = np.concatenate([idx_u1, swap_from_u2[:k]])
        idx_u2 = np.concatenate([idx_u2, swap_from_u1[:k]])

    # Final lists of paths:
    D_u1 = all_paths[idx_u1].tolist()
    D_u2 = all_paths[idx_u2].tolist()


    splits_dir = experiment_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_path": D_u1}).to_csv(splits_dir / "u1.csv", index=False)
    pd.DataFrame({"image_path": D_u2}).to_csv(splits_dir / "u2.csv", index=False)

    print(f"[SUS] Saved split to {splits_dir}/u1.csv and u2.csv")

    return D_u1, D_u2
