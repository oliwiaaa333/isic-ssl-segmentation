import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-7):
    num = 2 * (pred * target).sum(dim=(1, 2, 3)) + eps
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def bce_loss(pred, target):
    return nn.BCELoss()(pred, target)


def combined_loss(alpha=0.5, beta=0.5):
    def _loss(pred, target):
        return alpha * dice_loss(pred, target) + beta * bce_loss(pred, target)

    return _loss


# dla semi-supervised
bce_logits = nn.BCEWithLogitsLoss()


def dice_loss_logits(logits, target, eps=1e-7):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(1, 2, 3)) + eps
    den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps

    return 1 - (num / den).mean()


def combined_loss_logits(alpha=0.5, beta=0.5):
    def _loss(logits, target):
        dl = dice_loss_logits(logits, target)
        bl = bce_logits(logits, target)
        return alpha * dl + beta * bl

    return _loss


# dumm
def compute_pum_weights(pixel_unc, tau_p=0.5, gamma=5.0, eps=1e-7):
    u_norm = torch.clamp(pixel_unc / (tau_p + eps), 0., 1.)
    weights = torch.exp(-gamma * u_norm)

    return weights


def dumm_unsup_loss_logits(
        logits_u,
        pseudo_labels,
        pixel_unc,
        tau_p=0.5,
        gamma=5.0,
        sus_mask=None,
        alpha=0.5,
        beta=0.5,
        eps=1e-7
):
    if sus_mask is not None:
        logits_u = logits_u[sus_mask]
        pseudo_labels = pseudo_labels[sus_mask]
        pixel_unc = pixel_unc[sus_mask]

        if logits_u.shape[0] == 0:
            return torch.tensor(0.0, device=logits_u.device)

    pum_w = compute_pum_weights(pixel_unc, tau_p=tau_p, gamma=gamma)

    probs = torch.sigmoid(logits_u)

    num = (2 * probs * pseudo_labels * pum_w).sum(dim=(1, 2, 3)) + eps
    den = (probs * pum_w).sum(dim=(1, 2, 3)) + (pseudo_labels * pum_w).sum(dim=(1, 2, 3)) + eps

    dice_loss_val = 1 - (num / den).mean()

    bce_map = F.binary_cross_entropy_with_logits(
        logits_u,
        pseudo_labels,
        reduction="none"
    )
    bce_loss_val = (bce_map * pum_w).mean()

    loss = alpha * dice_loss_val + beta * bce_loss_val

    return loss
