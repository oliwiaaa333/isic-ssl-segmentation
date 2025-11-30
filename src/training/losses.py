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
    def _loss(pred,target):
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
