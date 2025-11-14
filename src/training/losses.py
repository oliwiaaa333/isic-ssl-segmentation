import torch
import torch.nn as nn


def dice_loss(pred, target, eps=1e-7):
    num = 2 * (pred * target).sum(dim=(1, 2, 3)) + eps
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def bce_loss(pred, target):
    return nn.BCELoss()(pred, target)


def combined_loss(pred, target, alpha=0.5):
    return alpha * dice_loss(pred, target) + (1 - alpha) * bce_loss(pred, target)