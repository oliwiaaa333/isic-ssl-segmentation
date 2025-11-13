import torch

DEFAULT_THR = 0.5
DEFAULT_EPS = 1e-7


def confusion_counts(pred, target, thr=DEFAULT_THR):
    # ujednolicenie wymiarow
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    pred_b = (pred >= thr).float()
    target = target.float()

    tp = (pred_b * target).sum(dim=(1, 2, 3))
    fp = (pred_b * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_b) * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred_b) * (1 - target)).sum(dim=(1, 2, 3))

    return tp, fp, fn, tn


def dice_coeff(pred, target, thr=DEFAULT_THR, eps=DEFAULT_EPS):
    tp, fp, fn, _ = confusion_counts(pred, target, thr)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice.mean().item()


def iou_score(pred, target, thr=DEFAULT_THR, eps=DEFAULT_EPS):
    tp, fp, fn, _ = confusion_counts(pred, target, thr)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return iou.mean().item()


def precision_score(pred, target, thr=DEFAULT_THR, eps=DEFAULT_EPS):
    tp, fp, _, _ = confusion_counts(pred, target, thr)
    prec = (tp + eps) / (tp + fp + eps)
    return prec.mean().item()


def recall_score(pred, target, thr=DEFAULT_THR, eps=DEFAULT_EPS):
    tp, _, fn, _ = confusion_counts(pred, target, thr)
    rec = (tp + eps) / (tp + fn + eps)
    return rec.mean().item()


def specificity_score(pred, target, thr=DEFAULT_THR, eps=DEFAULT_EPS):
    _, fp, _, tn = confusion_counts(pred, target, thr)
    spec = (tn + eps) / (tn + fp + eps)
    return spec.mean().item()
