import torch


def confusion_counts(pred, target, thr):
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


def dice_coeff(pred, target, thr, eps):
    tp, fp, fn, _ = confusion_counts(pred, target, thr)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice.mean().item()


def iou_score(pred, target, thr, eps):
    tp, fp, fn, _ = confusion_counts(pred, target, thr)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return iou.mean().item()


def precision_score(pred, target, thr, eps):
    tp, fp, _, _ = confusion_counts(pred, target, thr)
    prec = (tp + eps) / (tp + fp + eps)
    return prec.mean().item()


def recall_score(pred, target, thr, eps):
    tp, _, fn, _ = confusion_counts(pred, target, thr)
    rec = (tp + eps) / (tp + fn + eps)
    return rec.mean().item()


def specificity_score(pred, target, thr, eps):
    _, fp, _, tn = confusion_counts(pred, target, thr)
    spec = (tn + eps) / (tn + fp + eps)
    return spec.mean().item()


# zamiana logitow w sigmoid, potem wywolanie metryk
def evaluate_logits(logits, target, metric_fns, thr, eps):
    probs = torch.sigmoid(logits)

    results = {}
    for name, fn in metric_fns.items():
        results[name] = fn(probs, target, thr=thr, eps=eps)

    return results
