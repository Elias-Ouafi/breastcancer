"""Segmentation losses and metrics for the MRI lesion-detection pipeline.

All functions operate on PyTorch tensors shaped ``(N, 1, H, W)``. The metric
helpers (`dice_coeff`, `iou_score`) expect **binary** masks (0/1): threshold the
sigmoid output at 0.5 before calling them. `DiceBCELoss` takes raw **logits** and
uses the soft (un-thresholded) probabilities so gradients flow.

Localisation caveat
-------------------
Dice and IoU are *equal by construction* whenever the intersection is zero (no
overlap, or either mask empty) — both collapse to ``eps / (|P|+|T|+eps)``. They
diverge only under partial overlap. So a reported metric where Dice == IoU
exactly, frozen across epochs, is a red flag that the model never overlaps the
target (e.g. it collapsed to "predict nothing"), *not* a healthy score.

For evaluating **localisation** quality use `segmentation_scores`, which scores
per image and, by default, ignores empty-target frames. Scoring an empty mask
against an empty prediction as a perfect ``1.0`` (the raw ``eps/eps`` behaviour of
`dice_coeff`/`iou_score`) inflates the average with easy true negatives and hides
a model that has learnt nothing — this is exactly what made the old
``segmentation_metrics.csv`` read ~0.97 while the network predicted only zeros.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_sample_overlap(pred, target, eps):
    """Per-sample Dice and IoU plus the target's foreground count.

    Returns ``(dice, iou, target_sum)``, each a 1-D tensor of length ``N``.
    """
    n = pred.shape[0]
    pred = pred.reshape(n, -1)
    target = target.reshape(n, -1)
    inter = (pred * target).sum(dim=1)
    psum = pred.sum(dim=1)
    tsum = target.sum(dim=1)
    dice = (2.0 * inter + eps) / (psum + tsum + eps)
    iou = (inter + eps) / (psum + tsum - inter + eps)
    return dice, iou, tsum


def dice_coeff(pred, target, eps=1e-6):
    """Mean (soft) Dice coefficient over the batch. Inputs shaped (N, 1, H, W).

    Kept smooth (no empty-mask special-casing) so it is usable as a loss term; for
    reporting localisation quality use `segmentation_scores` instead.
    """
    dice, _, _ = _per_sample_overlap(pred, target, eps)
    return dice.mean()


def iou_score(pred, target, eps=1e-6):
    """Mean Intersection-over-Union (Jaccard) over the batch."""
    _, iou, _ = _per_sample_overlap(pred, target, eps)
    return iou.mean()


def segmentation_scores(pred, target, eps=1e-6, ignore_empty=True):
    """Localisation Dice/IoU with correct handling of empty-target frames.

    Scores each image independently and returns running sums so batches can be
    aggregated exactly (a plain mean-of-batch-means is biased when batches differ
    in size). Expects **binary** masks.

    Parameters
    ----------
    ignore_empty : bool
        If True (default), frames whose ground-truth mask is empty are excluded
        from the average. Such frames carry no lesion to localise, and counting
        the trivial empty-vs-empty ``1.0`` swamps the metric with true negatives.

    Returns
    -------
    dict with ``dice_sum``, ``iou_sum`` (floats) and ``count`` (int): the number
    of scored frames. Divide the sums by ``count`` for the mean (``count == 0``
    means no scorable frame in this batch).
    """
    dice, iou, tsum = _per_sample_overlap(pred, target, eps)
    if ignore_empty:
        keep = tsum > 0
        dice, iou = dice[keep], iou[keep]
    return {
        "dice_sum": float(dice.sum().item()),
        "iou_sum": float(iou.sum().item()),
        "count": int(dice.numel()),
    }


class DiceBCELoss(nn.Module):
    """Combined binary cross-entropy + soft-Dice loss.

    BCE stabilises early training and handles the large background class, while the
    Dice term directly optimises overlap — the metric we report. `bce_weight`
    blends the two (0.5 = equal parts).
    """

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        dice = 1.0 - dice_coeff(probs, target)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


def tversky_index(pred, target, alpha=0.3, beta=0.7, eps=1e-6):
    """Batch-mean Tversky index: trades recall vs precision via alpha/beta.

    Generalises Dice (the alpha=beta=0.5 case) by weighting false positives
    (``alpha``) separately from false negatives (``beta``). With ``beta > alpha``,
    missed lesion voxels are penalised more than spurious ones — appropriate here
    since the lesion is a tiny fraction of each slice and recall (not missing the
    lesion) matters more than precision.
    """
    n = pred.shape[0]
    pred = pred.reshape(n, -1)
    target = target.reshape(n, -1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1.0 - target)).sum(dim=1)
    fn = ((1.0 - pred) * target).sum(dim=1)
    return ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (Abraham & Khan, 2018) for foreground/background imbalance.

    Built on `tversky_index`: ``alpha``/``beta`` trade precision vs recall (defaults
    favour recall, suited to the tiny lesion-vs-background ratio in DBT/MRI slices),
    and ``gamma`` raises ``(1 - Tversky)`` to a power < 1, which amplifies the loss
    for harder/less-accurate slices relative to easy ones already segmented well.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        tversky = tversky_index(probs, target, self.alpha, self.beta, self.eps)
        return (1.0 - tversky).clamp_min(self.eps) ** self.gamma
