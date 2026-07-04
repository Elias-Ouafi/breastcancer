"""Segmentation losses and metrics for the MRI lesion-detection pipeline.

All functions operate on PyTorch tensors shaped ``(N, 1, H, W)``. The metric
helpers (`dice_coeff`, `iou_score`) expect **binary** masks (0/1): threshold the
sigmoid output at 0.5 before calling them. `DiceBCELoss` takes raw **logits** and
uses the soft (un-thresholded) probabilities so gradients flow.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(pred, target, eps=1e-6):
    """Mean Dice coefficient over the batch. Inputs shaped (N, 1, H, W)."""
    n = pred.shape[0]
    pred = pred.reshape(n, -1)
    target = target.reshape(n, -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return ((2.0 * inter + eps) / (union + eps)).mean()


def iou_score(pred, target, eps=1e-6):
    """Mean Intersection-over-Union (Jaccard) over the batch."""
    n = pred.shape[0]
    pred = pred.reshape(n, -1)
    target = target.reshape(n, -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()


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
