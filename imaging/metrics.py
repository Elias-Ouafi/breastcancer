"""Losses and metrics for the imaging pipelines.

Two families live here. The **segmentation** ones operate on PyTorch tensors shaped
``(N, 1, H, W)``: the metric helpers (`dice_coeff`, `iou_score`) expect **binary**
masks (0/1), so threshold the sigmoid output at 0.5 before calling them, while
`DiceBCELoss` takes raw **logits** and uses the soft probabilities so gradients flow.
The **classification** ones (`roc_auc`, `bootstrap_auc`, `operating_point`) take plain
numpy arrays of one score per case, because that is the shape a decision at the level
of a patient has -- and they resample patients, never slices, for the reason
`bootstrap_auc` spells out.

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

import numpy as np
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
    favour recall, suited to the tiny lesion-vs-background ratio in MRI slices),
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
        # Force fp32 regardless of an enclosing torch.amp.autocast context. The
        # fractional power below (gamma < 1) has gradient d/dx[x^gamma] = gamma *
        # x^(gamma-1), which diverges as x -> 0 -- and x = (1 - tversky) does hit
        # near-zero often once training sees enough easy, well-predicted negative
        # slices (e.g. after raising --neg-per-pos). fp16's limited precision near
        # zero turns that steep-but-finite gradient into an outright NaN; observed
        # in practice as train_loss -> nan and val Dice collapsing to 0 within one
        # epoch once neg_per_pos went from 2 to 8. The rest of the model (forward,
        # backward through the conv layers) stays in fp16 for speed -- only this
        # numerically sensitive tail is exempted.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits = logits.float()
            target = target.float()
            probs = torch.sigmoid(logits)
            tversky = tversky_index(probs, target, self.alpha, self.beta, self.eps)
            return (1.0 - tversky).clamp_min(self.eps) ** self.gamma


# --------------------------------------------------------------------------- #
# Classification, at the level a decision is taken: one score per case
# --------------------------------------------------------------------------- #

def roc_auc(labels, scores):
    """Area under the ROC curve, as ``P(score of a positive > score of a negative)``.

    Computed from the rank sum (the Mann-Whitney U identity) rather than by
    integrating a sampled curve: it is exact, ties count as half a win, and there is
    no threshold grid to choose. Returns ``nan`` when either class is missing -- an
    AUC of a single class is not 0.5, it is undefined, and saying so beats printing a
    number that looks like chance.
    """
    labels = np.asarray(labels).astype(float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    n_pos = int((labels > 0).sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    # Average the ranks of tied scores, so a model that cannot separate two cases
    # gets no credit for the order they happen to sit in.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, sorted_scores.size + 1):
        if i == sorted_scores.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[labels > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(labels, scores, n_resamples=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for :func:`roc_auc`, resampling **cases**.

    One case is one patient. Resampling slices instead would treat slices of the same
    breast as independent evidence and shrink the interval to a width the data does
    not support -- the same rule the segmentation evaluation follows.

    Resamples that draw a single class carry no AUC and are dropped; ``n_usable``
    says how many remained, so a corpus too small to support the interval shows it
    here rather than in a footnote.
    """
    labels = np.asarray(labels).astype(float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    point = roc_auc(labels, scores)
    if np.isnan(point):
        return {"auc": point, "lo": float("nan"), "hi": float("nan"),
                "n": int(labels.size), "n_usable": 0}

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, labels.size, size=(n_resamples, labels.size))
    draws = np.array([roc_auc(labels[p], scores[p]) for p in picks], dtype=float)
    usable = draws[~np.isnan(draws)]
    return {
        "auc": point,
        "lo": float(np.percentile(usable, 100 * alpha / 2)) if usable.size else float("nan"),
        "hi": float(np.percentile(usable, 100 * (1 - alpha / 2))) if usable.size else float("nan"),
        "n": int(labels.size),
        "n_positive": int((labels > 0).sum()),
        "n_usable": int(usable.size),
    }


def operating_point(labels, scores, threshold=0.5):
    """Sensitivity, specificity, PPV, NPV and accuracy at one threshold.

    The prevalence of the evaluated set is returned alongside, and not as decoration:
    sensitivity and specificity are properties of the model at this threshold, while
    PPV and NPV are not -- they move with prevalence. A PPV quoted without the
    prevalence it was measured at is a number without a unit (DOCUMENTATION.md, "Ce qu'on ne
    vise pas : la VPP").
    """
    labels = np.asarray(labels).astype(float).ravel() > 0
    positive = np.asarray(scores, dtype=float).ravel() >= threshold

    tp = int((positive & labels).sum())
    fp = int((positive & ~labels).sum())
    tn = int((~positive & ~labels).sum())
    fn = int((~positive & labels).sum())

    def ratio(num, den):
        return float(num / den) if den else float("nan")

    return {
        "threshold": float(threshold),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "accuracy": ratio(tp + tn, labels.size),
        "prevalence": ratio(tp + fn, labels.size),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def threshold_for_sensitivity(labels, scores, target):
    """Lowest threshold whose sensitivity still reaches ``target``.

    A screening tool does not pick its threshold to maximise an accuracy: the
    programme it is compared against publishes a sensitivity (82.8 %, DOCUMENTATION.md "Cible
    chiffrée"), and everything else is what that sensitivity costs. So the sensitivity
    is fixed first and the specificity is *read off*, never the reverse.

    Catching ``k`` of the ``n`` positives means thresholding at the k-th highest
    positive score, with ``k = ceil(target * n)``: the smallest k that reaches the
    target. Ties can drag extra cases over the line, so the achieved sensitivity is
    ``>= target`` rather than equal to it -- read it from :func:`operating_point`
    rather than assuming the target was met.

    Returns ``nan`` when the set holds no positive: a sensitivity is undefined there,
    and a threshold that pretends otherwise would quietly classify everything.
    """
    labels = np.asarray(labels).astype(float).ravel() > 0
    scores = np.asarray(scores, dtype=float).ravel()
    positives = np.sort(scores[labels])[::-1]
    if positives.size == 0:
        return float("nan")

    k = int(np.ceil(float(target) * positives.size))
    k = min(max(k, 1), positives.size)
    return float(positives[k - 1])


def bootstrap_operating_point(labels, decisions, n_resamples=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CIs for one operating point, resampling **cases**.

    Takes decisions already made (a 0/1 array), not scores and a threshold, because
    the threshold is not always one number: chosen fold by fold, each case is judged
    by a threshold its own fold never saw, and there is no single value to re-apply to
    a resample. Resampling the decisions keeps whatever rule produced them fixed,
    which is the property an interval on that rule's behaviour needs.

    Resamples missing a class are dropped the way :func:`bootstrap_auc` drops them --
    a specificity over zero negatives is undefined, not 0 -- and ``n_usable`` reports
    how many survived.
    """
    labels = np.asarray(labels).astype(float).ravel() > 0
    decisions = np.asarray(decisions).astype(float).ravel() > 0

    rng = np.random.default_rng(seed)
    draws = {"sensitivity": [], "specificity": [], "ppv": [], "npv": []}
    usable = 0
    for picks in rng.integers(0, labels.size, size=(n_resamples, labels.size)):
        y, d = labels[picks], decisions[picks]
        if not y.any() or y.all():
            continue
        usable += 1
        tp = int((d & y).sum())
        fp = int((d & ~y).sum())
        tn = int((~d & ~y).sum())
        fn = int((~d & y).sum())
        draws["sensitivity"].append(tp / (tp + fn))
        draws["specificity"].append(tn / (tn + fp))
        if tp + fp:
            draws["ppv"].append(tp / (tp + fp))
        if tn + fn:
            draws["npv"].append(tn / (tn + fn))

    out = {}
    for name, values in draws.items():
        arr = np.asarray(values, dtype=float)
        out[name] = {
            "lo": float(np.percentile(arr, 100 * alpha / 2)) if arr.size else float("nan"),
            "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))) if arr.size else float("nan"),
            "n_usable": int(arr.size),
        }
    out["n"] = int(labels.size)
    out["n_usable"] = int(usable)
    return out
