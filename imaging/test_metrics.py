"""Sanity checks for the segmentation metrics.

Run from the repository root with the project venv:

    python -m imaging.test_metrics

These are plain asserts (no pytest dependency) so they can double as a smoke test
right after installing PyTorch. They pin down the two properties that the frozen
``Dice == IoU == 0.97`` bug violated:

1. Under *partial* overlap, Dice and IoU are different and match their textbook
   values — the metric is not silently computing the same thing twice.
2. Empty-target frames do not inflate the localisation score: with
   ``ignore_empty`` they are excluded, so a "predict nothing" model scores ~0 on
   lesion frames instead of ~1 on background.
"""
from __future__ import annotations

import torch

from imaging.metrics import dice_coeff, iou_score, segmentation_scores


def _box(n, h, w, y0, y1, x0, x1):
    t = torch.zeros(n, 1, h, w)
    t[:, :, y0:y1, x0:x1] = 1.0
    return t


def test_partial_overlap_dice_and_iou_differ():
    H = W = 8
    target = _box(1, H, W, 0, 4, 0, 4)        # 16 px
    pred = _box(1, H, W, 2, 6, 2, 6)          # 16 px, overlap = 2x2 = 4 px
    # inter=4, |P|=|T|=16 -> Dice = 2*4/32 = 0.25 ; IoU = 4/(32-4) = 0.142857
    d = dice_coeff(pred, target).item()
    i = iou_score(pred, target).item()
    assert abs(d - 0.25) < 1e-4, d
    assert abs(i - 4 / 28) < 1e-4, i
    assert abs(d - i) > 1e-3, "Dice and IoU must differ under partial overlap"


def test_no_overlap_equal_but_zero():
    H = W = 8
    target = _box(1, H, W, 0, 3, 0, 3)
    pred = _box(1, H, W, 5, 8, 5, 8)          # disjoint -> inter = 0
    d = dice_coeff(pred, target).item()
    i = iou_score(pred, target).item()
    assert d < 1e-3 and i < 1e-3, (d, i)      # both ~0, and equal by construction


def test_perfect_overlap_is_one():
    target = _box(1, 8, 8, 1, 5, 1, 5)
    assert abs(dice_coeff(target, target).item() - 1.0) < 1e-4
    assert abs(iou_score(target, target).item() - 1.0) < 1e-4


def test_empty_target_is_ignored_for_localisation():
    H = W = 8
    empty = torch.zeros(1, 1, H, W)
    pred_empty = torch.zeros(1, 1, H, W)
    # Raw coeff rewards empty-vs-empty with 1.0 (the source of the inflated metric)...
    assert abs(dice_coeff(pred_empty, empty).item() - 1.0) < 1e-6
    # ...but segmentation_scores excludes it: nothing to score.
    s = segmentation_scores(pred_empty, empty, ignore_empty=True)
    assert s["count"] == 0, s


def test_predict_nothing_scores_zero_not_one():
    """A model that outputs all-zeros must score ~0 on lesion frames, not ~1."""
    H = W = 8
    # Batch: one background frame (empty) + one lesion frame; prediction is empty.
    target = torch.zeros(2, 1, H, W)
    target[1, :, 0:4, 0:4] = 1.0
    pred = torch.zeros(2, 1, H, W)
    s = segmentation_scores(pred, target, ignore_empty=True)
    assert s["count"] == 1, s                 # only the lesion frame is scored
    assert s["dice_sum"] < 1e-3, s            # and it scores ~0, exposing the collapse


def test_cross_batch_aggregation_is_exact():
    """Summing dice_sum/iou_sum/count across batches equals the pooled mean."""
    H = W = 8
    t = _box(3, H, W, 0, 4, 0, 4)
    p = _box(3, H, W, 2, 6, 2, 6)
    whole = segmentation_scores(p, t)
    a = segmentation_scores(p[:1], t[:1])
    b = segmentation_scores(p[1:], t[1:])
    pooled = (a["dice_sum"] + b["dice_sum"]) / (a["count"] + b["count"])
    assert abs(pooled - whole["dice_sum"] / whole["count"]) < 1e-6


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} metric tests passed.")


if __name__ == "__main__":
    main()
