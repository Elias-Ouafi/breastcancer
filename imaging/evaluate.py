"""Honest evaluation of a trained lesion-localisation U-Net on the held-out split.

    python -m imaging.evaluate --data-dir data/preprocessed_data/dce_mri_p2 \
        --checkpoint models/dce_mri_p2_negfix/unet_best.pt \
        --output-dir models/dce_mri_p2_negfix

``imaging.train`` reports a single mean Dice over lesion-bearing slices. That number
is real but it answers only one question ("once shown a lesion slice, how well is the
lesion outlined?") and it comes with no uncertainty, which makes it uncitable: on a
~28-patient test split, a mean without an interval says nothing about what a second
sample of patients would give. This module adds what a sceptical reader will ask for:

* **Bootstrap CI.** Resamples *patients* (not slices -- slices within a patient are
  strongly correlated, so resampling them would give a falsely narrow interval) and
  reports the 2.5th/97.5th percentiles of the patient-mean Dice/IoU.
* **Lesion-level sensitivity.** Dice says how well a lesion is outlined; it does not
  say whether the lesion was *found*. A slice counts as detected when the prediction
  overlaps the ground-truth box (IoU >= ``--hit-iou``), and separately when the
  largest predicted component's centroid falls inside it -- the looser criterion a
  radiologist would accept for "it pointed at the right thing".
* **False positives per volume.** Measured over *every* slice of each test volume,
  including lesion-free ones, which is the regime a real upload is in. This is the
  number the training metric structurally cannot see: ``positive_only=True`` never
  shows the model a background slice at eval time.
* **Inference timing.** Per slice and per whole volume, on the actual device, so the
  "under 10 s per volume" claim is measured rather than asserted.

Everything is written to ``eval_report.json`` (summary) and ``eval_per_patient.csv``
(one row per patient, so any number here can be traced back).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

try:  # allow both "python -m imaging.evaluate" and direct execution
    from .dataset import split_npz_by_patient
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset import split_npz_by_patient

import config  # noqa: E402 - repo root, importable under both invocations
from logging_setup import setup_logging

log = logging.getLogger(__name__)


def connected_components(mask, min_size=1):
    """Label 4-connected foreground components. Returns a list of (row, col) arrays.

    Implemented here rather than with ``scipy.ndimage.label`` because scipy is not a
    dependency of this project and pulling one in for a flood fill over a 256x256
    boolean array is not worth it.
    """
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask)
    height, width = mask.shape
    components = []
    for start_r, start_c in zip(*np.nonzero(mask)):
        if seen[start_r, start_c]:
            continue
        queue = deque([(start_r, start_c)])
        seen[start_r, start_c] = True
        pixels = []
        while queue:
            r, c = queue.popleft()
            pixels.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        if len(pixels) >= min_size:
            components.append(np.array(pixels))
    return components


def _box(mask):
    """Axis-aligned ``(r0, c0, r1, c1)`` bounds of the True pixels, or None if empty."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _iou(pred, target):
    """Plain IoU between two boolean arrays; 0.0 when the union is empty."""
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(inter / union) if union else 0.0


def _dice(pred, target):
    """Plain Dice between two boolean arrays; 0.0 when both are empty."""
    total = pred.sum() + target.sum()
    return float(2.0 * np.logical_and(pred, target).sum() / total) if total else 0.0


def _predict_volume(model, device, volume, image_size, threshold, batch_size=16):
    """Score every slice of ``volume``. Returns (binary masks at native size, seconds).

    Predictions are upsampled back to the slice's own resolution so overlap against
    the stored mask is measured in the mask's own pixels, matching what
    ``inference.predict_dce_mri`` returns to the app.
    """
    depth, height, width = volume.shape
    out = np.zeros((depth, height, width), dtype=bool)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, depth, batch_size):
            chunk = torch.from_numpy(volume[lo:lo + batch_size]).to(device)[:, None]
            chunk = F.interpolate(chunk, size=(image_size, image_size),
                                  mode="bilinear", align_corners=False)
            probs = torch.sigmoid(model(chunk))
            probs = F.interpolate(probs, size=(height, width),
                                  mode="bilinear", align_corners=False)
            out[lo:lo + batch_size] = (probs[:, 0] >= threshold).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


def evaluate_patient(model, device, path, image_size, threshold, hit_iou, min_component):
    """Score one preprocessed volume. Returns a per-patient metrics dict."""
    with np.load(path) as data:
        volume = data["volume"].astype(np.float32)
        mask = data["mask"] > 0
        case_id = str(data["case_id"]) if "case_id" in data.files else os.path.basename(path)

    pred, seconds = _predict_volume(model, device, volume, image_size, threshold)

    positive = mask.reshape(mask.shape[0], -1).any(axis=1)
    dices, ious, hits_iou, hits_centroid = [], [], [], []
    for z in np.nonzero(positive)[0]:
        p, t = pred[z], mask[z]
        dices.append(_dice(p, t))
        ious.append(_iou(p, t))
        hits_iou.append(_iou(p, t) >= hit_iou)

        # Looser "did it point at the right thing" criterion: the biggest predicted
        # blob's centre of mass lands inside the ground-truth box. A model can score a
        # poor Dice (boxes are coarse targets) while still localising correctly.
        components = connected_components(p, min_size=min_component)
        hit = False
        target_box = _box(t)
        if components and target_box is not None:
            biggest = max(components, key=len)
            cr, cc = biggest[:, 0].mean(), biggest[:, 1].mean()
            r0, c0, r1, c1 = target_box
            hit = (r0 <= cr <= r1) and (c0 <= cc <= c1)
        hits_centroid.append(hit)

    # False positives are counted on ground-truth-negative slices only: any predicted
    # component there is, by construction, a false alarm. This is the regime a real
    # full-volume upload is in and the one the training metric never looks at.
    negative_idx = np.nonzero(~positive)[0]
    fp_components = 0
    fp_slices = 0
    for z in negative_idx:
        components = connected_components(pred[z], min_size=min_component)
        if components:
            fp_slices += 1
            fp_components += len(components)

    return {
        "case_id": case_id,
        "n_slices": int(mask.shape[0]),
        "n_positive_slices": int(positive.sum()),
        "n_negative_slices": int(len(negative_idx)),
        "dice": float(np.mean(dices)) if dices else float("nan"),
        "iou": float(np.mean(ious)) if ious else float("nan"),
        "sensitivity_iou": float(np.mean(hits_iou)) if hits_iou else float("nan"),
        "sensitivity_centroid": float(np.mean(hits_centroid)) if hits_centroid else float("nan"),
        "fp_slices": fp_slices,
        "fp_components": fp_components,
        "seconds": seconds,
    }


def bootstrap_ci(values, n_resamples=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of ``values`` (one entry per patient).

    Patients are the resampling unit on purpose: slices inside a patient share
    anatomy, lesion and acquisition, so treating them as independent would shrink the
    interval to a width the data does not support.
    """
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[picks].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.percentile(means, 100 * alpha / 2)),
        "hi": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n": int(values.size),
    }


def run(args):
    from inference import load_unet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_unet(args.checkpoint, base=args.base_channels, device=device)
    log.info(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_paths, val_paths, test_paths = split_npz_by_patient(
        args.data_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    paths = {"test": test_paths, "val": val_paths, "train": train_paths}[args.split]
    log.info(f"Split '{args.split}': {len(paths)} patients "
          f"(train {len(train_paths)} / val {len(val_paths)} / test {len(test_paths)})\n")

    rows = []
    for i, path in enumerate(paths, 1):
        row = evaluate_patient(model, device, path, args.image_size, args.threshold,
                               args.hit_iou, args.min_component)
        rows.append(row)
        log.info(f"[{i:3d}/{len(paths)}] {row['case_id']:<18} "
              f"Dice {row['dice']:.3f}  sens(IoU) {row['sensitivity_iou']:.2f}  "
              f"sens(centre) {row['sensitivity_centroid']:.2f}  "
              f"FP {row['fp_components']:>5} sur {row['n_negative_slices']:>4} coupes  "
              f"{row['seconds']:.2f}s")

    if not rows:
        raise SystemExit("Empty split -- nothing to evaluate.")

    seconds = [r["seconds"] for r in rows]
    slices = sum(r["n_slices"] for r in rows)
    summary = {
        # Repo-relative: this report is committed, and an absolute path would bake the
        # machine that produced it (and its home directory) into a public artefact.
        "checkpoint": os.path.relpath(args.checkpoint, config.ROOT).replace(os.sep, "/"),
        "data_dir": os.path.relpath(args.data_dir, config.ROOT).replace(os.sep, "/"),
        "split": args.split,
        "n_patients": len(rows),
        "n_slices": slices,
        "threshold": args.threshold,
        "hit_iou": args.hit_iou,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dice": bootstrap_ci([r["dice"] for r in rows], args.bootstrap, seed=args.seed),
        "iou": bootstrap_ci([r["iou"] for r in rows], args.bootstrap, seed=args.seed),
        "sensitivity_iou": bootstrap_ci([r["sensitivity_iou"] for r in rows],
                                        args.bootstrap, seed=args.seed),
        "sensitivity_centroid": bootstrap_ci([r["sensitivity_centroid"] for r in rows],
                                             args.bootstrap, seed=args.seed),
        "fp_components_per_volume": bootstrap_ci([float(r["fp_components"]) for r in rows],
                                                 args.bootstrap, seed=args.seed),
        "fp_slice_rate": bootstrap_ci(
            [r["fp_slices"] / r["n_negative_slices"] if r["n_negative_slices"] else float("nan")
             for r in rows], args.bootstrap, seed=args.seed),
        "seconds_per_volume": {
            "mean": float(np.mean(seconds)),
            "median": float(np.median(seconds)),
            "max": float(np.max(seconds)),
        },
        "ms_per_slice": float(1000 * sum(seconds) / slices),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "eval_report.json")
    csv_path = os.path.join(args.output_dir, "eval_per_patient.csv")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(key, scale=1.0, unit=""):
        c = summary[key]
        return f"{c['mean'] * scale:.3f}{unit}  [IC95 {c['lo'] * scale:.3f} – {c['hi'] * scale:.3f}]"

    log.info(f"{'=' * 74}")
    log.info(f"Split {args.split} — {summary['n_patients']} patients, {slices} coupes")
    log.info(f"{'=' * 74}")
    log.info("Sur les coupes contenant une lésion (ce que mesurait déjà l'entraînement)")
    log.info(f"  Dice                        {fmt('dice')}")
    log.info(f"  IoU                         {fmt('iou')}")
    log.info(f"  Sensibilité (IoU>={args.hit_iou})      {fmt('sensitivity_iou', 100, '%')}")
    log.info(f"  Sensibilité (centre visé)   {fmt('sensitivity_centroid', 100, '%')}")
    log.info("Sur le volume entier (le régime d'un vrai upload)")
    log.info(f"  Faux positifs / volume      {fmt('fp_components_per_volume')}")
    log.info(f"  Coupes saines avec alarme   {fmt('fp_slice_rate', 100, '%')}")
    log.info("Temps d'inférence")
    log.info(f"  Par volume                  {summary['seconds_per_volume']['mean']:.2f} s "
          f"(médiane {summary['seconds_per_volume']['median']:.2f}, "
          f"max {summary['seconds_per_volume']['max']:.2f})")
    log.info(f"  Par coupe                   {summary['ms_per_slice']:.1f} ms")
    log.info(f"{json_path}\n{csv_path}")
    return summary


def build_arg_parser():
    p = argparse.ArgumentParser(description="Evaluate a lesion-localisation U-Net honestly.")
    p.add_argument("--data-dir", default=config.DCE_MRI_PREPROCESSED_DIR)
    p.add_argument("--checkpoint", default=config.DCE_MRI_UNET_CKPT)
    p.add_argument("--output-dir", default=config.DCE_MRI_MODEL_DIR)
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hit-iou", type=float, default=0.1,
                   help="IoU above which a lesion slice counts as detected. 0.1 is the "
                        "usual bar for box-level localisation -- the training masks are "
                        "boxes, so demanding a high IoU would measure box-fitting, not "
                        "finding.")
    p.add_argument("--min-component", type=int, default=10,
                   help="Predicted blobs smaller than this many pixels are ignored, so a "
                        "handful of stray pixels is not counted as a false lesion.")
    p.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the training seed, or the 'held-out' split will "
                        "contain patients the model was trained on.")
    return p


if __name__ == "__main__":
    setup_logging(logfile="evaluate.log")
    run(build_arg_parser().parse_args())
