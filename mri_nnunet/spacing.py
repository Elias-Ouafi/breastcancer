"""Choose the target spacing from the lesions, not from the dataset's median.

nnU-Net's default is the median spacing of the training set, which serves the organ. Here the
object is a lesion whose smallest box axis can be a centimetre, so the spacing is chosen so
that even the small ones keep at least ``spacing.min_voxels`` voxels on their shortest axis.

    python -m mri_nnunet spacing        # the distribution and the decision, no processing

The boxes' sizes are read from the ``box.json`` written at ingestion, in native voxels and
native spacing, so this runs before any resampling and does not depend on it.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import steps


def box_dimensions(out_dir):
    """``(patient_ids, extent_voxels (n, 3), native_spacing (n, 3))`` of the first box of each case."""
    ids, extents, spacings = [], [], []
    native = os.path.join(out_dir, "native")
    if not os.path.isdir(native):
        return ids, np.zeros((0, 3)), np.zeros((0, 3))
    for pid in sorted(os.listdir(native)):
        path = os.path.join(native, pid, "box.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            box = json.load(handle)
        for entry in box["boxes"]:
            ids.append(pid)
            extents.append(entry["ellipsoid"]["box_voxels"])
            spacings.append(box["native_spacing"])
    return ids, np.array(extents, float).reshape(-1, 3), np.array(spacings, float).reshape(-1, 3)


def describe(out_dir, settings):
    """The box-size distribution, the spacing it implies and how well that spacing serves it."""
    ids, extents, spacings = box_dimensions(out_dir)
    if not ids:
        raise FileNotFoundError(f"no ingested case with a box under {out_dir}; run the ingestion first")
    cfg = settings["spacing"]
    smallest = steps.smallest_axis_mm(extents, spacings)
    axes_mm = extents * spacings
    chosen, coverage = steps.auto_spacing(smallest, cfg["min_voxels"], cfg["percentile"],
                                          cfg["floor_mm"], cfg["ceil_mm"])
    native_median = float(np.median(spacings[:, :2]))
    return {
        "n_boxes": len(ids),
        "box_axis_mm": {axis: {"p05": float(np.percentile(axes_mm[:, i], 5)),
                               "median": float(np.median(axes_mm[:, i])),
                               "p95": float(np.percentile(axes_mm[:, i], 95))}
                        for i, axis in enumerate(("x", "y", "z"))},
        "smallest_axis_mm": {"min": float(smallest.min()), "p10": float(np.percentile(smallest, 10)),
                             "median": float(np.median(smallest)), "max": float(smallest.max())},
        "native_inplane_spacing_median_mm": native_median,
        "rule": f"{cfg['min_voxels']} voxels on the smallest axis for {100 - cfg['percentile']}% of lesions",
        "chosen_spacing_mm": chosen,
        "share_of_lesions_with_min_voxels": coverage,
        # What the alternative would have cost: the same statistic at the native in-plane spacing.
        "share_at_native_spacing": float((smallest / native_median >= cfg["min_voxels"]).mean()),
    }


def resolve(out_dir, settings):
    """The spacing (mm) to resample to: the configured number, or the one ``describe`` derives."""
    configured = settings["resample"]["spacing_mm"]
    if configured != "auto":
        return float(configured)
    report = describe(out_dir, settings)
    with open(os.path.join(out_dir, "spacing.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report["chosen_spacing_mm"]
