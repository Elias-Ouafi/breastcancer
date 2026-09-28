"""FROC: lesion sensitivity against the number of false positives per exam.

    python -m imaging.froc --predictions <dir> --labels <dir> --output reports/froc.json

Why this and not what `imaging.evaluate` reports
------------------------------------------------
`imaging.evaluate` answers "once shown a lesion slice, how well is the lesion outlined?"
(Dice) and counts false positives at **one** threshold, **per slice**. Three things make
that the wrong instrument for the question this project asks:

* **The target is a box.** The masks are ellipsoids inscribed in the published boxes, so a
  Dice of 1.0 would mean the model learnt to draw an ellipsoid, not to find a lesion. The
  criterion here is the one a radiologist would accept: the candidate's centre of mass
  falls inside the lesion's box.
* **A lesion is a 3D object.** Counting per slice scores one lesion once per slice it
  crosses, so a big lesion weighs more than a small one and a detection that drifts a slice
  is a miss and a false positive at once. Candidates and lesions here are 3D connected
  components.
* **One threshold is a choice, and a choice hides the trade-off.** The FROC sweeps every
  threshold and reports sensitivity at 0.5, 1, 2 and 4 false positives per exam, which is
  how detection is reported in breast imaging. The threshold stops being an assumption.

Everything is resampled over **patients** for the confidence intervals, for the reason
`imaging.evaluate.bootstrap_ci` gives: candidates inside one exam are not independent
evidence.

The core functions take numpy arrays and nothing else, so they run in CI without
SimpleITK, without a GPU and without the corpus; only the command line reads files.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FP_RATES = (0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------------- components

def label_components(mask):
    """Label the 6-connected foreground of a 3D boolean array. Returns ``(labels, count)``.

    ``labels`` is ``int32``, 0 on background and 1..count on the components.

    Written here rather than with ``scipy.ndimage.label`` because scipy is not a dependency
    of this project (``imaging.evaluate.connected_components`` made the same call in 2D).
    A voxel-by-voxel flood fill in Python would be minutes on a 300^3 volume, so this
    propagates the smallest id along the edges of the foreground graph with vectorised
    numpy: each round is one pass over the edge list, and the number of rounds is the
    longest path inside a component, not the number of voxels.
    """
    mask = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    if mask.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {mask.shape}")
    flat = mask.ravel()
    voxels = np.flatnonzero(flat)
    if voxels.size == 0:
        return np.zeros(mask.shape, dtype=np.int32), 0

    # Compact index: linear position in the volume -> 0..n-1 over foreground voxels only.
    compact = np.full(flat.size, -1, dtype=np.int64)
    compact[voxels] = np.arange(voxels.size)

    edges = []
    for axis in range(3):
        both = np.logical_and(np.take(mask, np.arange(mask.shape[axis] - 1), axis=axis),
                              np.take(mask, np.arange(1, mask.shape[axis]), axis=axis))
        if not both.any():
            continue
        lower = np.array(np.nonzero(both))          # index of the first voxel of each pair
        upper = lower.copy()
        upper[axis] += 1
        first = np.ravel_multi_index(tuple(lower), mask.shape)
        second = np.ravel_multi_index(tuple(upper), mask.shape)
        edges.append((compact[first], compact[second]))

    ids = np.arange(voxels.size, dtype=np.int64)
    if edges:
        left = np.concatenate([e[0] for e in edges])
        right = np.concatenate([e[1] for e in edges])
        while True:
            previous = ids
            ids = ids.copy()
            np.minimum.at(ids, left, previous[right])
            np.minimum.at(ids, right, previous[left])
            # Point every voxel at the smallest id its current id points at, which halves
            # the remaining chain length each round instead of shortening it by one.
            ids = ids[ids]
            if np.array_equal(ids, previous):
                break

    _, ids = np.unique(ids, return_inverse=True)
    labels = np.zeros(flat.size, dtype=np.int32)
    labels[voxels] = ids.astype(np.int32) + 1
    return labels.reshape(mask.shape), int(ids.max()) + 1


def _components(labels, count, values=None, spacing=None):
    """``[{voxels, centroid, bbox, score}]`` for each label, scored by ``values`` when given."""
    out = []
    flat_labels = labels.ravel()
    order = np.argsort(flat_labels, kind="stable")
    sorted_labels = flat_labels[order]
    starts = np.searchsorted(sorted_labels, np.arange(1, count + 1), side="left")
    ends = np.searchsorted(sorted_labels, np.arange(1, count + 1), side="right")
    flat_values = None if values is None else np.asarray(values).ravel()
    voxel_mm3 = float(np.prod(spacing)) if spacing is not None else None
    for start, end in zip(starts, ends):
        positions = order[start:end]
        coords = np.array(np.unravel_index(positions, labels.shape), dtype=float)
        entry = {
            "voxels": int(positions.size),
            "centroid": coords.mean(axis=1).tolist(),
            "bbox": [[int(c.min()), int(c.max())] for c in coords],
        }
        if voxel_mm3 is not None:
            entry["volume_mm3"] = round(entry["voxels"] * voxel_mm3, 2)
        if flat_values is not None:
            entry["score"] = float(flat_values[positions].max())
        out.append(entry)
    return out


def lesions(label_volume, spacing=None):
    """The reference lesions: 3D connected components of the ground-truth mask."""
    labels, count = label_components(np.asarray(label_volume) > 0)
    return _components(labels, count, spacing=spacing)


def candidates(probability, detection_threshold=0.1, min_voxels=10, spacing=None):
    """Candidate detections: components of ``probability >= detection_threshold``.

    Each candidate is scored by the **highest** probability inside it, which is the value a
    threshold sweep then compares against: raising the operating threshold removes whole
    candidates, it does not erode them. ``detection_threshold`` is deliberately low -- it
    only decides what counts as one candidate, while the sweep decides what is reported --
    and ``min_voxels`` drops specks that no reader would call a finding.
    """
    probability = np.asarray(probability, dtype=float)
    labels, count = label_components(probability >= detection_threshold)
    found = _components(labels, count, values=probability, spacing=spacing)
    return [c for c in found if c["voxels"] >= min_voxels]


# --------------------------------------------------------------------------- matching

def _inside(point, bbox):
    return all(lo <= p <= hi for p, (lo, hi) in zip(point, bbox))


def match(found, reference, criterion="centroid_in_box"):
    """Match candidates to lesions. Returns ``(hits, marks)``.

    ``hits`` is one bool per lesion (was it found by *some* candidate), ``marks`` one entry
    per candidate: ``{"score", "lesion"}`` with ``lesion`` the index it hit or ``None`` for a
    false positive.

    A second candidate on an already-detected lesion is **neither** a true nor a false
    positive -- it is the same finding reported twice, and counting it as an error would
    penalise a model for being sure. This is the usual FROC convention.

    ``criterion``:
    * ``centroid_in_box`` (default): the candidate's centre of mass falls inside the
      lesion's bounding box -- "it pointed at the right thing", the criterion the boxes
      actually support.
    * ``overlap``: the candidate shares at least one voxel with the lesion's box. Stricter
      about where the candidate is, looser about where its centre is; reported alongside so
      neither choice has to be taken on faith.
    """
    if criterion not in ("centroid_in_box", "overlap"):
        raise ValueError(f"unknown criterion {criterion!r}")
    hits = [False] * len(reference)
    marks = []
    for candidate in sorted(found, key=lambda c: -c["score"]):
        target = None
        for i, lesion in enumerate(reference):
            if criterion == "centroid_in_box":
                touching = _inside(candidate["centroid"], lesion["bbox"])
            else:
                touching = all(mine[0] <= theirs[1] and theirs[0] <= mine[1]
                               for mine, theirs in zip(candidate["bbox"], lesion["bbox"]))
            if touching:
                target = i
                break
        if target is not None:
            hits[target] = True
        marks.append({"score": float(candidate["score"]), "lesion": target})
    return hits, marks


def case_record(probability, label_volume, detection_threshold=0.1, min_voxels=10,
                spacing=None, criterion="centroid_in_box", case_id=None):
    """Everything one exam contributes to the curve, with no threshold fixed yet."""
    reference = lesions(label_volume, spacing=spacing)
    found = candidates(probability, detection_threshold, min_voxels, spacing=spacing)
    hits, marks = match(found, reference, criterion)
    del hits
    return {
        "case_id": case_id,
        "n_lesions": len(reference),
        "lesion_scores": [max([m["score"] for m in marks if m["lesion"] == i], default=None)
                          for i in range(len(reference))],
        "fp_scores": sorted((m["score"] for m in marks if m["lesion"] is None), reverse=True),
    }


# --------------------------------------------------------------------------- the curve

def curve(records):
    """The FROC curve: ``{thresholds, sensitivity, fp_per_exam}``, thresholds decreasing.

    A lesion counts as found at threshold ``t`` when the best candidate that hit it scored
    ``>= t``; a false positive counts when its candidate did. Both are read off the same
    sorted list of scores, so the curve is exact rather than sampled on a grid.
    """
    records = list(records)
    n_exams = len(records)
    if n_exams == 0:
        raise ValueError("no case to score")
    total_lesions = sum(r["n_lesions"] for r in records)
    lesion_scores = np.array([s for r in records for s in r["lesion_scores"] if s is not None],
                             dtype=float)
    fp_scores = np.array([s for r in records for s in r["fp_scores"]], dtype=float)

    thresholds = np.unique(np.concatenate([lesion_scores, fp_scores, [np.inf]]))[::-1]
    sensitivity, fp_per_exam = [], []
    for t in thresholds:
        found = int((lesion_scores >= t).sum())
        sensitivity.append(found / total_lesions if total_lesions else float("nan"))
        fp_per_exam.append(float((fp_scores >= t).sum()) / n_exams)
    return {
        "thresholds": [float(t) for t in thresholds],
        "sensitivity": sensitivity,
        "fp_per_exam": fp_per_exam,
        "n_exams": n_exams,
        "n_lesions": total_lesions,
    }


def sensitivity_at(records, fp_rates=DEFAULT_FP_RATES):
    """``{rate: sensitivity}``: the best sensitivity reachable at most ``rate`` FP per exam.

    ``nan`` when even the highest threshold already exceeds the rate -- the operating point
    does not exist on this data, which is a fact worth printing rather than rounding away.
    """
    points = curve(records)
    sens = np.asarray(points["sensitivity"], dtype=float)
    fps = np.asarray(points["fp_per_exam"], dtype=float)
    out = {}
    for rate in fp_rates:
        usable = sens[fps <= rate]
        out[str(rate)] = float(usable.max()) if usable.size else float("nan")
    return out


def bootstrap_sensitivity(records, fp_rates=DEFAULT_FP_RATES, n_resamples=2000, alpha=0.05,
                          seed=0):
    """Percentile bootstrap CI of :func:`sensitivity_at`, resampling **patients**.

    The whole curve is recomputed inside each resample: the false-positive rate is a
    property of the resampled set of exams, so re-reading a fixed curve would hold it
    constant and understate the interval.
    """
    records = list(records)
    point = sensitivity_at(records, fp_rates)
    rng = np.random.default_rng(seed)
    draws = {str(rate): [] for rate in fp_rates}
    for picks in rng.integers(0, len(records), size=(n_resamples, len(records))):
        resampled = [records[i] for i in picks]
        if sum(r["n_lesions"] for r in resampled) == 0:
            continue
        for key, value in sensitivity_at(resampled, fp_rates).items():
            if not np.isnan(value):
                draws[key].append(value)
    out = {}
    for rate in fp_rates:
        key = str(rate)
        values = np.asarray(draws[key], dtype=float)
        out[key] = {
            "sensitivity": point[key],
            "lo": float(np.percentile(values, 100 * alpha / 2)) if values.size else float("nan"),
            "hi": float(np.percentile(values, 100 * (1 - alpha / 2))) if values.size else float("nan"),
            "n_usable": int(values.size),
        }
    return out


def sensitivity_by_lesion_size(records, sizes_mm3, fp_rate=2.0, edges=(1000.0, 10000.0)):
    """Sensitivity at one FP rate, split by lesion volume -- small lesions are the hard ones.

    ``sizes_mm3`` is one list per record, in the order of its ``lesion_scores``. The
    threshold is the one the whole set reaches ``fp_rate`` at, so the groups are compared at
    the same operating point rather than each at its own.
    """
    points = curve(records)
    fps = np.asarray(points["fp_per_exam"], dtype=float)
    usable = np.flatnonzero(fps <= fp_rate)
    if usable.size == 0:
        return {}
    threshold = points["thresholds"][int(usable[np.argmax(np.asarray(points["sensitivity"])[usable])])]

    def band_of(size):
        for lo, hi in zip((0.0,) + tuple(edges), tuple(edges) + (float("inf"),)):
            if lo <= size < hi:
                return f"{lo:.0f}-{hi:.0f} mm3" if np.isfinite(hi) else f">= {lo:.0f} mm3"
        return "unknown"

    bands = {}
    for record, sizes in zip(records, sizes_mm3):
        for score, size in zip(record["lesion_scores"], sizes):
            bands.setdefault(band_of(size), []).append(score is not None and score >= threshold)
    return {name: {"sensitivity": float(np.mean(found)), "n_lesions": len(found)}
            for name, found in sorted(bands.items())}


# --------------------------------------------------------------------------- command line

def foreground_probability(array, lesion_class=1):
    """The lesion probability map from an array that may still carry its class axis.

    ``nnUNetv2_predict --save_probabilities`` writes ``(n_classes, Z, Y, X)``: background first,
    lesion second. Taking ``array[0]`` there would score the FROC on the background channel and
    report a model that finds nothing -- so the class axis is removed explicitly, and a volume
    that already has three dimensions is passed through untouched.
    """
    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[0] <= lesion_class:
            raise ValueError(f"no class {lesion_class} in a probability array of shape {array.shape}")
        return array[lesion_class]
    if array.ndim != 3:
        raise ValueError(f"expected a 3D volume or a 4D (class, z, y, x) array, got {array.shape}")
    return array


def _read_volume(path):
    """A prediction or label volume from ``.npy``/``.npz`` (numpy) or ``.nii.gz`` (SimpleITK)."""
    if path.endswith(".npy"):
        return foreground_probability(np.load(path)), None
    if path.endswith(".npz"):
        with np.load(path) as data:
            # "probabilities" is nnU-Net's own key; "probability" and the first array are the
            # shapes this repo's own exports take.
            key = next((k for k in ("probabilities", "probability") if k in data.files),
                       data.files[0])
            return foreground_probability(data[key]), None
    from mri_nnunet import sitk_io  # lazy: SimpleITK is the `nnunet` extra, not a base dependency

    image = sitk_io.read_nifti(path)
    return sitk_io.to_array(image), tuple(image.GetSpacing())[::-1]


def _case_id(name):
    """``Breast_MRI_037.nii.gz`` -> ``Breast_MRI_037`` (both extensions of a .nii.gz)."""
    return name.split(".")[0]


def _by_case(directory, preferred=(".npz", ".npy", ".nii.gz")):
    """``{case id: path}``, one file per case, the first extension of ``preferred`` that exists.

    A folder of nnU-Net predictions holds **two** files per case -- ``<case>.nii.gz`` (the
    labels it decided on) and ``<case>.npz`` (the probabilities ``--save_probabilities``
    wrote) -- plus a ``.pkl``. Pairing on file names would score every exam twice, once on a
    map whose only values are 0 and 1, and the thresholds swept over that are meaningless.
    The probabilities win because the FROC needs a score per candidate.
    """
    found = {}
    for name in sorted(os.listdir(directory)):
        extension = next((e for e in preferred if name.endswith(e)), None)
        if extension is None:
            continue
        case = _case_id(name)
        rank = preferred.index(extension)
        if case not in found or rank < found[case][0]:
            found[case] = (rank, os.path.join(directory, name))
    return {case: path for case, (_, path) in found.items()}


def _pairs(prediction_dir, label_dir):
    predictions = _by_case(prediction_dir)
    labels = _by_case(label_dir, preferred=(".nii.gz", ".npy", ".npz"))
    for case, path in sorted(predictions.items()):
        if case not in labels:
            log.warning("no label for %s, skipped", case)
            continue
        yield case, path, labels[case]


def run(args):
    records, sizes = [], []
    for case_id, prediction_path, label_path in _pairs(args.predictions, args.labels):
        probability, prediction_spacing = _read_volume(prediction_path)
        label_volume, label_spacing = _read_volume(label_path)
        # The label carries the geometry: a prediction saved as .npz (nnU-Net's
        # --save_probabilities) has none, and taking its missing spacing left every lesion
        # volume as nan, so the whole by-size breakdown collapsed into one "unknown" band.
        spacing = prediction_spacing if prediction_spacing is not None else label_spacing
        reference = lesions(label_volume, spacing=spacing)
        record = case_record(probability, label_volume, args.detection_threshold,
                             args.min_voxels, spacing, args.criterion, case_id)
        records.append(record)
        sizes.append([lesion.get("volume_mm3", float("nan")) for lesion in reference])
        log.info("%-20s %d lesion(s), %d false positive(s) above %.2f",
                 case_id, record["n_lesions"], len(record["fp_scores"]), args.detection_threshold)
    if not records:
        raise SystemExit(f"no prediction/label pair found under {args.predictions}")

    report = {
        "n_exams": len(records),
        "n_lesions": sum(r["n_lesions"] for r in records),
        "criterion": args.criterion,
        "detection_threshold": args.detection_threshold,
        "min_voxels": args.min_voxels,
        "sensitivity_at_fp": bootstrap_sensitivity(records, args.fp_rates, args.bootstrap),
        "by_lesion_size": sensitivity_by_lesion_size(records, sizes),
        "curve": curve(records),
        "per_case": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    log.info("FROC on %d exams, %d lesions (%s)", report["n_exams"], report["n_lesions"],
             args.criterion)
    for rate, entry in report["sensitivity_at_fp"].items():
        log.info("  %5s FP/exam   sensibilité %5.1f %%  [IC95 %.1f – %.1f]", rate,
                 100 * entry["sensitivity"], 100 * entry["lo"], 100 * entry["hi"])
    log.info(args.output)
    return report


def build_arg_parser():
    p = argparse.ArgumentParser(description="Lesion sensitivity against false positives per exam.")
    p.add_argument("--predictions", required=True,
                   help="folder of probability volumes (.nii.gz, .npy or .npz), one per exam")
    p.add_argument("--labels", required=True, help="folder of reference masks, same file names")
    p.add_argument("--output", default=os.path.join("reports", "froc.json"))
    p.add_argument("--criterion", choices=["centroid_in_box", "overlap"], default="centroid_in_box")
    p.add_argument("--detection-threshold", type=float, default=0.1,
                   help="probability above which voxels form a candidate; the sweep, not this, "
                        "decides what is reported")
    p.add_argument("--min-voxels", type=int, default=10,
                   help="candidates smaller than this are specks, not findings")
    p.add_argument("--fp-rates", type=float, nargs="*", default=list(DEFAULT_FP_RATES))
    p.add_argument("--bootstrap", type=int, default=2000)
    return p


if __name__ == "__main__":
    from logging_setup import setup_logging

    setup_logging(logfile="froc.log")
    run(build_arg_parser().parse_args())
