"""Who is in the test set, and which fold each remaining patient trains in.

    python -m mri_nnunet splits            # write splits.json, print the counts

Two decisions live here, and both are taken **once, before any training**:

* **A held-out test set.** nnU-Net's own 5-fold cross-validation reports the validation
  folds, and every choice made while looking at them (which configuration, which epoch,
  which post-processing -- nnU-Net picks its post-processing on the cross-validation)
  is a choice fitted to those patients. A number quoted from them is not a number on
  unseen data. The test patients are set aside here, exported to ``imagesTs`` rather
  than ``imagesTr``, and nnU-Net never sees them until the model is frozen.
* **The folds themselves.** Left to itself nnU-Net draws its 5 folds at random. Here they
  are drawn stratified by **scanner** (manufacturer, model, field strength), the only
  acquisition variable Duke actually publishes: ``InstitutionName`` is empty and the study
  date is anonymised to 1990-01-01. Without that, a fold can end up with most of one
  scanner's patients and the spread across folds measures the draw as much as the model.

Both are written to ``splits.json`` in silver (the decision, with the counts per scanner
that justify it) and to ``splits_final.json`` in the nnU-Net layout, which is the file
nnU-Net reads instead of drawing its own folds.

Allocation is systematic sampling inside each scanner group: the group is shuffled with
the seeded generator, then walked with a phase drawn once per group. A group of three
patients therefore contributes to the test set about 15 % of the time rather than never
(what a plain ``floor(0.15 * 3)`` would give) or always. Nothing here looks at an image,
so re-running it with the same seed and the same registry gives the same split.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import pipeline, registry

SPLITS_FILENAME = "splits.json"
NNUNET_SPLITS_FILENAME = "splits_final.json"
UNKNOWN_SCANNER = "unknown"


def scanner_of(row):
    """The stratification key of one registry row: manufacturer, model and field strength."""
    scanner = (row.get("scanner") or "").strip()
    field = (row.get("field_strength_t") or "").strip()
    if not scanner and not field:
        return UNKNOWN_SCANNER
    return f"{scanner} {field}T".strip()


def scanner_groups(silver_dir, cases):
    """``{case_id: scanner key}`` from ``registry.csv``; cases with no row are ``unknown``."""
    rows = {r["patient_id"]: r for r in registry.read_csv(os.path.join(silver_dir, "registry.csv"))}
    return {case: scanner_of(rows[case]) if case in rows else UNKNOWN_SCANNER for case in cases}


def _systematic(members, fraction, rng):
    """Take about ``fraction`` of ``members``, spread evenly, with a random phase.

    The phase is what lets a small group contribute proportionally *on average*: with
    three members and 15 %, ``floor`` arithmetic alone would never pick one.
    """
    picked, accumulated = [], float(rng.random())
    for member in members:
        accumulated += fraction
        if accumulated >= 1.0:
            accumulated -= 1.0
            picked.append(member)
    return picked


def assign(cases, groups, test_frac=0.15, folds=5, seed=42):
    """Split ``cases`` into a test set and ``folds`` stratified cross-validation folds.

    Returns the decision as a dict: ``test``, ``trainval``, ``folds`` (nnU-Net's own
    ``{"train", "val"}`` shape) and ``strata``, the count per scanner in each part, which
    is what makes "stratified" checkable rather than claimed.
    """
    cases = sorted(cases)
    if len(cases) < folds + 1:
        raise ValueError(f"{len(cases)} cases cannot fill {folds} folds and a test set")
    if not 0.0 <= test_frac < 1.0:
        raise ValueError(f"test_frac must be in [0, 1), not {test_frac}")

    rng = np.random.default_rng(seed)
    by_group = {}
    for case in cases:
        by_group.setdefault(groups.get(case, UNKNOWN_SCANNER), []).append(case)

    test, fold_members = [], [[] for _ in range(folds)]
    for group in sorted(by_group):
        members = list(by_group[group])
        rng.shuffle(members)
        chosen = set(_systematic(members, test_frac, rng))
        test += sorted(chosen)
        # Deal the rest round-robin from a random starting fold, so no fold is
        # systematically the one that gets the leftover of every scanner.
        start = int(rng.integers(0, folds))
        for i, case in enumerate([m for m in members if m not in chosen]):
            fold_members[(start + i) % folds].append(case)

    trainval = sorted(case for fold in fold_members for case in fold)
    if not trainval:
        raise ValueError("every case landed in the test set; lower test_frac")

    strata = {}
    for group in sorted(by_group):
        members = set(by_group[group])
        strata[group] = {
            "total": len(members),
            "test": len(members & set(test)),
            "folds": [len(members & set(fold)) for fold in fold_members],
        }

    return {
        "seed": int(seed),
        "test_frac": float(test_frac),
        "n_folds": int(folds),
        "stratified_by": "scanner (manufacturer, model, field strength)",
        "test": sorted(test),
        "trainval": trainval,
        "folds": [{"train": sorted(set(trainval) - set(fold)), "val": sorted(fold)}
                  for fold in fold_members],
        "strata": strata,
    }


def check(assignment):
    """Raise unless the split is usable: no overlap, nothing lost, no empty fold.

    A split is read by another program hours later, so it is verified where it is
    written rather than trusted: a patient in both the test set and a training fold
    would inflate every number downstream and leave no trace.
    """
    test, trainval = set(assignment["test"]), set(assignment["trainval"])
    if test & trainval:
        raise ValueError(f"{len(test & trainval)} case(s) in both the test set and training")
    seen = set()
    for i, fold in enumerate(assignment["folds"]):
        val = set(fold["val"])
        if not val:
            raise ValueError(f"fold {i} has an empty validation set")
        if val & seen:
            raise ValueError(f"fold {i} validates on cases another fold already validated")
        if set(fold["train"]) | val != trainval:
            raise ValueError(f"fold {i} does not cover exactly the training cases")
        if set(fold["train"]) & val:
            raise ValueError(f"fold {i} trains on its own validation cases")
        seen |= val
    if seen != trainval:
        raise ValueError(f"{len(trainval - seen)} training case(s) are in no validation fold")
    return True


def nnunet_splits(assignment):
    """The ``splits_final.json`` payload: one ``{"train", "val"}`` entry per fold."""
    return [{"train": list(fold["train"]), "val": list(fold["val"])} for fold in assignment["folds"]]


def write(silver_dir, assignment):
    """Write ``splits.json`` beside the corpus. Returns its path."""
    check(assignment)
    path = os.path.join(silver_dir, SPLITS_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(assignment, handle, indent=2)
    return path


def read(silver_dir):
    """The written split, or ``None`` when the corpus has none yet."""
    path = os.path.join(silver_dir, SPLITS_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build(silver_dir, settings, cases=None):
    """Assign every fully built case and write the decision. Returns ``(assignment, path)``."""
    cases = sorted(cases if cases is not None else pipeline.held_cases(silver_dir, settings))
    if not cases:
        raise FileNotFoundError(f"no built case under {silver_dir}; run the build first")
    cfg = settings.get("splits", {})
    assignment = assign(cases, scanner_groups(silver_dir, cases),
                        test_frac=cfg.get("test_frac", 0.15), folds=cfg.get("n_folds", 5),
                        seed=cfg.get("seed", settings.get("seed", 42)))
    return assignment, write(silver_dir, assignment)
