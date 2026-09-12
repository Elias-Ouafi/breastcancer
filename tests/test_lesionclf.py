"""The image side of step 2, and the metrics it is allowed to claim.

The metrics get as much attention here as the model, on purpose. This repository has
already published an accuracy measured before its own split and a "confidence" that
was a constant, and both survived because nothing checked what the number meant. So:
an AUC is checked against its definition rather than against itself, its interval is
checked to resample patients, and the PPV is checked to move with prevalence while
sensitivity and specificity do not -- which is the whole reason plan.md refuses to
quote a PPV without one.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from imaging.dataset import kfold_by_patient
from imaging.lesionclf import (
    LesionSliceDataset,
    build_arg_parser,
    cross_validate,
    lesion_slices,
    patient_scores,
)
from imaging.metrics import bootstrap_auc, operating_point, roc_auc


def write_case(directory, patient, label, series=0, depth=6, size=32,
               lesion_slices_count=2, with_label=True):
    """One preprocessed-looking ``.npz``: a crop, a mask on a few slices, a label."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(abs(hash((patient, series))) % 2**31)
    mask = np.zeros((depth, size, size), dtype=np.uint8)
    mask[1:1 + lesion_slices_count, 8:20, 8:20] = 1
    arrays = {
        "volume": rng.standard_normal((depth, size, size)).astype(np.float16),
        "mask": mask,
        "crop_offset": np.zeros(3, dtype=np.int32),
        "case_id": np.asarray(patient),
    }
    if with_label:
        arrays["label"] = np.asarray(label, dtype=np.uint8)
    path = os.path.join(directory, f"{patient}-s{series}.npz")
    np.savez_compressed(path, **arrays)
    return path


# --------------------------------------------------------------------------- #
# AUC against its definition
# --------------------------------------------------------------------------- #

def brute_force_auc(labels, scores):
    """P(positive scores above negative), ties as half -- the definition, spelled out."""
    pos = [s for y, s in zip(labels, scores) if y > 0]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@pytest.mark.parametrize("labels,scores,expected", [
    ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),     # perfectly separated
    ([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], 0.0),     # perfectly wrong
    ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5),     # no information at all
])
def test_the_auc_matches_the_textbook_cases(labels, scores, expected):
    assert roc_auc(labels, scores) == pytest.approx(expected)


def test_the_auc_matches_a_brute_force_count_including_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        labels = rng.integers(0, 2, 30)
        if labels.sum() in (0, 30):
            continue
        # Coarse scores, so ties happen often enough to be tested.
        scores = rng.integers(0, 5, 30) / 4
        assert roc_auc(labels, scores) == pytest.approx(brute_force_auc(labels, scores))


def test_a_single_class_has_no_auc_rather_than_a_chance_level_one():
    """0.5 would read as "no better than chance"; there is simply nothing to measure."""
    assert np.isnan(roc_auc([1, 1, 1], [0.2, 0.5, 0.9]))
    assert np.isnan(bootstrap_auc([0, 0], [0.1, 0.2], n_resamples=10)["auc"])


def test_the_interval_brackets_the_estimate_and_says_what_carries_it():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 40 + [1] * 20)
    scores = np.concatenate([rng.normal(0.4, 0.2, 40), rng.normal(0.6, 0.2, 20)])

    ci = bootstrap_auc(labels, scores, n_resamples=2000, seed=3)
    assert ci["lo"] <= ci["auc"] <= ci["hi"]
    assert ci["n"] == 60 and ci["n_positive"] == 20
    assert ci["n_usable"] > 1900          # a corpus this size loses almost no resample


def test_a_smaller_corpus_gives_a_wider_interval():
    """The reason this project never quotes a number without one."""
    rng = np.random.default_rng(2)

    def width(n):
        labels = np.array([0] * n + [1] * n)
        scores = np.concatenate([rng.normal(0.4, 0.2, n), rng.normal(0.6, 0.2, n)])
        ci = bootstrap_auc(labels, scores, n_resamples=1500, seed=5)
        return ci["hi"] - ci["lo"]

    assert width(10) > width(100)


# --------------------------------------------------------------------------- #
# The operating point, and what prevalence does to it
# --------------------------------------------------------------------------- #

def test_the_operating_point_counts_what_it_says():
    labels = [1, 1, 1, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1]
    point = operating_point(labels, scores, threshold=0.5)

    assert point["counts"] == {"tp": 2, "fp": 1, "tn": 3, "fn": 1}
    assert point["sensitivity"] == pytest.approx(2 / 3)
    assert point["specificity"] == pytest.approx(3 / 4)
    assert point["ppv"] == pytest.approx(2 / 3)
    assert point["prevalence"] == pytest.approx(3 / 7)


def test_the_ppv_moves_with_prevalence_while_sensitivity_does_not():
    """Same model, two populations: the PPV is a property of the mix, not the model."""
    rich = operating_point([1] * 20 + [0] * 20, [0.9] * 18 + [0.1] * 2 + [0.9] * 2 + [0.1] * 18)
    poor = operating_point([1] * 2 + [0] * 200,
                           [0.9, 0.9] + [0.9] * 20 + [0.1] * 180)

    assert rich["sensitivity"] == pytest.approx(0.9)
    assert poor["sensitivity"] == pytest.approx(1.0)
    assert rich["ppv"] > poor["ppv"]            # 0.90 against 0.09
    assert poor["prevalence"] < rich["prevalence"]


# --------------------------------------------------------------------------- #
# Folds over patients
# --------------------------------------------------------------------------- #

BALANCE = {**{f"B{i:02d}": 0 for i in range(20)}, **{f"C{i:02d}": 1 for i in range(12)}}


def build_corpus(root, balance=BALANCE, series_per_patient=1):
    for patient, label in balance.items():
        for s in range(series_per_patient):
            write_case(str(root), patient, label, series=s)
    return str(root)


def patients_in(paths):
    return {os.path.basename(p).split("-s")[0] for p in paths}


def test_every_patient_is_held_out_exactly_once(tmp_path):
    data_dir = build_corpus(tmp_path / "dbt", series_per_patient=2)
    folds = list(kfold_by_patient(data_dir, n_splits=4))

    held = [patients_in(h) for _t, h in folds]
    assert len(folds) == 4
    assert set().union(*held) == set(BALANCE)
    assert sum(len(h) for h in held) == len(BALANCE)      # no patient held out twice


def test_a_fold_never_trains_on_a_patient_it_scores(tmp_path):
    data_dir = build_corpus(tmp_path / "dbt", series_per_patient=2)
    for train, heldout in kfold_by_patient(data_dir, n_splits=4):
        assert patients_in(train) & patients_in(heldout) == set()


def test_each_fold_keeps_the_corpus_prevalence(tmp_path):
    data_dir = build_corpus(tmp_path / "dbt")
    corpus = 12 / 32
    for _train, heldout in kfold_by_patient(data_dir, n_splits=4):
        held = patients_in(heldout)
        share = sum(BALANCE[p] for p in held) / len(held)
        assert abs(share - corpus) < 0.13     # 8 patients per fold: 3/8 against 12/32


def test_an_unlabelled_corpus_still_folds_but_says_it_cannot_stratify(tmp_path):
    root = tmp_path / "unlabelled"
    for i in range(8):
        write_case(str(root), f"P{i}", 0, with_label=False)

    with pytest.warns(UserWarning, match="no exam-level label"):
        folds = list(kfold_by_patient(str(root), n_splits=4))
    assert sum(len(patients_in(h)) for _t, h in folds) == 8


# --------------------------------------------------------------------------- #
# What the model is fed, and how a patient's slices become one score
# --------------------------------------------------------------------------- #

def test_only_lesion_bearing_slices_are_used(tmp_path):
    """The crop's other slices show the same region without the lesion in it."""
    path = write_case(str(tmp_path), "P1", 1, depth=6, lesion_slices_count=2)
    images, patient, label = lesion_slices(path, image_size=16)

    assert images.shape == (2, 16, 16)       # 2 of the 6 slices carry mask
    assert (patient, label) == ("P1", 1)


def test_a_volume_without_a_label_is_refused_by_name(tmp_path):
    path = write_case(str(tmp_path), "P1", 1, with_label=False)
    with pytest.raises(KeyError, match="label"):
        lesion_slices(path, image_size=16)


def test_a_patient_score_is_the_mean_over_its_slices(tmp_path):
    write_case(str(tmp_path), "P1", 1, series=0, lesion_slices_count=3)
    write_case(str(tmp_path), "P1", 1, series=1, lesion_slices_count=1)
    write_case(str(tmp_path), "P2", 0, series=0, lesion_slices_count=2)
    paths = sorted(os.path.join(str(tmp_path), f) for f in os.listdir(str(tmp_path)))
    dataset = LesionSliceDataset(paths, image_size=16)

    import torch

    class Ramp(torch.nn.Module):
        """Scores each slice by its position, so the mean is arithmetic and known."""

        def __init__(self):
            super().__init__()
            self.seen = 0

        def forward(self, x):
            out = torch.arange(self.seen, self.seen + x.shape[0], dtype=torch.float32)
            self.seen += x.shape[0]
            return out[:, None]

    scores, labels = patient_scores(Ramp(), torch.device("cpu"), dataset)
    assert labels == {"P1": 1, "P2": 0}
    # P1 owns 4 slices, P2 owns 2; the dataset keeps the order of `paths`.
    expected_p1 = np.mean([1 / (1 + np.exp(-i)) for i in range(4)])
    assert scores["P1"] == pytest.approx(expected_p1, abs=1e-6)


def test_the_whole_cross_validation_runs_and_reports_what_it_measured(tmp_path, monkeypatch):
    """An integration check of the plumbing: two folds, one epoch, random pixels."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    data_dir = build_corpus(tmp_path / "dbt",
                            balance={f"B{i}": 0 for i in range(4)} | {f"C{i}": 1 for i in range(4)})

    args = build_arg_parser().parse_args([])
    args.data_dir = data_dir
    args.output_dir = str(tmp_path / "out")
    args.folds, args.epochs, args.image_size, args.bootstrap = 2, 1, 16, 50

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = cross_validate(args)

    assert report["corpus"]["patients"] == 8
    assert report["corpus"]["cancer_patients"] == 4
    assert report["auc_patient"]["n"] == 8
    assert report["operating_point_0.5"]["prevalence"] == pytest.approx(0.5)
    assert "not_a_detection_model" in report
    assert os.path.exists(os.path.join(args.output_dir, "cv_report.json"))
    assert os.path.exists(os.path.join(args.output_dir, "cv_predictions.csv"))
