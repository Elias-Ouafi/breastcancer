"""The split has to keep the prevalence, and keep the old one reproducible.

Two obligations pull in opposite directions. A two-class corpus needs the class
balance held constant across train/val/test: a specificity or a PPV is only readable
next to the prevalence it was measured at, and on 24 cancer patients against 48
benign ones an unlucky seed can round a class out of a split entirely. But the
published DCE-MRI numbers (Dice 0.533 over 28 test patients) name specific patients,
and that corpus carries no labels -- so its allocation has to come out exactly as it
did before stratification existed.

These tests hold both ends: the stratified path preserves prevalence, and the
unlabelled path reproduces the previous algorithm bit for bit.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from imaging.dataset import split_npz_by_patient


def write_corpus(root, patients, series_per_patient=2):
    """``patients`` maps patient id -> label (or None for an unlabelled corpus)."""
    os.makedirs(root, exist_ok=True)
    for patient, label in patients.items():
        for s in range(series_per_patient):
            arrays = {
                "volume": np.zeros((2, 8, 8), dtype=np.float16),
                "mask": np.zeros((2, 8, 8), dtype=np.uint8),
                "crop_offset": np.zeros(3, dtype=np.int32),
                "case_id": np.asarray(patient),
            }
            if label is not None:
                arrays["label"] = np.asarray(label, dtype=np.uint8)
            np.savez_compressed(os.path.join(root, f"{patient}-s{s}.npz"), **arrays)
    return str(root)


def patients_of(paths):
    return {os.path.basename(p).split("-s")[0] for p in paths}


def prevalence(paths, labels):
    patients = patients_of(paths)
    return sum(labels[p] for p in patients) / len(patients)


# The measured BCS-DBT balance: 48 benign patients, 24 cancers.
REAL_BALANCE = {**{f"DBT-B{i:03d}": 0 for i in range(48)},
                **{f"DBT-C{i:03d}": 1 for i in range(24)}}


def test_every_split_keeps_the_corpus_prevalence(tmp_path):
    data_dir = write_corpus(tmp_path / "dbt", REAL_BALANCE)
    train, val, test = split_npz_by_patient(data_dir, val_frac=0.15, test_frac=0.15)

    corpus = 24 / 72
    for name, paths in (("train", train), ("val", val), ("test", test)):
        assert paths, f"{name} is empty"
        # Up to rounding on 11 and 11 patients: a third, give or take one patient.
        assert abs(prevalence(paths, REAL_BALANCE) - corpus) < 0.06, name


def test_no_patient_straddles_two_splits(tmp_path):
    data_dir = write_corpus(tmp_path / "dbt", REAL_BALANCE)
    train, val, test = split_npz_by_patient(data_dir)

    tr, va, te = patients_of(train), patients_of(val), patients_of(test)
    assert tr & va == set() and tr & te == set() and va & te == set()
    assert len(tr | va | te) == len(REAL_BALANCE)
    # Both files of a patient travel together.
    assert len(train) + len(val) + len(test) == 2 * len(REAL_BALANCE)


def test_a_rare_class_never_disappears_from_training(tmp_path):
    """Three cancers against twenty benigns is where an unstratified draw fails."""
    labels = {**{f"B{i:02d}": 0 for i in range(20)}, **{f"C{i}": 1 for i in range(3)}}
    data_dir = write_corpus(tmp_path / "rare", labels)

    for seed in range(8):
        train, _val, _test = split_npz_by_patient(data_dir, seed=seed)
        assert any(labels[p] == 1 for p in patients_of(train)), seed
        assert any(labels[p] == 0 for p in patients_of(train)), seed


def test_the_same_seed_gives_the_same_split(tmp_path):
    data_dir = write_corpus(tmp_path / "dbt", REAL_BALANCE)
    first = split_npz_by_patient(data_dir, seed=7)
    again = split_npz_by_patient(data_dir, seed=7)
    other = split_npz_by_patient(data_dir, seed=8)
    assert first == again
    assert first != other


def test_a_patient_with_both_labels_counts_as_cancer(tmp_path):
    """Never observed in this collection; it must not be averaged away in silence."""
    root = tmp_path / "mixed"
    write_corpus(root, {"P1": 0}, series_per_patient=1)
    np.savez_compressed(os.path.join(str(root), "P1-s1.npz"),
                        volume=np.zeros((2, 8, 8), dtype=np.float16),
                        mask=np.zeros((2, 8, 8), dtype=np.uint8),
                        crop_offset=np.zeros(3, dtype=np.int32),
                        case_id=np.asarray("P1"), label=np.asarray(1, dtype=np.uint8))
    write_corpus(root, {f"B{i}": 0 for i in range(8)}, series_per_patient=1)
    write_corpus(root, {f"C{i}": 1 for i in range(8)}, series_per_patient=1)

    with pytest.warns(UserWarning, match="labelled"):
        split_npz_by_patient(str(root), seed=3)


# --------------------------------------------------------------------------- #
# The corpus with no labels: the published DCE-MRI split must not move
# --------------------------------------------------------------------------- #

def previous_algorithm(keys, val_frac, test_frac, seed):
    """The allocation exactly as it was before stratification, for comparison."""
    keys = list(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_val = min(n_val, max(0, n - n_test - 1))
    n_test = min(n_test, max(0, n - n_val - 1))
    return (keys[n_test + n_val:], keys[n_test:n_test + n_val], keys[:n_test])


def test_an_unlabelled_corpus_reproduces_the_previous_allocation(tmp_path):
    """186 unlabelled patients, the DCE-MRI shape, and the seed the numbers used."""
    patients = {f"Breast_MRI_{i:03d}": None for i in range(1, 187)}
    data_dir = write_corpus(tmp_path / "mri", patients, series_per_patient=1)

    with pytest.warns(UserWarning, match="no exam-level label"):
        train, val, test = split_npz_by_patient(data_dir, seed=42)

    expected = previous_algorithm(sorted(patients), val_frac=0.15, test_frac=0.15, seed=42)
    assert [patients_of([p]).pop() for p in test] == expected[2]
    assert [patients_of([p]).pop() for p in val] == expected[1]
    assert [patients_of([p]).pop() for p in train] == expected[0]
    assert len(test) == 28  # the published DCE-MRI test split
