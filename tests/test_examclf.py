"""The exam-level decision head: max-pooling MIL, and the aggregation it depends on.

The label is exam-level ("this exam has a cancer somewhere"), but the signal is not:
most slices of a cancer exam show nothing. Averaging over slices -- what ``lesionclf``
does, correctly, for a corpus already cropped to the lesion -- would wash out the
minority that matters here. So a bag's score is the **max** over its slices, and that
is the one property these tests protect: a single suspicious slice must be able to
carry the whole exam, and a whole patient must be carried by their single most
suspicious exam, not diluted by the other views.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from imaging.exambank import ExamBank, build_exam_bank
from imaging.examclf import (
    ExamBagDataset,
    build_arg_parser,
    cross_validate,
    exam_scores,
    pool_to_patient,
)
from imaging.metrics import bootstrap_auc, roc_auc


def write_exam(directory, patient, label, exam="s0", depth=8, size=16):
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(abs(hash((patient, exam))) % 2**31)
    mask = np.zeros((depth, size, size), dtype=np.uint8)
    if label:
        mask[2, 4:8, 4:8] = 1
    volume = rng.standard_normal((depth, size, size)).astype(np.float16)
    path = os.path.join(directory, f"{patient}-{exam}.npz")
    np.savez_compressed(path, volume=volume, mask=mask,
                        crop_offset=np.zeros(3, dtype=np.int32),
                        case_id=np.asarray(patient), label=np.asarray(label, dtype=np.uint8))
    return path


# --------------------------------------------------------------------------- #
# Bag sampling
# --------------------------------------------------------------------------- #

def test_a_bag_always_has_exactly_k_slices_even_from_a_shorter_exam(tmp_path):
    """Sampling with replacement when depth < k, so every bag is the same shape."""
    path = write_exam(str(tmp_path / "src"), "DBT-P1", label=1, depth=3)
    bank = ExamBank(build_exam_bank([path], str(tmp_path / "bank"), image_size=8))

    dataset = ExamBagDataset(bank, ["DBT-P1-s0"], k=10, seed=0)
    imgs, label = dataset[0]
    assert imgs.shape == (10, 1, 8, 8)
    assert label.item() == 1


def test_a_bag_never_mixes_two_exams(tmp_path):
    root = str(tmp_path / "src")
    write_exam(root, "DBT-P1", label=0, exam="rmlo", depth=6)
    write_exam(root, "DBT-P2", label=1, exam="lcc", depth=6)
    paths = [os.path.join(root, f) for f in os.listdir(root)]
    bank = ExamBank(build_exam_bank(paths, str(tmp_path / "bank"), image_size=8))

    dataset = ExamBagDataset(bank, ["DBT-P1-rmlo", "DBT-P2-lcc"], k=4, seed=1)
    for i, exam in enumerate(["DBT-P1-rmlo", "DBT-P2-lcc"]):
        _, label = dataset[i]
        assert label.item() == bank.exam_label[exam]


def test_pos_weight_reflects_the_bags_this_dataset_draws_from(tmp_path):
    root = str(tmp_path / "src")
    write_exam(root, "DBT-P1", label=0)
    write_exam(root, "DBT-P2", label=0)
    write_exam(root, "DBT-P3", label=1)
    paths = [os.path.join(root, f) for f in os.listdir(root)]
    bank = ExamBank(build_exam_bank(paths, str(tmp_path / "bank"), image_size=8))

    dataset = ExamBagDataset(bank, bank.unique_exams(), k=4)
    assert dataset.pos_weight() == pytest.approx(2.0)  # 2 negative / 1 positive


# --------------------------------------------------------------------------- #
# Aggregation: max, not mean -- checked against a model whose output is known
# --------------------------------------------------------------------------- #

class _ConstantLogit(nn.Module):
    """Returns a fixed logit per input row, read from a lookup by the row's mean.

    Standing in for a trained encoder so the aggregation can be checked without
    training one: the model's own weights would make the "expected" score a second
    computation of the thing being tested.
    """

    def __init__(self, lookup):
        super().__init__()
        self.lookup = lookup
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        keys = x.mean(dim=(1, 2, 3))
        out = []
        for k in keys:
            k = float(k)
            nearest = min(self.lookup, key=lambda ref: abs(ref - k))
            out.append([self.lookup[nearest]])
        return torch.tensor(out) + self.dummy * 0


def test_exam_score_is_the_maximum_slice_probability(tmp_path):
    """One bright, confident slice must be able to carry a whole exam of dim ones."""
    path = write_exam(str(tmp_path / "src"), "DBT-P1", label=1, depth=4, size=4)
    bank_dir = build_exam_bank([path], str(tmp_path / "bank"), image_size=4)

    # Overwrite the bank's slices with four distinct, known constant values, before
    # `ExamBank` opens the memmap read-only.
    values = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float16)
    volumes_path = os.path.join(bank_dir, "volumes.npy")
    writable = np.lib.format.open_memmap(volumes_path, mode="r+")
    writable[:] = values[:, None, None]
    writable.flush()
    del writable

    bank = ExamBank(bank_dir)
    logits = {0.1: -5.0, 0.2: -3.0, 0.3: -1.0, 0.9: 4.0}  # sigmoid(4.0) is the max
    model = _ConstantLogit(logits)

    scores, labels = exam_scores(model, torch.device("cpu"), bank, ["DBT-P1-s0"])
    assert scores["DBT-P1-s0"] == pytest.approx(torch.sigmoid(torch.tensor(4.0)).item(),
                                                abs=1e-4)
    # Not the mean of the four sigmoids, which would sit far lower.
    assert scores["DBT-P1-s0"] > 0.9
    assert labels["DBT-P1-s0"] == 1


def test_patient_pooling_takes_the_most_suspicious_view():
    """A patient's score is their most suspicious exam, not an average of views."""
    exam_score = {"DBT-P1-lcc": 0.1, "DBT-P1-rmlo": 0.8, "DBT-P2-lcc": 0.3}
    exam_label = {"DBT-P1-lcc": 0, "DBT-P1-rmlo": 1, "DBT-P2-lcc": 0}
    exam_patient = {"DBT-P1-lcc": "DBT-P1", "DBT-P1-rmlo": "DBT-P1", "DBT-P2-lcc": "DBT-P2"}

    scores, labels = pool_to_patient(exam_score, exam_label, exam_patient)
    assert scores == {"DBT-P1": pytest.approx(0.8), "DBT-P2": pytest.approx(0.3)}
    assert labels == {"DBT-P1": 1, "DBT-P2": 0}


def test_a_patient_is_a_cancer_patient_if_any_exam_is():
    """The label follows the same rule TransformData applies to a mixed series."""
    exam_score = {"DBT-P1-lcc": 0.1, "DBT-P1-rmlo": 0.2}
    exam_label = {"DBT-P1-lcc": 0, "DBT-P1-rmlo": 1}
    exam_patient = {"DBT-P1-lcc": "DBT-P1", "DBT-P1-rmlo": "DBT-P1"}

    _scores, labels = pool_to_patient(exam_score, exam_label, exam_patient)
    assert labels["DBT-P1"] == 1


# --------------------------------------------------------------------------- #
# The metrics this head is measured against
# --------------------------------------------------------------------------- #

def test_auc_rewards_separating_the_pooled_patient_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    separated = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    tied = np.full(6, 0.5)
    assert roc_auc(y, separated) == pytest.approx(1.0)
    assert roc_auc(y, tied) == pytest.approx(0.5)


def test_bootstrap_resamples_patients_not_slices():
    """Same guard as lesionclf's: the CI must not shrink by pretending N is bigger."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 15 + [1] * 5)
    s = y + rng.normal(0, 0.3, size=20)
    result = bootstrap_auc(y, s, n_resamples=500, seed=0)
    assert result["n"] == 20
    assert result["lo"] <= result["auc"] <= result["hi"]


# --------------------------------------------------------------------------- #
# End to end, on a corpus too small to learn anything -- just the plumbing
# --------------------------------------------------------------------------- #

def test_cross_validate_runs_and_reports_the_documented_shape(tmp_path):
    root = str(tmp_path / "src")
    for i in range(8):
        write_exam(root, f"DBT-P{i:02d}", label=i % 2, depth=5, size=32)
    paths = [os.path.join(root, f) for f in os.listdir(root)]

    args = build_arg_parser().parse_args([])
    args.data_dir = root
    args.bank_dir = str(tmp_path / "bank")
    args.output_dir = str(tmp_path / "out")
    args.folds = 2
    args.epochs = 1
    args.bag_size = 3
    args.image_size = 32
    args.bootstrap = 50
    args.batch_size = 2

    report = cross_validate(args)
    assert report["corpus"]["exams"] == len(paths)
    assert report["corpus"]["patients"] == 8
    assert set(report["auc_patient"]) >= {"auc", "lo", "hi", "n"}
    assert os.path.exists(os.path.join(args.output_dir, "cv_report.json"))
    assert os.path.exists(os.path.join(args.output_dir, "cv_predictions.csv"))
    assert os.path.exists(os.path.join(args.output_dir, "fold0.pt"))


def test_every_patient_is_scored_exactly_once_across_folds(tmp_path):
    """The point of pooling out-of-fold predictions: full coverage, one score each."""
    root = str(tmp_path / "src")
    for i in range(10):
        write_exam(root, f"DBT-P{i:02d}", label=i % 2, depth=5, size=32)

    args = build_arg_parser().parse_args([])
    args.data_dir = root
    args.bank_dir = str(tmp_path / "bank")
    args.output_dir = str(tmp_path / "out")
    args.folds = 5
    args.epochs = 1
    args.bag_size = 3
    args.image_size = 32
    args.bootstrap = 50

    report = cross_validate(args)
    assert report["corpus"]["patients"] == 10
    assert report["auc_patient"]["n"] == 10
