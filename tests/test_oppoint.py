"""The operating point is a published claim, so the way it is computed gets tested.

Two properties carry the weight here. The threshold must be chosen without seeing the
patients it then judges -- the same discipline as the cross-validation one level up,
and the one a reader is entitled to assume when the report says "hors du pli". And the
report must not quote a number the data does not support: a sensitivity over zero
positives is undefined, not 1.0.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imaging.metrics import (  # noqa: E402
    bootstrap_operating_point,
    threshold_for_sensitivity,
)
from imaging.oppoint import (  # noqa: E402
    build_report,
    out_of_fold_decisions,
    read_predictions,
    render_markdown,
    summarise,
    write_report,
)

# --------------------------------------------------------------------------- #
# threshold_for_sensitivity
# --------------------------------------------------------------------------- #

def test_the_threshold_reaches_the_sensitivity_it_was_asked_for():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    thr = threshold_for_sensitivity(labels, scores, 0.75)

    assert thr == pytest.approx(0.7)  # 3 of 4 positives caught
    assert summarise(labels, scores >= thr)["sensitivity"] >= 0.75


def test_a_target_between_two_positives_rounds_up_rather_than_down():
    """Asking for 60 % of 4 positives means catching 3, not 2: a sensitivity target is
    a floor to reach, not a number to land nearest to."""
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.0])

    thr = threshold_for_sensitivity(labels, scores, 0.6)

    assert summarise(labels, scores >= thr)["sensitivity"] == pytest.approx(0.75)


def test_ties_can_overshoot_the_target_and_that_is_reported_not_hidden():
    labels = np.array([1, 1, 1, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.1])

    thr = threshold_for_sensitivity(labels, scores, 0.25)

    # One positive was asked for; the three tied with it come along.
    assert summarise(labels, scores >= thr)["sensitivity"] == pytest.approx(1.0)


def test_no_positive_gives_no_threshold_rather_than_a_number():
    thr = threshold_for_sensitivity(np.zeros(5), np.linspace(0, 1, 5), 0.8)
    assert np.isnan(thr)


def test_the_full_target_catches_every_positive():
    labels = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.85, 0.4, 0.3, 0.2])

    thr = threshold_for_sensitivity(labels, scores, 1.0)

    assert summarise(labels, scores >= thr)["sensitivity"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# out_of_fold_decisions -- the property the report's honesty rests on
# --------------------------------------------------------------------------- #

def test_a_folds_threshold_never_sees_that_folds_patients():
    """The guard this module exists for.

    Fold 0 holds the only patients whose scores are extreme. If its threshold were
    computed on all the data, those scores would move it; computed on folds 1-4 only,
    it cannot. The test pins the second behaviour by building data where the two
    answers differ.
    """
    labels = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    scores = np.array([9.0, 8.0, 7.0, 6.0, 0.4, 0.3, 0.2, 0.1])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    _, thresholds = out_of_fold_decisions(labels, scores, folds, 1.0)

    # Fold 0's threshold comes from fold 1 alone, whose positives are 0.4 and 0.3.
    assert thresholds[0] == pytest.approx(0.3)
    # And not from the pooled data, whose lowest positive is also 0.3 -- so check the
    # other direction too, where the answers genuinely differ.
    assert thresholds[1] == pytest.approx(8.0)
    assert threshold_for_sensitivity(labels, scores, 1.0) == pytest.approx(0.3)


def test_every_patient_is_decided_exactly_once():
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    folds = np.array([0, 0, 1, 1, 2, 2])

    decisions, thresholds = out_of_fold_decisions(labels, scores, folds, 0.5)

    assert decisions.shape == labels.shape
    assert set(thresholds) == {0, 1, 2}


def test_a_fold_whose_complement_has_no_positive_calls_nothing():
    """Counted, not guessed at: no threshold means no positive call for that fold."""
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    folds = np.array([1, 1, 0, 0])  # fold 0's complement holds both positives; fold 1's holds none

    decisions, thresholds = out_of_fold_decisions(labels, scores, folds, 0.8)

    assert np.isnan(thresholds[1])
    assert not decisions[folds == 1].any()


# --------------------------------------------------------------------------- #
# bootstrap_operating_point
# --------------------------------------------------------------------------- #

def test_the_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.repeat([1, 0], 60)
    decisions = np.concatenate([rng.random(60) < 0.8, rng.random(60) < 0.2])
    point = summarise(labels, decisions)

    boot = bootstrap_operating_point(labels, decisions, n_resamples=2000, seed=0)

    for key in ("sensitivity", "specificity"):
        assert boot[key]["lo"] <= point[key] <= boot[key]["hi"]
        assert boot[key]["n_usable"] > 0


def test_resamples_missing_a_class_are_dropped_not_counted_as_zero():
    """One positive among many negatives: most resamples miss it entirely. Those carry
    no sensitivity, and averaging them in as 0 would invent a number."""
    labels = np.array([1] + [0] * 40)
    decisions = np.array([True] + [False] * 40)

    boot = bootstrap_operating_point(labels, decisions, n_resamples=500, seed=0)

    assert boot["n_usable"] < 500
    assert boot["n"] == 41


# --------------------------------------------------------------------------- #
# read_predictions -- refuses rather than defaults
# --------------------------------------------------------------------------- #

def _write_csv(path, rows, columns=("patient", "label", "score", "fold")):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_predictions_round_trip(tmp_path):
    path = _write_csv(tmp_path / "p.csv", [
        {"patient": "P1", "label": 1, "score": 0.9, "fold": 0},
        {"patient": "P2", "label": 0, "score": 0.1, "fold": 1},
    ])

    patients, labels, scores, folds = read_predictions(path)

    assert patients == ["P1", "P2"]
    assert labels.tolist() == [1, 0]
    assert scores.tolist() == [0.9, 0.1]
    assert folds.tolist() == [0, 1]


def test_a_file_without_a_fold_column_is_refused(tmp_path):
    """Falling back to the naive threshold here would publish the optimistic number
    under the honest one's name."""
    path = _write_csv(tmp_path / "p.csv",
                      [{"patient": "P1", "label": 1, "score": 0.9}],
                      columns=("patient", "label", "score"))

    with pytest.raises(ValueError, match="fold"):
        read_predictions(path)


def test_a_patient_named_twice_is_refused(tmp_path):
    """One row is one patient. Two rows would weight that patient twice in the ROC."""
    path = _write_csv(tmp_path / "p.csv", [
        {"patient": "P1", "label": 1, "score": 0.9, "fold": 0},
        {"patient": "P1", "label": 1, "score": 0.4, "fold": 1},
    ])

    with pytest.raises(ValueError, match="twice"):
        read_predictions(path)


def test_an_empty_file_is_refused(tmp_path):
    path = _write_csv(tmp_path / "p.csv", [])
    with pytest.raises(ValueError, match="no prediction rows"):
        read_predictions(path)


# --------------------------------------------------------------------------- #
# the report itself
# --------------------------------------------------------------------------- #

def _toy_report(n_resamples=200):
    rng = np.random.default_rng(1)
    labels = np.repeat([1, 0], 40)
    scores = np.concatenate([rng.normal(1.0, 1.0, 40), rng.normal(0.0, 1.0, 40)])
    folds = np.tile([0, 1, 2, 3], 20)
    return build_report(labels, scores, folds, n_resamples=n_resamples, seed=0)


def test_the_report_quotes_prevalence_beside_ppv():
    """A PPV without its prevalence is a number without a unit (plan.md)."""
    report = _toy_report()
    honest = report["out_of_fold_threshold"]

    assert "ppv" in honest and "prevalence" in honest
    assert honest["prevalence"] == pytest.approx(0.5)


def test_the_report_carries_the_chance_reference():
    """20 % specificity reads as a weak result until the chance line is beside it."""
    report = _toy_report()
    chance = report["chance_reference"]
    honest = report["out_of_fold_threshold"]

    assert chance["specificity_at_achieved_sensitivity"] == pytest.approx(
        1.0 - honest["sensitivity"])
    assert chance["beats_chance"] == (honest["specificity"] > 1.0 - honest["sensitivity"])


def test_the_written_report_names_no_path_outside_the_repository(tmp_path):
    """A versioned report should not name someone's home directory."""
    report = _toy_report()
    abs_predictions = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "models", "examclf", "cv_predictions.csv")

    json_path, md_path = write_report(report, str(tmp_path), abs_predictions)

    stored = json.load(open(json_path))["predictions"]
    assert stored == "models/examclf/cv_predictions.csv"
    assert not os.path.isabs(stored)
    with open(md_path, encoding="utf-8") as f:
        assert "C:\\Users" not in f.read()


def test_the_rendered_report_states_research_use_only():
    text = render_markdown(_toy_report(), "models/examclf/cv_predictions.csv")
    assert "Research Use Only" in text


def test_the_rendered_report_writes_french_decimals():
    """The document is in French; 20.6 % in it is a typo, not a number."""
    text = render_markdown(_toy_report(), "models/examclf/cv_predictions.csv")
    assert "20.6 %" not in text
    assert "%" in text and "," in text
