"""The split decision: a held-out test set, folds stratified by scanner, and the guards.

Pure functions over case ids and a registry, so no image, no SimpleITK and no GPU: this
file runs in CI, where the `nnunet` extra is not installed.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mri_nnunet import registry, splits  # noqa: E402


def make_cases(n, prefix="Breast_MRI_"):
    return [f"{prefix}{i:03d}" for i in range(1, n + 1)]


def groups_of(cases, scanners):
    """Round-robin the cases over ``scanners``, deterministically."""
    return {case: scanners[i % len(scanners)] for i, case in enumerate(sorted(cases))}


def test_every_case_lands_exactly_once():
    cases = make_cases(60)
    a = splits.assign(cases, groups_of(cases, ["A 1.5T", "B 3T", "C 3T"]))
    splits.check(a)
    assert sorted(a["test"] + a["trainval"]) == sorted(cases)
    assert not set(a["test"]) & set(a["trainval"])


def test_each_training_case_validates_in_exactly_one_fold():
    cases = make_cases(60)
    a = splits.assign(cases, groups_of(cases, ["A 1.5T", "B 3T"]), folds=5)
    validated = [case for fold in a["folds"] for case in fold["val"]]
    assert sorted(validated) == a["trainval"]
    assert len(validated) == len(set(validated))
    for fold in a["folds"]:
        assert not set(fold["train"]) & set(fold["val"])
        assert sorted(fold["train"] + fold["val"]) == a["trainval"]


def test_test_set_is_about_the_requested_fraction():
    cases = make_cases(186)
    a = splits.assign(cases, groups_of(cases, ["A 1.5T", "B 3T", "C 3T", "D 1.5T"]), test_frac=0.15)
    assert 0.11 <= len(a["test"]) / len(cases) <= 0.19


def test_folds_are_stratified_by_scanner():
    """Each scanner is spread over the folds, not concentrated in one.

    With 100 patients of one scanner over 5 folds, a random draw gives ~20 each; the
    failure this guards against is a fold holding nearly all of one scanner, which would
    make the spread across folds measure the draw rather than the model.
    """
    cases = make_cases(150)
    scanners = groups_of(cases, ["A 1.5T", "B 3T", "C 3T"])
    a = splits.assign(cases, scanners, folds=5)
    for scanner in {"A 1.5T", "B 3T", "C 3T"}:
        per_fold = [sum(1 for c in fold["val"] if scanners[c] == scanner) for fold in a["folds"]]
        assert max(per_fold) - min(per_fold) <= 2, (scanner, per_fold)


def test_strata_counts_match_the_split():
    cases = make_cases(60)
    scanners = groups_of(cases, ["A 1.5T", "B 3T"])
    a = splits.assign(cases, scanners)
    for scanner, counts in a["strata"].items():
        members = {c for c in cases if scanners[c] == scanner}
        assert counts["total"] == len(members)
        assert counts["test"] == len(members & set(a["test"]))
        assert sum(counts["folds"]) + counts["test"] == counts["total"]


def test_same_seed_same_split_and_a_different_seed_differs():
    cases = make_cases(80)
    scanners = groups_of(cases, ["A 1.5T", "B 3T"])
    first = splits.assign(cases, scanners, seed=42)
    again = splits.assign(cases, scanners, seed=42)
    other = splits.assign(cases, scanners, seed=7)
    assert first == again
    assert first["test"] != other["test"]


def test_case_order_does_not_change_the_split():
    """The draw depends on the seed, not on the order the directory happened to list."""
    cases = make_cases(60)
    scanners = groups_of(cases, ["A 1.5T", "B 3T"])
    assert splits.assign(cases, scanners) == splits.assign(list(reversed(cases)), scanners)


def test_a_small_scanner_group_is_not_silently_excluded_from_the_test_set():
    """Over many seeds a 3-patient scanner contributes to the test set about 15 % of the time.

    ``floor(0.15 * 3) == 0`` would give it a zero probability, which is how a whole scanner
    ends up absent from the only set that judges the model.
    """
    cases = make_cases(3)
    scanners = {c: "rare 3T" for c in cases}
    picked = [len(splits.assign(cases, scanners, folds=2, seed=s)["test"]) for s in range(200)]
    assert 0 < sum(picked) / len(picked) < 1.2


def test_unknown_scanner_is_its_own_stratum():
    cases = make_cases(20)
    a = splits.assign(cases, {}, folds=3)
    assert set(a["strata"]) == {splits.UNKNOWN_SCANNER}


def test_scanner_key_reads_the_registry_row():
    assert splits.scanner_of({"scanner": "SIEMENS Avanto", "field_strength_t": "1.5"}) == "SIEMENS Avanto 1.5T"
    assert splits.scanner_of({"scanner": "", "field_strength_t": ""}) == splits.UNKNOWN_SCANNER


def test_scanner_groups_reads_registry_csv(tmp_path):
    path = os.path.join(tmp_path, "registry.csv")
    registry.write_csv(path, registry.REGISTRY_COLUMNS, [
        {"patient_id": "Breast_MRI_001", "scanner": "SIEMENS Avanto", "field_strength_t": "1.5"},
        {"patient_id": "Breast_MRI_002", "scanner": "GE SIGNA HDx", "field_strength_t": "3"},
    ])
    found = splits.scanner_groups(str(tmp_path), ["Breast_MRI_001", "Breast_MRI_002", "Breast_MRI_003"])
    assert found == {"Breast_MRI_001": "SIEMENS Avanto 1.5T", "Breast_MRI_002": "GE SIGNA HDx 3T",
                     "Breast_MRI_003": splits.UNKNOWN_SCANNER}


def test_too_few_cases_refuses():
    with pytest.raises(ValueError, match="cannot fill"):
        splits.assign(make_cases(4), {}, folds=5)


def test_a_test_fraction_of_one_refuses():
    with pytest.raises(ValueError, match="test_frac"):
        splits.assign(make_cases(40), {}, test_frac=1.0)


# --- the guards themselves: check() must fail on a broken split ---------------------------

def _valid():
    cases = make_cases(40)
    return splits.assign(cases, groups_of(cases, ["A 1.5T", "B 3T"]), folds=4)


def test_check_catches_a_patient_in_both_test_and_training():
    a = _valid()
    a["trainval"].append(a["test"][0])
    a["folds"][0]["val"].append(a["test"][0])
    with pytest.raises(ValueError, match="both the test set and training"):
        splits.check(a)


def test_check_catches_a_case_validated_twice():
    a = _valid()
    a["folds"][1]["val"].append(a["folds"][0]["val"][0])
    a["folds"][1]["train"] = sorted(set(a["folds"][1]["train"]) - {a["folds"][0]["val"][0]})
    with pytest.raises(ValueError, match="another fold already validated"):
        splits.check(a)


def test_check_catches_training_on_the_validation_cases():
    a = _valid()
    a["folds"][0]["train"] = sorted(set(a["folds"][0]["train"]) | {a["folds"][0]["val"][0]})
    with pytest.raises(ValueError, match="trains on its own validation"):
        splits.check(a)


def test_check_catches_a_case_no_fold_validates():
    """Trained on in every fold, validated in none: it is never scored, silently.

    The case stays in each fold's ``train`` so the per-fold coverage guard is satisfied and
    only the last guard can catch it.
    """
    a = _valid()
    never_scored = a["folds"][0]["val"].pop()
    a["folds"][0]["train"] = sorted(a["folds"][0]["train"] + [never_scored])
    with pytest.raises(ValueError, match="in no validation fold"):
        splits.check(a)


def test_check_catches_a_case_dropped_from_a_fold_entirely():
    a = _valid()
    dropped = a["folds"][0]["val"].pop()
    for fold in a["folds"]:
        fold["train"] = [c for c in fold["train"] if c != dropped]
    with pytest.raises(ValueError, match="does not cover exactly"):
        splits.check(a)


def test_check_catches_an_empty_fold():
    a = _valid()
    moved = a["folds"][0]["val"]
    a["folds"][0]["val"] = []
    a["folds"][1]["val"] += moved
    with pytest.raises(ValueError, match="empty validation set"):
        splits.check(a)


def test_write_refuses_a_broken_split(tmp_path):
    a = _valid()
    a["folds"][0]["train"] = sorted(set(a["folds"][0]["train"]) | set(a["folds"][0]["val"]))
    with pytest.raises(ValueError):
        splits.write(str(tmp_path), a)
    assert not os.path.exists(os.path.join(tmp_path, splits.SPLITS_FILENAME))


def test_write_then_read_round_trip(tmp_path):
    a = _valid()
    path = splits.write(str(tmp_path), a)
    assert json.load(open(path, encoding="utf-8")) == a
    assert splits.read(str(tmp_path)) == a
    assert splits.read(str(tmp_path / "elsewhere")) is None


def test_nnunet_payload_shape():
    a = _valid()
    payload = splits.nnunet_splits(a)
    assert len(payload) == 4
    assert all(set(entry) == {"train", "val"} for entry in payload)
    assert payload[0]["val"] == a["folds"][0]["val"]
