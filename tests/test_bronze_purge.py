"""The one function that deletes raw data, and the manifest rule that makes a purge survivable.

``purge_bronze_series`` is destructive by design (bronze is a transit zone, the DICOM are
held once, in silver), so what is worth testing is everything it must *refuse* to do:
delete before silver is readable, delete outside bronze, delete on the strength of an
empty file. The manifest tests pin the second half: a pass that only sees what a purge
left behind must not erase the record of what it already wrote.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import lineage
import TransformData
from TransformData import _is_silver_volume, _with_previous_cases, purge_bronze_series


def _npz(path, **arrays):
    arrays = arrays or {"volume": np.zeros((2, 4, 4), np.float16),
                        "mask": np.zeros((2, 4, 4), np.uint8)}
    np.savez_compressed(path, **arrays)
    return str(path)


@pytest.fixture
def bronze(tmp_path):
    root = tmp_path / "tcia"
    series = root / "1.2.3"
    series.mkdir(parents=True)
    (series / "a.dcm").write_bytes(b"\0" * 400)
    (series / "b.dcm").write_bytes(b"\0" * 600)
    return root, series


def test_a_promoted_series_is_deleted_and_its_size_reported(tmp_path, bronze):
    root, series = bronze
    silver = _npz(tmp_path / "1.2.3.npz")

    assert purge_bronze_series(str(series), [silver], bronze_root=str(root)) == 1000
    assert not series.exists()
    assert os.path.exists(silver)  # silver is never touched


@pytest.mark.parametrize("damage", ["missing", "empty", "truncated", "wrong_arrays"])
def test_it_keeps_the_series_unless_silver_is_a_readable_volume(tmp_path, bronze, damage):
    root, series = bronze
    silver = tmp_path / "1.2.3.npz"
    if damage == "empty":
        silver.write_bytes(b"")
    elif damage == "truncated":
        _npz(silver)
        silver.write_bytes(silver.read_bytes()[:40])  # an interrupted write is not empty
    elif damage == "wrong_arrays":
        _npz(silver, something_else=np.zeros(3))

    assert purge_bronze_series(str(series), [str(silver)], bronze_root=str(root)) == 0
    assert (series / "a.dcm").exists()


def test_it_waits_for_every_corpus_that_reads_the_series(tmp_path, bronze):
    root, series = bronze
    ready = _npz(tmp_path / "lesion.npz")
    not_yet = str(tmp_path / "exam.npz")

    assert purge_bronze_series(str(series), [ready, not_yet], bronze_root=str(root)) == 0
    assert series.exists()


def test_it_refuses_to_delete_without_any_silver_file(bronze):
    root, series = bronze
    assert purge_bronze_series(str(series), [], bronze_root=str(root)) == 0
    assert series.exists()


@pytest.mark.parametrize("where", ["the_root_itself", "a_nested_folder", "a_sibling"])
def test_it_never_deletes_outside_a_direct_child_of_bronze(tmp_path, bronze, where):
    """A wrong path must raise before it can reach ``rmtree``: the tables sit in bronze
    too, and one mistaken argument would take them with the series."""
    root, series = bronze
    nested = series / "deeper"
    nested.mkdir()
    sibling = tmp_path / "silver_like"
    sibling.mkdir()
    target = {"the_root_itself": root, "a_nested_folder": nested, "a_sibling": sibling}[where]
    silver = _npz(tmp_path / "x.npz")

    with pytest.raises(ValueError, match="not a direct child"):
        purge_bronze_series(str(target), [silver], bronze_root=str(root))
    assert target.exists()


def test_a_series_already_gone_is_not_an_error(tmp_path, bronze):
    root, series = bronze
    silver = _npz(tmp_path / "x.npz")
    assert purge_bronze_series(str(root / "9.9.9"), [silver], bronze_root=str(root)) == 0


def test_a_failed_deletion_frees_nothing_and_does_not_raise(tmp_path, bronze, monkeypatch):
    root, series = bronze
    silver = _npz(tmp_path / "x.npz")

    def locked(path):
        raise PermissionError("file in use")

    monkeypatch.setattr(TransformData.shutil, "rmtree", locked)
    assert purge_bronze_series(str(series), [silver], bronze_root=str(root)) == 0


def test_silver_check_accepts_a_real_volume_only(tmp_path):
    assert _is_silver_volume(_npz(tmp_path / "ok.npz"))
    assert not _is_silver_volume(str(tmp_path / "absent.npz"))
    (tmp_path / "text.npz").write_text("hello")
    assert not _is_silver_volume(str(tmp_path / "text.npz"))


# --- the manifest describes the corpus, not the last pass ---------------------------

def test_a_pass_that_sees_nothing_keeps_the_manifest_of_what_silver_holds(tmp_path):
    silver = tmp_path / "silver"
    silver.mkdir()
    _npz(silver / "A.npz")
    _npz(silver / "B.npz")
    lineage.write_manifest(str(silver), source="x", parameters={},
                           cases={"A": {"label": 1}, "B": {"label": 0}})

    # A second pass after a purge: bronze holds no annotated patient, summary is empty.
    assert set(_with_previous_cases(str(silver), {})) == {"A", "B"}


def test_a_rewritten_case_takes_the_new_entry(tmp_path):
    silver = tmp_path / "silver"
    silver.mkdir()
    _npz(silver / "A.npz")
    lineage.write_manifest(str(silver), source="x", parameters={}, cases={"A": {"label": 0}})

    assert _with_previous_cases(str(silver), {"A": {"label": 1}}) == {"A": {"label": 1}}


def test_an_entry_whose_volume_was_deleted_does_not_linger(tmp_path):
    silver = tmp_path / "silver"
    silver.mkdir()
    lineage.write_manifest(str(silver), source="x", parameters={}, cases={"A": {"label": 1}})

    assert _with_previous_cases(str(silver), {}) == {}


def test_a_first_pass_has_no_previous_manifest_to_carry(tmp_path):
    assert _with_previous_cases(str(tmp_path / "never_built"), {"A": {"label": 1}}) == {
        "A": {"label": 1}}
