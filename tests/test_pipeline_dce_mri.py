"""Tests for the orchestration logic, not for the pipeline's compute.

Running the real flow costs hours of download and GPU. What is worth testing cheaply
is the part that decides *whether* to spend them: stage ordering, the resume point,
and the skip-when-output-exists rule. A bug there either redoes a 60 GB download or,
worse, silently skips a stage whose output is stale.

Prefect tasks keep the undecorated function on ``.fn``, so each stage body can be
called directly with no flow context and no server.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("prefect", reason="orchestration extra not installed")

import config  # noqa: E402
from pipelines.dce_mri import (  # noqa: E402
    STAGES,
    _describe,
    build_arg_parser,
    dce_mri_pipeline,
    download,
    preprocess,
    purge,
    train_unet,
)


def _args(**overrides):
    args = build_arg_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_stage_order_is_the_dependency_order():
    """Each stage consumes what the previous one writes; the order is not cosmetic."""
    assert STAGES == ["download", "preprocess", "purge", "train", "evaluate"]


def test_rejects_an_unknown_resume_point():
    # ``.fn`` runs the body directly. Calling the flow itself would spin up a temporary
    # Prefect server for nine seconds to reach a check that happens on line one.
    with pytest.raises(ValueError, match="start_at"):
        dce_mri_pipeline.fn(start_at="trian")


@pytest.mark.parametrize("start_at,expected", [
    ("download", 5),
    ("preprocess", 4),
    ("purge", 3),
    ("train", 2),
    ("evaluate", 1),
])
def test_describe_plans_exactly_the_remaining_stages(start_at, expected):
    plan = _describe(_args(start_at=start_at))
    assert f"Would run {expected} stage(s)" in plan
    for stage in STAGES[STAGES.index(start_at):]:
        assert stage in plan


def test_describe_never_points_outside_the_project():
    """A typo in config.py would show up here as a path escaping the repo."""
    plan = _describe(_args())
    for line in plan.splitlines():
        if "->" in line and os.sep in line:
            target = line.split("->", 1)[1].strip()
            assert os.path.commonpath([config.ROOT, target]) == config.ROOT, target


def test_download_skips_when_the_raw_layer_is_already_populated(tmp_path, monkeypatch):
    """Re-running the flow must not re-fetch 60 GB that is already on disk."""
    raw = tmp_path / "tcia"
    (raw / "duke_mri" / "series-1").mkdir(parents=True)
    monkeypatch.setattr(config, "TCIA_DIR", str(raw))

    # No monkeypatching of the downloader: if the skip fails, the import of
    # tcia_utils/network call is what breaks, which is a loud enough failure.
    result = download.fn(max_patients=5, max_gb=1)
    assert result == os.path.join(str(raw), "duke_mri")


def test_preprocess_skips_when_volumes_exist(tmp_path, monkeypatch):
    out = tmp_path / "dce"
    out.mkdir()
    (out / "Breast_MRI_001.npz").write_bytes(b"")
    monkeypatch.setattr(config, "DCE_MRI_SILVER_DIR", str(out))

    assert preprocess.fn("ignored-raw-dir", "ignored-boxes") == str(out)


def test_preprocess_reports_a_missing_annotation_table_by_name(tmp_path, monkeypatch):
    """The failure a first-time user hits, so it must name the file, not stack-trace."""
    monkeypatch.setattr(config, "DCE_MRI_SILVER_DIR", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="Annotation_Boxes"):
        preprocess.fn(str(tmp_path), str(tmp_path / "Annotation_Boxes.xlsx"))


def test_train_skips_when_the_checkpoint_exists(tmp_path, monkeypatch):
    ckpt = tmp_path / "unet_best.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setattr(config, "DCE_MRI_UNET_CKPT", str(ckpt))
    monkeypatch.setattr(config, "DCE_MRI_MODEL_DIR", str(tmp_path))

    assert train_unet.fn(str(tmp_path), epochs=1) == str(ckpt)


def test_only_the_download_retries():
    """Retrying a crashed training run just burns another hour on the same exception."""
    assert download.retries == 3
    assert preprocess.retries == 0
    assert train_unet.retries == 0


# --- purge: bronze is a transit zone ------------------------------------------------

def _silver_volume(silver, patient):
    os.makedirs(silver, exist_ok=True)
    np.savez_compressed(os.path.join(silver, f"{patient}.npz"),
                        volume=np.zeros((2, 4, 4), np.float16),
                        mask=np.zeros((2, 4, 4), np.uint8))


def _bronze_world(tmp_path, monkeypatch):
    """Patients A (annotated, in silver) and B (no annotation, never preprocessed), two
    DCE phases each, in a bronze ``duke_mri`` folder."""
    import lineage
    import TransformData

    raw = tmp_path / "tcia" / "duke_mri"
    silver = tmp_path / "dce"
    groups = {}
    for patient in ("A", "B"):
        groups[patient] = {}
        for rank in (0, 2):
            folder = raw / f"{patient}-{rank}"
            folder.mkdir(parents=True)
            (folder / "1-1.dcm").write_bytes(b"\0" * 100)
            groups[patient][rank] = str(folder)
    _silver_volume(str(silver), "A")
    lineage.write_manifest(str(silver), source="x", parameters={}, cases={"A": {"label": 1}})

    monkeypatch.setattr(config, "TCIA_DIR", str(tmp_path / "tcia"))
    monkeypatch.setattr(config, "DCE_MRI_SILVER_DIR", str(silver))
    # The real nnU-Net corpus must not leak in: by default there is none.
    monkeypatch.setattr(config, "DCE_MRI_NNUNET_SILVER_DIR", str(tmp_path / "no_nnunet"))
    # Reading DICOM headers is not what is under test: hand the purge its grouping.
    monkeypatch.setattr(TransformData, "group_dce_series_by_patient", lambda root: groups)
    return raw, silver


def test_purge_deletes_every_phase_of_a_promoted_patient_and_only_theirs(tmp_path, monkeypatch):
    raw, _ = _bronze_world(tmp_path, monkeypatch)

    report = purge.fn(str(raw))

    assert sorted(os.listdir(raw)) == ["B-0", "B-2"]  # B has no volume in silver: it stays
    assert report == {"purged_series": 2, "freed_bytes": 200, "kept": False, "kept_for_nnunet": 0}


def test_keep_bronze_leaves_the_dicom_where_they_are(tmp_path, monkeypatch):
    raw, _ = _bronze_world(tmp_path, monkeypatch)

    report = purge.fn(str(raw), keep_bronze=True)

    assert len(os.listdir(raw)) == 4 and report["kept"] is True


def test_purge_keeps_a_patient_whose_volume_does_not_open(tmp_path, monkeypatch):
    raw, silver = _bronze_world(tmp_path, monkeypatch)
    (silver / "A.npz").write_bytes(b"not an archive")

    assert purge.fn(str(raw))["purged_series"] == 0
    assert len(os.listdir(raw)) == 4


def test_purge_tolerates_a_bronze_that_is_already_gone(tmp_path, monkeypatch):
    _bronze_world(tmp_path, monkeypatch)
    assert purge.fn(str(tmp_path / "nowhere"))["purged_series"] == 0


def test_download_is_not_repeated_when_bronze_is_empty_but_silver_is_full(tmp_path, monkeypatch):
    """The end state of the medallion flow. Reading it as "nothing downloaded" would
    fetch 60 GB again on every run."""
    import sys
    import types

    monkeypatch.setattr(config, "TCIA_DIR", str(tmp_path / "tcia"))
    monkeypatch.setattr(config, "DCE_MRI_SILVER_DIR", str(tmp_path / "dce"))
    _silver_volume(str(tmp_path / "dce"), "A")
    fetched = []
    fake = types.ModuleType("ExtractData")
    fake.download_dce_mri_series = lambda **kw: fetched.append(kw)
    monkeypatch.setitem(sys.modules, "ExtractData", fake)

    target = download.fn(max_patients=5, max_gb=1)

    assert target == os.path.join(str(tmp_path / "tcia"), "duke_mri")
    assert fetched == [], "the flow went back to the network"


def test_forcing_a_preprocess_on_a_purged_bronze_says_what_is_missing(tmp_path):
    boxes = tmp_path / "Annotation_Boxes.xlsx"
    boxes.write_bytes(b"")
    empty = tmp_path / "duke_mri"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="Bronze holds no series"):
        preprocess.fn(str(empty), str(boxes), force=True)


def test_the_dry_run_shows_the_purge_and_the_opt_out():
    assert "deletes the patients now in silver" in _describe(_args())
    assert "--keep-bronze" in _describe(_args(keep_bronze=True))


# --- purge with the nnU-Net corpus as a second reader ---------------------------------

def _nnunet_world(tmp_path, monkeypatch, verified=(0,), write_files=True):
    """The bronze world of `_bronze_world`, plus an nnU-Net corpus that ingested patient A."""
    import json

    raw, silver = _bronze_world(tmp_path, monkeypatch)
    nnunet = tmp_path / "nnunet"
    native = nnunet / "native" / "A"
    native.mkdir(parents=True)
    (native / "ingest.json").write_text(json.dumps({"phases": [0, 2], "verified_phases": list(verified)}))
    if write_files:
        for phase in (0, 2):
            (native / f"A_ph{phase}.nii.gz").write_bytes(b"nifti")
    monkeypatch.setattr(config, "DCE_MRI_NNUNET_SILVER_DIR", str(nnunet))
    return raw


def test_the_nnunet_corpus_keeps_the_phases_it_did_not_verify(tmp_path, monkeypatch):
    raw = _nnunet_world(tmp_path, monkeypatch, verified=(0,))

    report = purge.fn(str(raw))

    assert sorted(os.listdir(raw)) == ["A-2", "B-0", "B-2"]      # only A's verified phase is gone
    assert report["purged_series"] == 1 and report["kept_for_nnunet"] == 1


def test_a_patient_the_nnunet_corpus_never_ingested_is_not_purged(tmp_path, monkeypatch):
    raw = _nnunet_world(tmp_path, monkeypatch, verified=())

    assert purge.fn(str(raw))["purged_series"] == 0
    assert len(os.listdir(raw)) == 4


def test_a_verified_phase_whose_native_file_is_gone_is_not_purged(tmp_path, monkeypatch):
    """`ingest.json` says verified, but the copy is not on disk any more: the DICOM must stay."""
    raw = _nnunet_world(tmp_path, monkeypatch, verified=(0, 2), write_files=False)

    assert purge.fn(str(raw))["purged_series"] == 0
    assert len(os.listdir(raw)) == 4


def test_without_an_nnunet_corpus_the_purge_is_unchanged(tmp_path, monkeypatch):
    raw, _ = _bronze_world(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "DCE_MRI_NNUNET_SILVER_DIR", str(tmp_path / "does_not_exist"))

    assert purge.fn(str(raw))["purged_series"] == 2
