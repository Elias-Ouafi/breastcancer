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
    train_unet,
)


def _args(**overrides):
    args = build_arg_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_stage_order_is_the_dependency_order():
    """Each stage consumes what the previous one writes; the order is not cosmetic."""
    assert STAGES == ["download", "preprocess", "train", "evaluate"]


def test_rejects_an_unknown_resume_point():
    # ``.fn`` runs the body directly. Calling the flow itself would spin up a temporary
    # Prefect server for nine seconds to reach a check that happens on line one.
    with pytest.raises(ValueError, match="start_at"):
        dce_mri_pipeline.fn(start_at="trian")


@pytest.mark.parametrize("start_at,expected", [
    ("download", 4),
    ("preprocess", 3),
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
    monkeypatch.setattr(config, "DCE_MRI_PREPROCESSED_DIR", str(out))

    assert preprocess.fn("ignored-raw-dir", "ignored-boxes") == str(out)


def test_preprocess_reports_a_missing_annotation_table_by_name(tmp_path, monkeypatch):
    """The failure a first-time user hits, so it must name the file, not stack-trace."""
    monkeypatch.setattr(config, "DCE_MRI_PREPROCESSED_DIR", str(tmp_path / "empty"))
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
