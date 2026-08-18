"""The manifest only earns its place if it stays honest about what produced a folder."""
from __future__ import annotations

import json
import os

import config
from lineage import MANIFEST_NAME, git_revision, read_manifest, relative_path, write_manifest


def test_manifest_round_trips(tmp_path):
    out = str(tmp_path / "dce")
    write_manifest(out, source="data/raw_data/tcia", parameters={"crop": False},
                   cases={"Breast_MRI_001": {"shape": [160, 448, 448]}})

    manifest = read_manifest(out)
    assert manifest["n_cases"] == 1
    assert manifest["parameters"]["crop"] is False
    assert manifest["cases"]["Breast_MRI_001"]["shape"] == [160, 448, 448]
    assert manifest["generated_at"].endswith("+00:00"), "timestamps must be unambiguous UTC"


def test_missing_manifest_reads_as_none(tmp_path):
    """A folder without one is a folder whose run did not finish -- not an error."""
    assert read_manifest(str(tmp_path)) is None


def test_manifest_records_no_absolute_paths(tmp_path):
    """It gets committed and read by other people; nobody needs my home directory."""
    out = str(tmp_path / "dce")
    write_manifest(out, source=config.TCIA_DIR, parameters={"boxes": config.MRI_ANNOTATION_BOXES})

    raw = (tmp_path / "dce" / MANIFEST_NAME).read_text(encoding="utf-8")
    assert config.ROOT.replace(os.sep, "/") not in raw.replace("\\\\", "/")
    assert json.loads(raw)["source"] == "data/raw_data/tcia"


def test_relative_path_tolerates_none():
    assert relative_path(None) is None


def test_git_revision_is_a_hash_or_none():
    """None outside a checkout is fine; a wrong hash is not."""
    revision = git_revision()
    if revision is not None:
        base = revision.removesuffix("-dirty")
        assert 6 <= len(base) <= 12, revision
        assert all(c in "0123456789abcdef" for c in base), revision


def test_dirty_tree_is_marked(tmp_path, monkeypatch):
    """A manifest naming a clean commit that is not the code that ran is worse than none."""
    import subprocess

    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        out = "abc1234\n" if "rev-parse" in cmd else " M TransformData.py\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_revision() == "abc1234-dirty"
