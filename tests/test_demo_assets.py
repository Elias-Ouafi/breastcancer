"""The demo must run from a fresh clone. These tests are what says it still does.

``run_demo.py`` preflights the same things, but only when someone runs it -- which is
typically five minutes before a pitch. The failure mode is specific and has happened:
an artefact moves, ``.gitignore`` stops covering its new path, and the checkpoint or a
demo case quietly stops being committed. The clone still installs, the app still
starts, and the demo dies on the first click.

So these assert the whole chain: the files exist, git really tracks them, and their
contents carry what ``inference.predict_dce_mri`` reads.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

import config

DEMO_CASES = sorted(
    os.path.join(config.DEMO_CASES_DIR, f)
    for f in (os.listdir(config.DEMO_CASES_DIR) if os.path.isdir(config.DEMO_CASES_DIR) else [])
    if f.endswith(".npz")
)

# Artefacts a clone must receive for the demo and for the README's numbers.
VERSIONED_ARTEFACTS = [
    config.DCE_MRI_UNET_CKPT,
    os.path.join(config.DCE_MRI_MODEL_DIR, "eval_report.json"),
    config.SLICE_CLF_CKPT,
    os.path.join(config.SLICE_CLF_DIR, "sliceclf_test_metrics.json"),
]


def _git_tracks(path):
    """True when git has `path` in the index (not merely present on disk)."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", os.path.relpath(path, config.ROOT)],
        cwd=config.ROOT, capture_output=True, text=True,
    )
    return result.returncode == 0


def test_three_demo_cases_are_present():
    """The launcher and the app's one-click buttons both assume exactly these."""
    assert len(DEMO_CASES) == 3, DEMO_CASES


@pytest.mark.parametrize("path", VERSIONED_ARTEFACTS)
def test_versioned_artefacts_exist(path):
    assert os.path.exists(path), f"missing: {os.path.relpath(path, config.ROOT)}"


@pytest.mark.parametrize("path", VERSIONED_ARTEFACTS)
def test_versioned_artefacts_are_tracked_by_git(path):
    """Existing locally is not enough -- a clone only gets what git tracks."""
    assert _git_tracks(path), (
        f"{os.path.relpath(path, config.ROOT)} exists but git does not track it; "
        "check the .gitignore exception for its directory"
    )


@pytest.mark.parametrize("path", DEMO_CASES or [pytest.param(None, marks=pytest.mark.skip)])
def test_demo_case_is_tracked_and_well_formed(path):
    """Each case must carry a pinned slice, since the model cannot find one itself.

    ``forced_slice`` is what lets the app report ``slice_preselected: true`` instead of
    pretending it located the lesion. A case without it would silently turn the demo
    into the claim the README is careful not to make.
    """
    assert _git_tracks(path), f"{os.path.basename(path)} is not tracked by git"

    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        assert {"volume", "mask", "forced_slice", "case_id"} <= keys, sorted(keys)

        volume, mask = z["volume"], z["mask"]
        assert volume.ndim == 3, volume.shape
        assert mask.shape == volume.shape, (mask.shape, volume.shape)

        forced = int(z["forced_slice"])
        assert 0 <= forced < volume.shape[0], (forced, volume.shape)
        # The pinned slice is pinned *because* a lesion is on it.
        assert mask[forced].sum() > 0, f"{os.path.basename(path)}: no lesion on slice {forced}"


def test_demo_cases_stay_small_enough_to_ship():
    """A slab (4-5 MB), not the ~30 MB full volume, is why these can live in git.

    The ceiling guards against ``make_demo_case`` being rerun with ``slim=False``, or
    with a wider slab, and a clone quietly gaining a hundred megabytes.
    """
    for path in DEMO_CASES:
        size_mb = os.path.getsize(path) / 1024 ** 2
        assert size_mb < 10, f"{os.path.basename(path)} is {size_mb:.1f} MB"
