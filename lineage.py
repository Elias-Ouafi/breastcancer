"""A record, beside each preprocessed dataset, of what produced it.

Right now a folder of ``.npz`` files answers none of the questions you need answered
six weeks later: which raw series did this come from, which phase was subtracted, was
it cropped, which commit of the preprocessing code ran, and when. The metrics quoted
in the README depend on all five, and none of them is recoverable from the files.

So each preprocessing run drops a ``manifest.json`` next to its output. It is small,
it is JSON, and it is written last -- an interrupted run leaves no manifest, which is
itself the correct signal that the folder is partial.

This is deliberately not a lineage *system*. There is no server, no database and no
run identifier to look up; the cost of those only pays off with several teams and
scheduled reruns. What is being bought here is the ability to answer "where did this
come from" from the folder itself.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timezone

import config

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def git_revision():
    """Short commit hash of the code that ran, or None outside a git checkout.

    Suffixed with ``-dirty`` when the tree has uncommitted changes: a manifest naming
    a clean commit that does not describe the code that actually ran is worse than no
    manifest at all.
    """
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT, capture_output=True, text=True, timeout=10,
        )
        if rev.returncode != 0:
            return None
        revision = rev.stdout.strip()

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=config.ROOT, capture_output=True, text=True, timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            revision += "-dirty"
        return revision
    except (OSError, subprocess.SubprocessError):
        return None


def relative_path(path):
    """Repo-relative, forward-slashed — a manifest should not name someone's home dir."""
    if path is None:
        return None
    try:
        return os.path.relpath(str(path), config.ROOT).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        return str(path)


def _scrub(value):
    """Rewrite any absolute in-repo path found in `value` as a repo-relative one.

    ``parameters`` is free-form, so relying on every caller to remember would mean the
    guarantee holds until the first one forgets. Enforcing it here makes it a property
    of the manifest rather than a convention.
    """
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and os.path.isabs(value):
        try:
            if os.path.commonpath([config.ROOT, value]) == config.ROOT:
                return relative_path(value)
        except ValueError:  # different drive, or not a path at all
            pass
    return value


def write_manifest(output_dir, source, parameters, cases=None, warnings=None):
    """Write ``manifest.json`` into ``output_dir`` and return the path.

    ``cases`` is the per-case summary from ``validation.summarise``, keyed by case id;
    it makes the manifest enough to spot a shape or class-balance shift between two
    runs without reopening a single ``.npz``.
    """
    cases = cases or {}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "source": relative_path(source),
        "output_dir": relative_path(output_dir),
        "parameters": _scrub(parameters),
        "n_cases": len(cases),
        "validation_warnings": list(warnings or []),
        "cases": cases,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info("Lineage manifest: %s (%d cases, %d warning(s))",
             relative_path(path), len(cases), len(manifest["validation_warnings"]))
    return path


def read_manifest(output_dir):
    """Return the manifest for a preprocessed folder, or None if it has none."""
    path = os.path.join(output_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
