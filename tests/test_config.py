"""The storage layout is a contract, so it gets tested like one.

Every pipeline script now reads its paths from ``config``. That makes a typo there a
silent, repo-wide failure -- a training run that writes 5 GB outside the data tree, or
a checkpoint that lands somewhere ``.gitignore`` does not cover. These tests pin the
three properties the rest of the project assumes.
"""
from __future__ import annotations

import os

import pytest

import config


# Every public path constant, discovered rather than listed, so a new one added to
# config.py is covered without touching this file. A constant may also be a *group* of
# paths (a tuple of paths is checked member by member), so grouping does not buy an
# exemption from the contract.
def _path_members(name):
    value = getattr(config, name)
    if isinstance(value, str):
        return [(name, value)]
    if isinstance(value, tuple) and value and all(isinstance(v, str) for v in value):
        return [(f"{name}[{i}]", v) for i, v in enumerate(value)]
    return []


PATH_CONSTANTS = sorted(
    member for name in dir(config) if name.isupper()
    for member in _path_members(name)
)


def test_there_are_path_constants_to_check():
    """Guard against the discovery above silently matching nothing."""
    assert len(PATH_CONSTANTS) > 10, PATH_CONSTANTS


@pytest.mark.parametrize("name,value", PATH_CONSTANTS)
def test_paths_are_absolute_and_inside_the_repo(name, value):
    """A relative path would resolve against the caller's cwd, not the project.

    That is exactly the bug this module was introduced to remove: ``--data-dir
    silver`` only worked when you happened to run from the repo root.
    """
    assert os.path.isabs(value), f"{name} is relative: {value}"
    assert os.path.commonpath([config.ROOT, value]) == config.ROOT, \
        f"{name} escapes the repo: {value}"


@pytest.mark.parametrize("layer", ["BRONZE_DIR", "SILVER_DIR", "GOLD_DIR"])
def test_layers_live_under_the_data_tree(layer):
    """`.gitignore` excludes `data/*`; a layer outside it would leak into git."""
    assert os.path.commonpath([config.DATA_DIR, getattr(config, layer)]) == config.DATA_DIR


def test_models_are_not_inside_the_data_tree():
    """Checkpoints are versioned, data is not -- and git cannot re-include a file whose
    parent directory is excluded. Nesting the two would make that impossible to express.
    """
    assert os.path.commonpath([config.DATA_DIR, config.MODELS_DIR]) != config.DATA_DIR


def test_ensure_dirs_creates_and_returns(tmp_path):
    target = os.path.join(str(tmp_path), "a", "b")
    assert config.ensure_dirs(target) == (target,)
    assert os.path.isdir(target)
    config.ensure_dirs(target)  # idempotent
