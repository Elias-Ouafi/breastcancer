"""What a clone must install for the demo, and what the UI may promise about it.

Three drifts this pins, all of which show up only in front of an audience:

* the base install of ``pyproject.toml`` and the Dockerfile install different things, so
  the image works and a host clone does not (or the reverse);
* a use case gets an extra but not a place in ``all``, or code imports a package no
  extra declares -- ``pip install -e ".[all]"`` then no longer means "everything";
* the upload widget offers a format the served backend cannot score. That was a real
  500 on a dropped DICOM, with the server-side temp path printed on the page.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib

import pytest

import config

DOCKERFILE = os.path.join(config.ROOT, "Dockerfile")
PYPROJECT = os.path.join(config.ROOT, "pyproject.toml")

# Import name -> distribution name, where the two differ.
IMPORT_TO_DISTRIBUTION = {"botocore": "boto3", "PIL": "pillow", "tcia_utils": "tcia-utils", "gdcm": "python-gdcm",
                          "yaml": "pyyaml"}


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _project():
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)["project"]


def _name(requirement):
    return re.split(r"[<>=!~\[ ]", requirement.strip())[0].lower().replace("_", "-")


def test_the_base_install_is_exactly_the_demo():
    """torch, Flask, numpy, Pillow: the whole third-party surface of the demo path.

    run_demo -> app.server -> app.predictor -> inference -> imaging.unet. Anything
    else in the base install is weight a reviewer downloads for nothing; anything
    missing is an ImportError on a machine that installed only this.
    """
    names = {_name(requirement) for requirement in _project()["dependencies"]}
    assert names == {"torch", "flask", "numpy", "pillow"}, sorted(names)


def test_the_all_extra_is_the_union_of_every_other_extra():
    """A new extra that is not in `all` is a use case `pip install -e ".[all]"` forgets."""
    extras = _project()["optional-dependencies"]
    (self_reference,) = extras["all"]
    included = set(re.search(r"\[(.*)\]", self_reference).group(1).replace(" ", "").split(","))
    assert included == set(extras) - {"all"}, (
        f"`all` couvre {sorted(included)}, les extras sont {sorted(set(extras) - {'all'})}"
    )


def test_every_third_party_import_of_the_project_is_declared_somewhere():
    """Nothing the code imports may be absent from both the base install and the extras.

    Walks the project's own modules (tests excluded: they import pytest and moto, which
    `dev` declares, and the scan would only restate that). A package imported at module
    level and declared nowhere is the bug that turns "install everything" into a
    ModuleNotFoundError on the first command.
    """
    project = _project()
    declared = {_name(r) for r in project["dependencies"]}
    for requirements in project["optional-dependencies"].values():
        declared |= {_name(r) for r in requirements if not r.startswith("breastcancer[")}
    # dbt is imported as `dbt`, shipped by dbt-duckdb (which depends on dbt-core).
    declared |= {"dbt"}
    stdlib = set(sys.stdlib_module_names)

    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=config.ROOT, check=True,
                             capture_output=True, text=True).stdout.split()
    # The project's own names: every module and package. The imaging modules import their
    # siblings twice on purpose (`from .dataset` and a bare `from dataset`), so a
    # sibling counts as local, not as a package to declare.
    local = {os.path.splitext(os.path.basename(path))[0] for path in tracked}
    local |= {part for path in tracked for part in path.split("/")[:-1]}
    undeclared = {}
    for path in tracked:
        if path.startswith(("tests/", "scripts/", "data/")) or path == "conftest.py":
            continue
        tree = ast.parse(_read(os.path.join(config.ROOT, path)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                dist = IMPORT_TO_DISTRIBUTION.get(root, root).lower().replace("_", "-")
                if root in stdlib or root in local or dist in declared or root == "__future__":
                    continue
                undeclared.setdefault(root, set()).add(path)
    assert not undeclared, (
        "importés mais déclarés dans aucun extra de pyproject.toml : "
        + ", ".join(f"{k} ({', '.join(sorted(v))})" for k, v in sorted(undeclared.items()))
    )


def test_the_image_installs_from_the_same_list():
    """One list, read by both. A second copy in the Dockerfile is a copy that rots."""
    dockerfile = _read(DOCKERFILE)
    assert "COPY pyproject.toml" in dockerfile
    assert "tomllib" in dockerfile and "['project']['dependencies']" in dockerfile
    # torch stays a separate line: it comes from the CPU index, not PyPI.
    assert 'pip install --no-cache-dir "Flask' not in dockerfile, (
        "le Dockerfile réinstalle une liste en dur à côté de pyproject.toml"
    )


def test_no_requirements_file_is_left_to_drift():
    """`pyproject.toml` is the only list; a `requirements*.txt` would be a second one."""
    leftovers = [name for name in os.listdir(config.ROOT)
                 if name.startswith("requirements") and name.endswith(".txt")]
    assert leftovers == [], leftovers


def test_the_image_healthcheck_stays_within_its_timeout():
    """`--check` now loads the U-Net; a 10 s healthcheck must use `--fast-check`."""
    dockerfile = _read(DOCKERFILE)
    healthcheck = dockerfile.split("HEALTHCHECK", 1)[1].split("\n\n", 1)[0]
    assert "--fast-check" in healthcheck, healthcheck


@pytest.mark.parametrize("backend,expected", [
    ("dce_mri", {".npz"}),
    ("mock", {".npz", ".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".nii", ".gz"}),
])
def test_the_form_only_accepts_what_the_backend_can_score(backend, expected):
    from app.server import _allowed_extensions

    assert _allowed_extensions(backend) == expected


def test_the_dropzone_advertises_the_served_backend_formats():
    """The hint text, the `accept` attribute and the server check are one set."""
    from flask import render_template

    from app.predictor import DceMriUNetPredictor
    from app.server import _page_context, app

    with app.test_request_context():
        context = _page_context(DceMriUNetPredictor())
        page = render_template("index.html", **context)

    assert context["accept"] == ".npz"
    assert "DICOM" not in context["accept_hint"], context["accept_hint"]
    assert 'accept=".npz"' in page
    # The widget must not keep a hard-coded list beside the rendered one.
    assert ".dcm,.dicom" not in page


def test_a_rejected_upload_never_prints_the_server_side_path(monkeypatch, tmp_path):
    """The page names the file the user sent, not where the server put it."""
    import io

    import app.predictor as predictor_module
    import app.server as server_module

    monkeypatch.setenv("MRI_APP_BACKEND", "dce_mri")
    monkeypatch.setattr(predictor_module, "_PREDICTORS", {})
    client = server_module.app.test_client()

    response = client.post("/predict", data={"mri": (io.BytesIO(b"x"), "examen.dcm")},
                           content_type="multipart/form-data")
    page = response.get_data(as_text=True)

    assert response.status_code == 400
    assert "examen.dcm" in page
    assert server_module.UPLOAD_DIR not in page
    assert "Format non pris en charge" in page
