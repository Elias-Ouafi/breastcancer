"""What a clone must install for the demo, and what the UI may promise about it.

Two drifts this pins, both of which show up only in front of an audience:

* ``requirements-demo.txt`` and the Dockerfile install different things, so the
  image works and a host clone does not (or the reverse).
* The upload widget offers a format the served backend cannot score. That was a
  real 500 on a dropped DICOM, with the server-side temp path printed on the page.
"""
from __future__ import annotations

import os
import re

import pytest

import config

DEMO_REQUIREMENTS = os.path.join(config.ROOT, "requirements-demo.txt")
DOCKERFILE = os.path.join(config.ROOT, "Dockerfile")
PYPROJECT = os.path.join(config.ROOT, "pyproject.toml")


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _requirements(text):
    """Requirement lines, comments and blanks dropped."""
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_the_demo_requirements_cover_what_the_demo_imports():
    """torch, Flask, numpy, Pillow: the whole third-party surface of the demo path.

    run_demo -> app.server -> app.predictor -> inference -> imaging.unet. Anything
    else in the list is weight a reviewer downloads for nothing; anything missing
    is an ImportError on a machine that installed only this file.
    """
    names = {re.split(r"[<>=!~\[]", line)[0].strip().lower()
             for line in _requirements(_read(DEMO_REQUIREMENTS))}
    assert names == {"torch", "flask", "numpy", "pillow"}, sorted(names)


def test_the_demo_requirements_match_the_project_pins():
    """Same version bounds as `pyproject.toml`, so the two installs agree."""
    pyproject = _read(PYPROJECT)
    for requirement in _requirements(_read(DEMO_REQUIREMENTS)):
        assert f'"{requirement}"' in pyproject, (
            f"{requirement} n'apparaît pas tel quel dans pyproject.toml : "
            "les deux installations divergeraient"
        )


def test_the_image_installs_from_the_same_file():
    """One list, read by both. A second copy in the Dockerfile is a copy that rots."""
    dockerfile = _read(DOCKERFILE)
    assert "pip install --no-cache-dir -r requirements-demo.txt" in dockerfile
    # torch stays a separate line: it comes from the CPU index, not PyPI.
    assert 'pip install --no-cache-dir "Flask' not in dockerfile, (
        "le Dockerfile réinstalle une liste en dur à côté de requirements-demo.txt"
    )


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
