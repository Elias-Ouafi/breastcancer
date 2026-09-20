"""The demo click, end to end, with the real checkpoint.

``test_demo_assets.py`` proves the files exist, that git tracks them and that their
keys are the ones ``predict_dce_mri`` reads. None of that runs the model. A
checkpoint truncated by a bad clone, a torch build that cannot allocate, a template
that stopped rendering the overlay: all pass the asset tests and die on the first
click of a pitch.

So this file does what the first click does -- ``POST /demo/1`` on the ``dce_mri``
backend -- and asserts the page that comes back carries the three things the demo
shows: a verdict, an annotated slice, and the statement that the slice was pinned by
a human rather than found by the model.

Cost: one checkpoint load and one 25-slice volume, ~2 s on CPU. CI installs torch
from the CPU index, so it runs there too -- which is the point. Skipped when the
checkpoint is absent, so a contributor without the artefacts still gets a green run.
"""
from __future__ import annotations

import os

import pytest

import config

pytestmark = pytest.mark.skipif(
    not os.path.exists(config.DCE_MRI_UNET_CKPT),
    reason="checkpoint absent : rien à fumer",
)


@pytest.fixture()
def demo_client(monkeypatch):
    """A Flask test client serving the real DCE-MRI backend."""
    monkeypatch.setenv("MRI_APP_BACKEND", "dce_mri")

    import app.predictor as predictor_module
    import app.server as server_module

    # get_predictor memoises per backend name; a previous test may have cached the
    # mock under a different env. Clearing keeps this test independent of order.
    monkeypatch.setattr(predictor_module, "_PREDICTORS", {})
    server_module.app.config["TESTING"] = True
    with server_module.app.test_client() as client:
        yield client


def test_the_first_demo_click_returns_a_result_page(demo_client):
    response = demo_client.post("/demo/1")
    assert response.status_code == 200, response.status_code
    page = response.get_data(as_text=True)

    assert "Résultat de l'analyse" in page
    # The annotated slice is embedded as a data URI: no data URI, no image on screen.
    assert "data:image/png;base64," in page
    # The claim the README is careful to make, and only that one.
    assert "choisie à l'avance par un humain" in page


def test_the_api_scores_a_demo_case_with_a_pinned_slice(demo_client):
    """The result contract, checked on the file the buttons actually score."""
    from app.predictor import get_predictor

    cases = sorted(f for f in os.listdir(config.DEMO_CASES_DIR) if f.endswith(".npz"))
    result = get_predictor().predict(os.path.join(config.DEMO_CASES_DIR, cases[0]))

    assert result["backend"] == "dce_mri"
    assert result["best_slice"] is not None, "aucune coupe retenue : écran vide en démo"
    assert result["slice_preselected"] is True, (
        "la coupe doit être signalée comme imposée ; sinon la page laisse croire "
        "que le modèle l'a trouvée seul"
    )
    assert result["box_xywh"] is not None
    assert 0 <= result["best_slice"] < result["n_slices"]


def test_the_launcher_preflight_passes_on_this_machine():
    """`run_demo.py --check` is what an operator runs the evening before."""
    import run_demo

    problems, cases = run_demo._check_files()
    assert problems == [], problems
    model_problems, elapsed, result = run_demo._check_prediction(cases)
    assert model_problems == [], model_problems
    assert elapsed is not None and result is not None


def test_the_preflight_reports_a_busy_port_instead_of_starting():
    """A second launch on a used port printed the success banner and served nothing.

    On Windows werkzeug's bind succeeds on a port it already holds (SO_REUSEADDR),
    so "Running on http://127.0.0.1:5000" was printed while the OS kept routing to
    the first process. The launcher now asks by connecting, not by binding.
    """
    import socket

    import run_demo

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        busy = listener.getsockname()[1]
        assert run_demo._port_is_free(busy) is False

    # Released: the same port must read as free again.
    assert run_demo._port_is_free(busy) is True
