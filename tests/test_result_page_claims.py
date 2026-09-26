"""The result screen must not claim more than the code delivers.

This is a wording test, deliberately: the screen carried a "Confiance 100 %" pill
computed from a saturated constant (see DOCUMENTATION.md, "Écarts doc <-> code relevés le
2026-09-12"). A claim removed by hand comes back by hand, so it is pinned here.

The template is rendered directly rather than through /predict: that route needs a
checkpoint, which is not needed to check what the page says. A fabricated result dict
stands in, carrying exactly the saturated value the served checkpoint produces.
"""
from __future__ import annotations

from flask import render_template

# The constant the served DCE-MRI checkpoint returns: max per-pixel lesion
# probability, 1.0000 on 28/28 test patients and 160/160 slices of Breast_MRI_001.
SATURATED = {
    "lesion_detected": True,
    "slice_preselected": False,
    "slice_selector": "segmentation_confidence",
    "confidence": 1.0,
    "best_slice": 74,
    "box_xywh": [102, 88, 41, 37],
    "box_full_frame_xywh": [102, 88, 41, 37],
    "n_slices": 160,
    "inference_ms": 71.0,
    "backend": "dce_mri",
}


def render(template, **context):
    from app.server import app

    with app.test_request_context():
        return render_template(template, **context)


def render_result(backend="dce_mri", **overrides):
    result = dict(SATURATED, **overrides)
    return render(
        "result.html", result=result, backend=backend, filename="Breast_MRI_001.npz",
        overlay_data_uri=None, overlay_box_pct=None, strip=None)


def test_the_result_page_shows_no_detection_confidence():
    """`best_conf` is a constant: displaying it as a percentage invented a measure."""
    body = render_result()
    assert "Confiance" not in body       # the pill, capitalised as it was displayed
    assert "100 %" not in body           # and the rounded certainty it printed


def test_the_result_page_shows_the_raw_value_under_its_own_name():
    """Hiding it would be the other failure mode: 1,0000 is what a reader should see."""
    body = render_result()
    assert "Probabilité max. par pixel" in body
    assert "1,0000" in body


def test_the_result_page_says_the_verdict_is_constant():
    body = render_result()
    assert "constante" in body
    assert "28/28" in body


def test_the_mock_backend_reports_no_model_value_at_all():
    """Nothing was computed, so there is no probability to report, raw or otherwise."""
    body = render_result(backend="mock", slice_selector="mock", confidence=0.62)
    assert "Probabilité max. par pixel" not in body
    assert "Confiance" not in body
    assert "0,6200" not in body


def test_the_imaging_pages_keep_them():
    body = render_result()
    assert "Dice" in body and "0,53" in body


def test_no_page_cites_the_removed_dbt_model():
    """The exam-level DBT classifier was deleted on 2026-09-20; no screen may cite it.

    A panel kept describing it as "servie par un autre modèle" after its code was gone,
    so a reader met figures (272 patients, VPP 20,4 %) for a model the app cannot run.
    """
    pages = [render_result(), render("how.html", backend="dce_mri")]
    for body in pages:
        assert "y a-t-il un cancer" not in body
        assert "272 patients" not in body
        assert "20,4" not in body


def test_the_metrics_row_holds_only_the_imaging_model():
    """Only the DCE-MRI localisation figures sit in the headline pills."""
    body = render_result()
    metrics_row = body.split('<div class="metrics">')[1].split("</div>")[0]
    assert "Dice" in metrics_row
    assert "Spécificité" not in metrics_row
