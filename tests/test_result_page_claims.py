"""The two screens must not claim more than the code delivers.

These are wording tests, deliberately. Both screens carried a number that the code
contradicts -- a "Confiance 100 %" pill computed from a saturated constant, and a
"mesurée sur 20 % des 569 cas tenus à l'écart de l'entraînement" that describes a
different model than the one being served (see plan.md, "Écarts doc <-> code relevés
le 2026-09-12"). A claim removed by hand comes back by hand, so it is pinned here.

The templates are rendered directly rather than through /predict or /biopsie: both
routes need a checkpoint or an exported model, and neither is needed to check what
the page says. Fabricated result dicts stand in, carrying exactly the saturated
value the served checkpoint produces.
"""
from __future__ import annotations

import pytest
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


@pytest.mark.parametrize("stale", [
    "tenus à l'écart de l'entraînement",  # the served model is fitted on 569/569
    "97,7",                               # accuracy of another model entirely
    "99,8",                               # its ROC-AUC
])
def test_the_biopsy_page_claims_no_out_of_sample_measurement(stale):
    from app.server import _biopsy_form_context

    body = render("biopsy.html", backend="mock", **_biopsy_form_context(
        result={"prediction": 1.0, "diagnosis": "Malignant",
                "malignant_probability": 0.9999999999991}))
    assert stale not in body
    assert "sans jeu de test" in body


def test_the_biopsy_page_borrows_no_imaging_metric():
    """Dice, lesion sensitivity and false positives per volume belong to the U-Net.

    They sat in base.html's shared limits panel, so step 2 displayed them right under
    a sentence saying this model never reads an image.
    """
    from app.server import _biopsy_form_context

    body = render("biopsy.html", backend="mock", **_biopsy_form_context())
    for imaging_only in ("Dice", "0,53", "99,97 %"):
        assert imaging_only not in body


def test_the_imaging_pages_keep_them():
    body = render_result()
    assert "Dice" in body and "0,53" in body
