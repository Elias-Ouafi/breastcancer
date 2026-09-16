"""The result screen must not claim more than the code delivers.

This is a wording test, deliberately: the screen carried a "Confiance 100 %" pill
computed from a saturated constant (see docs/journal.md, "Écarts doc <-> code relevés le
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


def test_the_page_says_the_detection_question_has_no_operating_point():
    """The screen localises; it does not decide whether there is a cancer.

    Saying nothing about the second question let the first one's numbers stand in for
    it -- a reader sees "Sensibilité 88 %" on a cancer-detection tool and reads it as
    the detection rate. The panel now names the gap and the model that owns it.
    """
    body = render_result()
    assert "y a-t-il un cancer" in body
    assert "aucun point de fonctionnement publiable" in body.lower()


def test_the_detection_numbers_are_attributed_to_their_own_corpus():
    """A number in this panel must say which model and which corpus it came from.

    The exam-level figures are measured on 272 patients / 56 cancers; the Dice pills
    above them on 28 DCE-MRI test patients. Printing the first set without its corpus
    beside it is how they get read as the second's.
    """
    body = render_result()
    assert "272 patients" in body and "56 cancers" in body
    assert "20,4 %" in body and "20,6 %" in body


def test_the_page_does_not_promote_the_broken_head_into_a_headline_pill():
    """The exam-level numbers stay in prose, behind a disclosure.

    Adding a "Spécificité 20,4 %" pill next to "Dice 0,53" would put two models'
    metrics on one row, which is exactly what was removed from /biopsie once already.
    """
    body = render_result()
    metrics_row = body.split('<div class="metrics">')[1].split("</div>")[0]
    assert "20,4" not in metrics_row
    assert "Spécificité" not in metrics_row
