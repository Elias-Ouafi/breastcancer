"""The /biopsie page has to answer, and to say why when it cannot.

Step 2 was trained, measured and persisted for weeks without a single caller (see
plan.md, "Étape 2 — faite, mesurée, débranchée"). What was missing was not the model
but the route, so these tests are about the route: that it renders thirty named
fields, that a submission comes back with a verdict, and that the three ways a
submission can be wrong each produce a message rather than a stack trace.
"""
from __future__ import annotations

import os

import pytest

import config
from tabular_export import ARTEFACT_NAME

ARTEFACT_PATH = os.path.join(config.TABULAR_MODEL_DIR, ARTEFACT_NAME)
needs_model = pytest.mark.skipif(
    not os.path.exists(ARTEFACT_PATH),
    reason="no exported tabular model on this machine")


@pytest.fixture
def client():
    from app.server import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_the_form_renders_all_thirty_fields(client):
    from app.server import MEASUREMENTS, STATISTICS

    body = client.get("/biopsie").get_data(as_text=True)
    assert body.count('type="text"') == 30
    for name, _label, _hint in MEASUREMENTS:
        for suffix, _stat in STATISTICS:
            assert f'name="{name}{suffix}"' in body


def test_the_form_renders_without_a_model(client, monkeypatch):
    """A checkout that has never run the training script still gets a usable page."""
    monkeypatch.setattr("app.server.os.path.exists", lambda path: False)
    body = client.get("/biopsie").get_data(as_text=True)
    assert "Modèle absent" in body
    assert body.count('type="text"') == 30


@needs_model
def test_an_example_prefills_every_field(client):
    from app.server import BIOPSY_EXAMPLES

    body = client.get("/biopsie?exemple=malin").get_data(as_text=True)
    for value in BIOPSY_EXAMPLES["malin"]["values"][:5]:
        assert f'value="{value}"' in body


def test_an_unknown_example_name_renders_an_empty_form(client):
    """Query strings come from users; an unknown one is not an error, just empty."""
    body = client.get("/biopsie?exemple=../../etc/passwd").get_data(as_text=True)
    assert body.count('value=""') == 30


@needs_model
def test_a_complete_submission_returns_a_verdict(client):
    from app.server import BIOPSY_EXAMPLES, _feature_order

    for name, expected in (("malin", "Malin"), ("benin", "Bénin")):
        form = dict(zip(_feature_order(), BIOPSY_EXAMPLES[name]["values"]))
        body = client.post("/biopsie", data=form).get_data(as_text=True)
        assert expected in body


@needs_model
def test_a_saturated_probability_is_not_rendered_as_a_certainty(client):
    """0.9999999999991 rounds to "100,0 %", which reads as a promise. It is bounded."""
    from app.server import BIOPSY_EXAMPLES, _feature_order

    form = dict(zip(_feature_order(), BIOPSY_EXAMPLES["malin"]["values"]))
    body = client.post("/biopsie", data=form).get_data(as_text=True)
    assert "100,0 %" not in body and "100.0 %" not in body
    assert "99,9 %" in body


def test_an_incomplete_submission_says_how_many_are_missing(client):
    body = client.post("/biopsie", data={"radius1": "12.0"}).get_data(as_text=True)
    assert "29 mesure(s) manquante(s)" in body


def test_a_non_numeric_submission_is_refused_with_a_message(client):
    from app.server import _feature_order

    form = {c: "1.0" for c in _feature_order()}
    form["radius1"] = "douze"
    body = client.post("/biopsie", data=form).get_data(as_text=True)
    assert "doivent être des nombres" in body


@needs_model
def test_a_comma_decimal_separator_is_accepted(client):
    """The page is in French, the keyboard produces "12,7", and it must not 500."""
    from app.server import BIOPSY_EXAMPLES, _feature_order

    form = {c: str(v).replace(".", ",")
            for c, v in zip(_feature_order(), BIOPSY_EXAMPLES["benin"]["values"])}
    body = client.post("/biopsie", data=form).get_data(as_text=True)
    assert "Bénin" in body


@needs_model
def test_the_json_api_returns_the_same_verdict(client):
    from app.server import BIOPSY_EXAMPLES, _feature_order

    payload = {"features": dict(zip(_feature_order(), BIOPSY_EXAMPLES["malin"]["values"]))}
    response = client.post("/api/biopsie", json=payload)
    assert response.status_code == 200
    assert response.get_json()["diagnosis"] == "Malignant"


def test_the_json_api_rejects_a_body_without_features(client):
    response = client.post("/api/biopsie", json={"radius1": 12.0})
    assert response.status_code == 400
    assert "features" in response.get_json()["error"]


@needs_model
def test_the_json_api_names_a_missing_feature(client):
    response = client.post("/api/biopsie", json={"features": {"radius1": 12.0}})
    assert response.status_code == 400
    assert "Missing features" in response.get_json()["error"]


def test_the_home_page_links_to_step_two(client):
    """The route existing is not the same as it being reachable."""
    assert "/biopsie" in client.get("/").get_data(as_text=True)
