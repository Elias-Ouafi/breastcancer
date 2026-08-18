"""The flattened model has to be the Spark model, and to fail loudly when it is not.

The export exists so that step 2 can be served without a JVM. That trade is only
sound while the arrays reproduce what Spark computes, and nothing in the artefact
itself would reveal a drift: a stale ``tabular_model.json`` scores every record
happily and wrongly. Two things guard it here -- an arithmetic check against a
pipeline whose answer can be worked out by hand, and, when a JVM is present, the real
comparison against ``PipelineModel`` over the whole dataset.

The rest is about refusing bad input, because a form submits strings and a JSON body
submits whatever it likes.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import config
from tabular_export import (
    ARTEFACT_NAME,
    FORMAT_VERSION,
    ScoringError,
    TabularScorer,
)

FEATURES = ["a", "b"]


def _artefact(**overrides):
    """A two-feature pipeline whose output can be computed by hand.

    Identity-ish throughout: no imputation shift, unit scaling, an identity PCA and a
    single positive coefficient. Anything the scorer gets wrong shows up as an
    arithmetic difference rather than as a plausible-looking number.
    """
    artefact = {
        "format_version": FORMAT_VERSION,
        "feature_order": FEATURES,
        "label_map": {"0.0": "Benign", "1.0": "Malignant"},
        "served_model": "logistic",
        "produces_probability": True,
        "impute_surrogates": [10.0, 20.0],
        "scaler_mean": [0.0, 0.0],
        "scaler_std": [1.0, 1.0],
        "pca_components": [[1.0, 0.0], [0.0, 1.0]],
        "pca_explained_variance": [0.7, 0.3],
        "rescaler_mean": [0.0, 0.0],
        "rescaler_std": [1.0, 1.0],
        "coefficients": [1.0, 0.0],
        "intercept": 0.0,
    }
    artefact.update(overrides)
    return artefact


def test_the_arithmetic_is_the_pipeline_not_an_approximation_of_it():
    scorer = TabularScorer(_artefact())
    # margin = a, so P(malignant) = sigmoid(a). sigmoid(2) = 0.8807970779...
    result = scorer.predict({"a": 2.0, "b": 0.0})
    assert result["margin"] == pytest.approx(2.0)
    assert result["malignant_probability"] == pytest.approx(0.8807970779778823, abs=1e-12)
    assert result["diagnosis"] == "Malignant"


def test_the_decision_boundary_sits_at_a_zero_margin():
    scorer = TabularScorer(_artefact())
    assert scorer.predict({"a": -0.001, "b": 0.0})["diagnosis"] == "Benign"
    assert scorer.predict({"a": 0.001, "b": 0.0})["diagnosis"] == "Malignant"


def test_a_huge_margin_does_not_overflow_to_nan():
    """exp(1000) is inf, and inf/inf is nan -- which would render as a blank verdict."""
    scorer = TabularScorer(_artefact())
    for margin in (1000.0, -1000.0):
        probability = scorer.predict({"a": margin, "b": 0.0})["malignant_probability"]
        assert np.isfinite(probability)
        assert 0.0 <= probability <= 1.0


def test_standardising_a_constant_column_does_not_divide_by_zero():
    """Spark maps a zero standard deviation to zero rather than to infinity."""
    scorer = TabularScorer(_artefact(scaler_std=[0.0, 1.0]))
    assert np.isfinite(scorer.predict({"a": 5.0, "b": 1.0})["margin"])


def test_a_missing_value_is_imputed_with_the_surrogate():
    scorer = TabularScorer(_artefact())
    assert scorer.predict({"a": None, "b": 1.0})["margin"] == pytest.approx(10.0)


def test_features_may_arrive_in_any_order_when_named():
    scorer = TabularScorer(_artefact())
    assert (scorer.predict({"b": 1.0, "a": 2.0})["margin"]
            == scorer.predict({"a": 2.0, "b": 1.0})["margin"])


def test_a_missing_feature_is_named_rather_than_defaulted():
    with pytest.raises(ScoringError, match="Missing features.*'b'"):
        TabularScorer(_artefact()).predict({"a": 1.0})


def test_an_unexpected_feature_is_rejected_not_ignored():
    """Silently dropping it would score a record the caller did not describe."""
    with pytest.raises(ScoringError, match="Unexpected features"):
        TabularScorer(_artefact()).predict({"a": 1.0, "b": 2.0, "diagnosis": 1.0})


def test_a_sequence_of_the_wrong_length_is_rejected():
    with pytest.raises(ScoringError, match="Expected 2 feature values"):
        TabularScorer(_artefact()).predict([1.0])


def test_text_that_is_not_a_number_is_rejected():
    with pytest.raises(ScoringError, match="must be numbers"):
        TabularScorer(_artefact()).predict({"a": "abc", "b": 1.0})


def test_an_artefact_from_another_format_version_refuses_to_load():
    """A field that moved must not be read as though it had not."""
    with pytest.raises(ScoringError, match="format version"):
        TabularScorer(_artefact(format_version=FORMAT_VERSION + 1))


def test_a_missing_artefact_says_how_to_produce_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_tabular_model"):
        TabularScorer.load(str(tmp_path))


def test_load_round_trips_through_disk(tmp_path):
    path = tmp_path / ARTEFACT_NAME
    path.write_text(json.dumps(_artefact()), encoding="utf-8")
    scorer = TabularScorer.load(str(tmp_path))
    assert scorer.feature_order == FEATURES
    assert scorer.n_components == 2


# --------------------------------------------------------------------------- #
# The real thing, when the machine can run it
# --------------------------------------------------------------------------- #

ARTEFACT_PATH = os.path.join(config.TABULAR_MODEL_DIR, ARTEFACT_NAME)
PIPELINE_PATH = os.path.join(config.TABULAR_MODEL_DIR, "pipeline_model")


@pytest.mark.skipif(not os.path.exists(ARTEFACT_PATH),
                    reason="no exported tabular model on this machine")
def test_the_shipped_artefact_describes_the_wisconsin_pipeline():
    """Guards the shape of what is versioned, which CI does have."""
    scorer = TabularScorer.load(config.TABULAR_MODEL_DIR)
    assert len(scorer.feature_order) == 30
    assert scorer.n_components == 10
    # 95% of the variance is the documented selection rule for k (Final_Report.md).
    assert scorer.explained_variance.sum() == pytest.approx(0.95, abs=0.02)


@pytest.mark.skipif(not os.path.exists(ARTEFACT_PATH),
                    reason="no exported tabular model on this machine")
def test_the_shipped_artefact_separates_a_known_benign_from_a_known_malignant():
    """Two records straight out of the Wisconsin table, with the labels it gives them.

    Not a performance measurement -- that is in Final_Report.md, on a held-out split.
    This is the smoke test that the versioned file is the fitted model and not, say,
    an all-zero export that would still load and still answer.
    """
    from app.server import BIOPSY_EXAMPLES

    scorer = TabularScorer.load(config.TABULAR_MODEL_DIR)
    assert scorer.predict(BIOPSY_EXAMPLES["malin"]["values"])["diagnosis"] == "Malignant"
    assert scorer.predict(BIOPSY_EXAMPLES["benin"]["values"])["diagnosis"] == "Benign"


@pytest.mark.skipif(not os.path.isdir(PIPELINE_PATH),
                    reason="no Spark PipelineModel to compare against")
def test_the_export_matches_spark_on_the_whole_dataset():
    """The check the export's whole claim rests on. Needs pyspark, a JVM and network.

    Skipped in CI, which installs neither Spark nor Java on purpose (see ci.yml).
    It runs where the model is fitted, which is the machine where a divergence would
    be introduced.
    """
    pytest.importorskip("pyspark")

    import inference
    from ExtractData import extract_breast_cancer_wisconsin_diagnostic_data

    raw = extract_breast_cancer_wisconsin_diagnostic_data()
    if raw is None:
        pytest.skip("Wisconsin dataset unavailable (network or storage cap)")

    scorer = TabularScorer.load(config.TABULAR_MODEL_DIR)
    # A sample, not all 569: each Spark call is a session round-trip, and a divergence
    # in an affine pipeline is not going to hide in the rows this misses.
    for position in (0, 1, 42, 200, 568):
        record = {c: float(raw.iloc[position][c]) for c in scorer.feature_order}
        fast = scorer.predict(record)
        slow = inference.predict_tabular_spark(record)
        assert fast["prediction"] == slow["prediction"]
        assert fast["malignant_probability"] == pytest.approx(
            slow["malignant_probability"], abs=1e-9)
