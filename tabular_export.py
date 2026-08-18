"""The tabular pipeline, flattened into arrays that score without a JVM.

The Wisconsin model is fitted with Spark MLlib and that is not going to change: the
whole point of ``Main.py`` is to be a Spark pipeline. But *serving* it through Spark
was never going to work. ``inference.predict_tabular`` starts a Spark session and
loads a ``PipelineModel`` from disk on every call -- seconds of JVM startup to score
thirty floats -- and the demo image deliberately ships neither Spark nor a JVM,
because carrying them would triple its size for a code path the imaging demo never
touches. So the app could not call it, and did not.

The way out is that every stage of this pipeline is an affine map or a matrix
product:

    impute -> standardise -> PCA -> standardise -> logistic

Nothing here is iterative, data-dependent at inference time, or otherwise awkward to
restate. Extracting the fitted constants -- surrogate means, two sets of mean/std,
the 30xk component matrix, the coefficients -- gives an artefact of a few kilobytes
that reproduces the Spark model exactly, not approximately. `ScoringError` covers the
only ways that can go wrong, all of them about the input rather than the maths.

This is an export, not a reimplementation. The numbers all come from the fitted
``PipelineModel``; if the training pipeline changes, this file does not need to,
because it reads whatever the stages hold. What it does need is the parity check in
``tests/test_tabular_export.py``, which scores the whole Wisconsin dataset both ways
and fails if they diverge -- the guarantee is worth exactly as much as that test.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# Bumped when the artefact's shape changes in a way a stale file would survive
# silently. A scorer refusing to load beats a scorer reading a field that moved.
FORMAT_VERSION = 1

ARTEFACT_NAME = "tabular_model.json"


class ScoringError(ValueError):
    """The input to a scorer does not match what the exported model expects."""


# --------------------------------------------------------------------------- #
# Export (training time -- needs Spark, runs once)
# --------------------------------------------------------------------------- #

def _dense_vector(vector):
    return [float(v) for v in vector.toArray()]


def export_pipeline(model, metadata, model_dir):
    """Flatten a fitted Spark ``PipelineModel`` into ``tabular_model.json``.

    ``model`` is the pipeline built by ``train_tabular_model.train_and_save``, whose
    stages are, in order: imputer, assembler, scaler, PCA, rescaler, classifier. The
    stages are read by position because that is how the pipeline is constructed a few
    lines away, in the same repository -- a lookup by class would be indirection
    without a second caller to justify it.
    """
    imputer, _assembler, scaler, pca, rescaler, classifier = model.stages

    feature_order = metadata["feature_order"]

    # surrogateDF is a one-row DataFrame keyed by output column. Read it here, inside
    # the session that produced it, rather than storing a handle that dies with it.
    surrogates = imputer.surrogateDF.head().asDict()

    artefact = {
        "format_version": FORMAT_VERSION,
        "feature_order": feature_order,
        "label_map": metadata["label_map"],
        "served_model": metadata["served_model"],
        "produces_probability": metadata["produces_probability"],
        "impute_surrogates": [float(surrogates[c]) for c in feature_order],
        "scaler_mean": _dense_vector(scaler.mean),
        "scaler_std": _dense_vector(scaler.std),
        # pc is 30 x k, column j being component j: `scaled @ pc` is the projection.
        "pca_components": [[float(v) for v in row] for row in pca.pc.toArray()],
        "pca_explained_variance": _dense_vector(pca.explainedVariance),
        "rescaler_mean": _dense_vector(rescaler.mean),
        "rescaler_std": _dense_vector(rescaler.std),
        "coefficients": _dense_vector(classifier.coefficients),
        "intercept": float(classifier.intercept),
    }

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, ARTEFACT_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artefact, f, indent=2)

    log.info("Exported JVM-free tabular model to %s (%d features -> %d components)",
             path, len(feature_order), len(artefact["pca_explained_variance"]))
    return path


# --------------------------------------------------------------------------- #
# Scoring (inference time -- numpy only)
# --------------------------------------------------------------------------- #

def _standardise(values, mean, std):
    """Spark's StandardScaler with withMean=True, withStd=True.

    A zero standard deviation means a constant column, which carries no information;
    Spark maps it to zero rather than dividing, and so does this.
    """
    centred = values - mean
    return np.divide(centred, std, out=np.zeros_like(centred), where=std != 0)


class TabularScorer:
    """A fitted Wisconsin pipeline, scoring one record at a time without Spark."""

    def __init__(self, artefact):
        version = artefact.get("format_version")
        if version != FORMAT_VERSION:
            raise ScoringError(
                f"Tabular artefact is format version {version!r}, this code reads "
                f"{FORMAT_VERSION}. Re-export it: python train_tabular_model.py")

        self.feature_order = list(artefact["feature_order"])
        self.label_map = dict(artefact["label_map"])
        self.served_model = artefact["served_model"]
        self.produces_probability = bool(artefact["produces_probability"])

        self._surrogates = np.asarray(artefact["impute_surrogates"], dtype=float)
        self._scaler_mean = np.asarray(artefact["scaler_mean"], dtype=float)
        self._scaler_std = np.asarray(artefact["scaler_std"], dtype=float)
        self._pca = np.asarray(artefact["pca_components"], dtype=float)
        self._rescaler_mean = np.asarray(artefact["rescaler_mean"], dtype=float)
        self._rescaler_std = np.asarray(artefact["rescaler_std"], dtype=float)
        self._coefficients = np.asarray(artefact["coefficients"], dtype=float)
        self._intercept = float(artefact["intercept"])

        self.explained_variance = np.asarray(artefact["pca_explained_variance"], dtype=float)
        self.n_components = int(self._pca.shape[1])

    @classmethod
    def load(cls, model_dir):
        path = os.path.join(model_dir, ARTEFACT_NAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No exported tabular model at {path!r}. Fit and export it first: "
                "python train_tabular_model.py (needs a JVM, once)")
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def order_features(self, features):
        """The 30 values in the persisted order, from a mapping or a sequence.

        A mapping is validated by key, which is what a form or a JSON body should
        send. A bare sequence is accepted too, and trusted to already be in order --
        there is nothing in a list of floats to check it against.
        """
        if hasattr(features, "keys"):
            missing = [c for c in self.feature_order if c not in features]
            unexpected = [c for c in features if c not in self.feature_order]
            if missing:
                raise ScoringError(f"Missing features: {missing}")
            if unexpected:
                raise ScoringError(f"Unexpected features: {unexpected}")
            values = [features[c] for c in self.feature_order]
        else:
            values = list(features)
            if len(values) != len(self.feature_order):
                raise ScoringError(
                    f"Expected {len(self.feature_order)} feature values, got {len(values)}")

        try:
            return np.asarray([float(v) if v is not None else np.nan for v in values])
        except (TypeError, ValueError) as exc:
            raise ScoringError(f"Feature values must be numbers: {exc}") from exc

    def predict(self, features):
        """Score one record. Returns the same shape of dict as the Spark path did."""
        values = self.order_features(features)

        # Impute before anything else: a NaN would otherwise poison every component.
        values = np.where(np.isnan(values), self._surrogates, values)

        scaled = _standardise(values, self._scaler_mean, self._scaler_std)
        components = scaled @ self._pca
        final = _standardise(components, self._rescaler_mean, self._rescaler_std)

        margin = float(final @ self._coefficients + self._intercept)
        prediction = 1.0 if margin > 0 else 0.0

        probability = None
        if self.produces_probability:
            # Logistic regression: the margin is the logit of the positive class.
            # Written to not overflow for margins far from zero, where exp() does.
            probability = float(1.0 / (1.0 + np.exp(-margin)) if margin >= 0
                                else np.exp(margin) / (1.0 + np.exp(margin)))

        return {
            "prediction": prediction,
            "diagnosis": self.label_map[str(prediction)],
            "malignant_probability": probability,
            "margin": margin,
        }
