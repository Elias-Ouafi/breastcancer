"""Fit and persist the tabular pipeline as a single reusable Spark ``PipelineModel``.

The training entry point (``Main.py`` -> ``AnalyzeData.analyze_data``) refits the
whole preprocessing chain *and* every classifier on each run and keeps nothing on
disk, so there is no way to score a new patient without re-running Spark end to end.
This module closes that gap for the inference layer: it fits the exact same
preprocessing (mean-impute -> standardise -> PCA(95% variance) -> re-standardise) plus
one classifier as a single ``Pipeline``, then saves the fitted ``PipelineModel`` so
``inference.predict_tabular`` can load it and score raw 30-feature inputs.

Design notes
------------
* The label (``StringIndexer`` on ``Diagnosis``) is applied *outside* the persisted
  pipeline. The saved model therefore only depends on the 30 feature columns and can
  transform new, unlabelled rows — a pipeline carrying the ``StringIndexer`` would
  demand a ``Diagnosis`` column at prediction time, which new inputs do not have.
* The served model defaults to **Logistic Regression**: it is tied for best on every
  metric in ``Final_Report.md`` (ROC-AUC 99.83% vs Linear SVM's 99.89%) and, unlike
  MLlib's ``LinearSVC``, emits a calibrated ``probability`` column — the confidence a
  decision-support UI needs. Pass ``served_model="linear_svm"`` for the top-ROC-AUC
  model (prediction only, no probability).
"""
from __future__ import annotations

import json
import os

import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.feature import Imputer, PCA, StandardScaler, StringIndexer, VectorAssembler

from TransformData import _get_spark, _pandas_to_spark
from ExtractData import extract_breast_cancer_wisconsin_diagnostic_data

DEFAULT_MODEL_DIR = os.path.join("results", "tabular_model")
# 'B' < 'M' under alphabetAsc, so benign -> 0.0 and malignant -> 1.0 (positive class).
LABEL_MAP = {0.0: "Benign", 1.0: "Malignant"}

_CLASSIFIERS = {
    "logistic": lambda: LogisticRegression(featuresCol="features", labelCol="label"),
    "linear_svm": lambda: LinearSVC(featuresCol="features", labelCol="label"),
}


def _choose_k(scaled_df, n_features, variance):
    """Smallest number of PCA components whose cumulative variance reaches ``variance``.

    Mirrors ``TransformData.apply_pca``: MLlib's PCA takes a fixed ``k`` rather than a
    variance target, so we fit at full rank once to read the explained-variance profile
    and pick ``k`` from it.
    """
    full = PCA(k=n_features, inputCol="scaled_features", outputCol="pca_features").fit(scaled_df)
    cumulative = np.cumsum(full.explainedVariance.toArray())
    return min(int(np.searchsorted(cumulative, variance) + 1), n_features)


def train_and_save(model_dir=DEFAULT_MODEL_DIR, served_model="logistic",
                   variance=0.95, seed=42):
    """Fit the full tabular pipeline on the Wisconsin data and persist it.

    Writes ``<model_dir>/pipeline_model`` (Spark ``PipelineModel``) and
    ``<model_dir>/metadata.json`` (feature order, label map, PCA ``k``, served model).
    Returns the metadata dict.
    """
    if served_model not in _CLASSIFIERS:
        raise ValueError(f"served_model must be one of {list(_CLASSIFIERS)}; got {served_model!r}")

    spark = _get_spark()
    raw = extract_breast_cancer_wisconsin_diagnostic_data()
    if raw is None:
        raise RuntimeError("Could not fetch the Wisconsin dataset (storage cap or network).")

    df = _pandas_to_spark(spark, raw)
    feature_cols = [c for c in df.columns if c != "Diagnosis"]
    n_features = len(feature_cols)

    # Index the label once, up front, so it is NOT part of the persisted pipeline.
    labelled = StringIndexer(
        inputCol="Diagnosis", outputCol="label", stringOrderType="alphabetAsc"
    ).fit(df).transform(df)

    # Preprocessing stages shared by k-selection and the final pipeline.
    imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols, strategy="mean")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled")
    scaler = StandardScaler(inputCol="assembled", outputCol="scaled_features",
                            withMean=True, withStd=True)

    prelim = Pipeline(stages=[imputer, assembler, scaler]).fit(labelled)
    k = _choose_k(prelim.transform(labelled), n_features, variance)

    pca = PCA(k=k, inputCol="scaled_features", outputCol="pca_features")
    # Re-standardise the PCA components before the classifier (parity with the
    # existing AnalyzeData.prepare_data step).
    rescaler = StandardScaler(inputCol="pca_features", outputCol="features",
                              withMean=True, withStd=True)
    classifier = _CLASSIFIERS[served_model]()

    pipeline = Pipeline(stages=[imputer, assembler, scaler, pca, rescaler, classifier])
    model = pipeline.fit(labelled)

    os.makedirs(model_dir, exist_ok=True)
    model.write().overwrite().save(os.path.join(model_dir, "pipeline_model"))

    metadata = {
        "feature_order": feature_cols,
        "label_map": {str(k_): v for k_, v in LABEL_MAP.items()},
        "pca_components": int(k),
        "served_model": served_model,
        "produces_probability": served_model == "logistic",
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved tabular {served_model} PipelineModel to {model_dir!r} "
          f"(PCA components: {k}).")
    return metadata


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fit and persist the tabular pipeline.")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--served-model", default="logistic", choices=list(_CLASSIFIERS))
    p.add_argument("--variance", type=float, default=0.95)
    args = p.parse_args()

    spark = _get_spark()
    try:
        train_and_save(args.model_dir, args.served_model, args.variance)
    finally:
        spark.stop()
