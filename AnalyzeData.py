"""Tabular breast-cancer classification with PySpark MLlib.

This replaces the previous scikit-learn implementation. The train/test split,
feature scaling, the classifiers themselves and the evaluation metrics are all
computed with Spark MLlib so the quantitative pipeline runs on the same
distributed stack as the rest of the project.

Notes on the scikit-learn -> Spark MLlib mapping:
  - ``StandardScaler``/``LogisticRegression``/``RandomForestClassifier`` have
    direct MLlib equivalents.
  - ``SVC`` -> :class:`LinearSVC` (MLlib only ships a linear SVM).
  - ``MLPClassifier`` -> :class:`MultilayerPerceptronClassifier`.
  - ``KNeighborsClassifier`` has no MLlib equivalent, so it is replaced by
    :class:`GBTClassifier` (gradient-boosted trees), which also stands in for the
    XGBoost model the old pipeline used.
"""
import os

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


def get_spark(app_name="breastcancer-tabular"):
    """Return the active :class:`SparkSession`, creating one if needed."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def prepare_data(data: DataFrame, label_col="Diagnosis", seed=42):
    """Assemble features, index the label, scale, and split into train/test.

    ``data`` is expected to hold the PCA components (``PC1``..``PCk``) plus the
    ``label_col`` column. Returns ``(train_df, test_df, n_features)``.
    """
    feature_cols = [c for c in data.columns if c != label_col]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled")
    # Re-standardise the components (parity with the previous sklearn pipeline,
    # which scaled again after PCA). withMean centres like sklearn's default.
    scaler = StandardScaler(
        inputCol="assembled", outputCol="features", withMean=True, withStd=True
    )
    # 'M'/'B' -> numeric label. The majority class ('B') maps to 0.0, so the
    # positive (malignant) class is label 1.0 — matching the old encoding.
    indexer = StringIndexer(
        inputCol=label_col, outputCol="label", stringOrderType="alphabetAsc"
    )

    assembled = assembler.transform(data)
    scaled = scaler.fit(assembled).transform(assembled)
    labelled = indexer.fit(scaled).transform(scaled)

    prepared = labelled.select("features", "label").cache()
    train_df, test_df = prepared.randomSplit([0.8, 0.2], seed=seed)
    return train_df, test_df, len(feature_cols)


def _evaluate(predictions):
    """Compute accuracy / precision / recall / F1 / ROC AUC for a prediction set."""
    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(predictions)
    precision = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="precisionByLabel", metricLabel=1.0,
    ).evaluate(predictions)
    recall = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="recallByLabel", metricLabel=1.0,
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

    # ROC AUC needs a rawPrediction/probability column; every model below emits
    # rawPrediction, so this is always available.
    if "rawPrediction" in predictions.columns:
        metrics["ROC AUC"] = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        ).evaluate(predictions)
    return metrics


def train_models(train_df, test_df, n_features):
    """Train and evaluate multiple Spark MLlib classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(
            featuresCol="features", labelCol="label"
        ),
        "Random Forest": RandomForestClassifier(
            featuresCol="features", labelCol="label", seed=42
        ),
        "Linear SVM": LinearSVC(featuresCol="features", labelCol="label"),
        "Gradient-Boosted Trees": GBTClassifier(
            featuresCol="features", labelCol="label", seed=42
        ),
        "Neural Network": MultilayerPerceptronClassifier(
            featuresCol="features", labelCol="label", seed=42,
            layers=[n_features, max(n_features, 8), 2], maxIter=200,
        ),
    }

    results = {}
    for name, model in models.items():
        fitted = model.fit(train_df)
        predictions = fitted.transform(test_df)
        results[name] = _evaluate(predictions)
    return results


def save_results(results, save_path="data/model_results.csv"):
    """Persist the per-model metrics to CSV so the report can be regenerated.

    Rows are model names, columns are the metrics from :func:`_evaluate`
    (Accuracy, Precision, Recall, F1-Score, ROC AUC). This is the authoritative
    Spark MLlib output the quantitative report is built from.
    """
    metrics_df = pd.DataFrame(results).T
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_df.to_csv(save_path)
    return metrics_df


def create_model_comparison_plot(results, save_path="plots/model_comparison.png"):
    """Create and save a comparison plot of model performance."""
    if plt is None:
        print("matplotlib not installed; skipping model comparison plot.")
        return

    metrics_df = pd.DataFrame(results).T

    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind="bar", rot=45)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def analyze_data(data: DataFrame):
    """Main entry point: prepare the data, train the models, evaluate, and plot."""
    train_df, test_df, n_features = prepare_data(data)
    results = train_models(train_df, test_df, n_features)
    save_results(results)
    create_model_comparison_plot(results)
    return results
