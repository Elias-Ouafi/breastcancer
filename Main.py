import logging

import config
from AnalyzeData import analyze_data
from ExtractData import extract_breast_cancer_wisconsin_diagnostic_data
from logging_setup import setup_logging
from TransformData import _get_spark, transform_data

log = logging.getLogger(__name__)


def main():
    """Main function to run the entire breast cancer analytical project.
    It is composed of 2 main parts:
    1. Train the best AI model to predict breast cancer based quantitative values.
    2. Train an AI model to predict breast cancer based images.

    The quantitative pipeline runs on PySpark / Spark MLlib. A future extension
    of this project will be to train an AI model to predict breast cancer using
    the best features from both parts.
    """
    # Create necessary directories (layout defined in config.py)
    config.ensure_dirs(config.WISCONSIN_DIR, config.TABULAR_MODEL_DIR,
                       config.REPORTS_DIR, config.PLOTS_DIR)

    # Start (or reuse) the Spark session that drives the tabular pipeline.
    spark = _get_spark()

    log.info("PART 1: Breast cancer analysis project based on quantitative values...")

    try:
        # Step 1: Extract data (pandas fetch from UCI; lifted into Spark next step)
        log.info("Step 1: Extract the data")
        raw_data = extract_breast_cancer_wisconsin_diagnostic_data()

        # Step 2: Transform data (Spark MLlib: impute -> scale -> PCA)
        log.info("Step 2: Transform the data")
        transformed_data, pca, feature_contributions, top_features = transform_data(raw_data)

        # Step 3: Analyze data (Spark MLlib classifiers)
        log.info("Step 3: Analyze the data")
        results = analyze_data(transformed_data)

        log.info("Analysis complete. Results summary:")
        for model_name, metrics in results.items():
            log.info("%s:", model_name)
            for metric, value in metrics.items():
                log.info("  %-12s %.4f", metric, value)

        log.info("Metric tables written to %s, figures to %s",
                 config.REPORTS_DIR, config.PLOTS_DIR)
    finally:
        spark.stop()


if __name__ == "__main__":
    setup_logging(logfile="tabular_pipeline.log")
    main()
