"""A queryable metadata catalogue of the DBT data, in one DuckDB file.

Until now answering "how many cancer patients of the validation split are on disk, and
which of them made it into the exam corpus?" meant opening three CSVs, two manifests
and a directory listing in pandas, and writing the join again every time. This
package loads all of it once and layers it the way a warehouse would:

    raw    the sources as published, untouched (tables)       <- Python (build.py)
    stg    one cleaned, typed, renamed view per source         <- dbt models
    mart   fct_series, dim_patient, corpus and coverage tables <- dbt models
    qa     offending rows of every dbt test, and their results <- dbt tests

    python -m catalog build        # ~10 s; writes data/curated_data/catalog/
    python -m catalog checks       # the data-quality report of the last build
    python -m catalog query "SELECT split, status, count(*) FROM mart.dim_patient GROUP BY ALL"
    python -m catalog docs         # dbt documentation site, lineage graph included

The transformations are a dbt project in ``catalog/dbt``. Python keeps only what dbt
cannot do: scanning the download directory, flattening JSON manifests, pooling CSVs
whose schemas differ, and moving the finished file into place atomically.
"""
