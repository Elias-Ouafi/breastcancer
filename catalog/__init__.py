"""A queryable metadata catalogue of the DBT data, in one DuckDB file.

Until now answering "how many cancer patients of the validation split are on disk, and
which of them made it into the exam corpus?" meant opening three CSVs, two manifests
and a directory listing in pandas, and writing the join again every time. This
package loads all of it once and layers it the way a warehouse would:

    raw    the sources as published, untouched (tables)
    stg    one cleaned, typed, renamed view per source
    mart   the tables people query: fct_series, dim_patient, corpus and coverage summaries
    qa     data-quality checks and their results

    python -m catalog build        # ~seconds; writes data/curated_data/catalog/
    python -m catalog checks       # the data-quality report of the last build
    python -m catalog query "SELECT split, status, count(*) FROM mart.dim_patient GROUP BY ALL"

The SQL lives in ``catalog/sql`` as plain files, one model per file, so the layering
can move to dbt without rewriting it.
"""
