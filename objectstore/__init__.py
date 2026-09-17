"""Publish the data layers to S3-compatible object storage (MinIO locally, S3 in the cloud).

    python -m objectstore plan                  # what a sync would upload, nothing sent
    python -m objectstore sync                  # tables, manifests, catalogue Parquet
    python -m objectstore sync --dicom-sample 5 # + the raw DICOM of 5 series, as a demo
    python -m objectstore ls

Keys mirror the local layers, so a path in the bucket says which layer it belongs to:

    raw/tcia/tables/BCS-DBT-labels-train-v2.csv
    raw/tcia/series/<SeriesInstanceUID>/<file>.dcm
    preprocessed/<corpus>/manifest.json
    curated/catalog/<table>.parquet
    _meta/last_sync.json

What is published by default is what is light and worth sharing: the 9 annotation
tables (8 MB), the two corpus manifests and the catalogue's mart tables in Parquet --
readable straight from the bucket by DuckDB, Spark or pandas. The 83 GB of DICOM stay
on disk; the sync knows how to send them, and does so for a sample when asked.

The sync is idempotent and never deletes: an object whose size and MD5 match the local
file is skipped, one that differs is uploaded again, and an object only present in the
bucket is reported, not removed -- the raw layer is never rewritten, and a publish step
has no business destroying what someone else put there.
"""
