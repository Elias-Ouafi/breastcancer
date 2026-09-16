# Architecture decision records

One decision per file: the context that forced it, what was decided, and what it costs.
Each record points to the dated entry in [journal.md](../journal.md) (French) that holds
the full measurements.

| # | Decision | In effect since |
|---|---|---|
| [0001](0001-layered-storage-and-central-paths.md) | Layered storage, every path defined once | by 2026-08-18 |
| [0002](0002-validate-at-write-and-record-lineage.md) | Validate at the single write point, record lineage beside the data | 2026-08-18 |
| [0003](0003-orchestrate-with-prefect.md) | Orchestrate with Prefect; idempotent stages; retry only transient failures | by 2026-08-18 |
| [0004](0004-match-annotations-by-join.md) | Match annotations to series by a join, never by inference | 2026-09-13 |
| [0005](0005-one-source-one-geometry.md) | Both classes from one source, through one geometry | 2026-09-13 |
| [0006](0006-patient-level-evaluation.md) | Patient-level evaluation, out-of-fold operating threshold | 2026-09-15 |
| [0007](0007-groupnorm-over-pretrained-encoder.md) | GroupNorm from scratch over an ImageNet-pretrained encoder | 2026-08-02 |
| [0008](0008-demo-reproducible-from-a-clone.md) | The demo runs from a clone: versioned, tested artefacts and a narrow image | 2026-08-18 |
| [0009](0009-duckdb-metadata-catalogue.md) | A DuckDB metadata catalogue with SQL layers and SQL checks | 2026-09-16 |
| [0010](0010-dbt-for-the-catalogue-transformations.md) | dbt for the catalogue's transformations and tests | 2026-09-16 |
