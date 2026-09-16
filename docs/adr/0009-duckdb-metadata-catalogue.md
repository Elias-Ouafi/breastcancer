# 0009 — A DuckDB metadata catalogue with SQL layers and SQL checks

**Status:** accepted · **In effect:** 2026-09-16 · **Details:** [catalog.md](../catalog.md)

## Context

The metadata about the DBT data was spread across 9 CSV tables (22,032 labels rows),
two `manifest.json` files (1,130 cases) and ~1,000 series folders. Each question about
it ("which cancer patients are not downloaded?", "does every corpus label agree with
its source?") was answered by a throwaway pandas script. Those joins were never tested,
and the answers were never kept.

Whatever replaces them has to run on a laptop, from a clone, inside the CI that runs
the tests, with no server to operate.

## Decision

- **Engine: DuckDB**, embedded, one file.
  - PostgreSQL would need a running server for a single-user, rebuild-from-files
    workload.
  - SQLite lacks the analytical SQL used here (`GROUP BY ALL`, `FILTER`, `ANTI JOIN`,
    `quantile_cont`), cannot read a list of CSVs with differing schemas in one call,
    and has no Parquet export.
  - pandas is what is being replaced.
- **Warehouse-style layers:** `raw` (untouched, source file kept per row) → `stg` (one
  typed view per source) → `mart` (tables with a declared grain) → `qa`. There is one
  SQL model per file, so migrating to dbt means moving files, not rewriting logic.
- **Data-quality checks as SQL files** that return offending rows, each declaring
  `error` or `warn` in a header. Error failures set a non-zero exit code, but the
  catalogue is still written, because it is the tool for investigating the failure.
- **Full rebuild, atomic.** The build writes to a temporary file and moves it into place
  only when every layer has run. It takes ~4 s on the full collection, so incremental
  loading would add complexity for nothing.
- **Parquet export** of the marts, so any other tool can read them without DuckDB.

## Consequences

- The collection's status counts (4,581 / 278 / 112 / 89) are now reproduced
  independently of `TransformData`, in SQL.
- The first build surfaced facts no script had reported: 3 cancer patients never
  downloaded, and 60 annotated test-split patients on disk but in no corpus.
- All 14 checks pass on the real data. Tests inject defects to show that the checks can
  fail.
- DuckDB becomes a base dependency (a single wheel, no service).
- Not covered yet: the DCE-MRI corpus, which has no manifest. The catalogue also records
  that a series is present on disk, not that it is readable.
