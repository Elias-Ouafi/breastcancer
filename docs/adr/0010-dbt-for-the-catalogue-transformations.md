# 0010 — dbt for the catalogue's transformations and tests

**Status:** accepted · **In effect:** 2026-09-16 · **Refines:** [0009](0009-duckdb-metadata-catalogue.md) · **Details:** [catalog.md](../catalog.md)

## Context

The first version of the catalogue (ADR 0009) ran its layers as plain SQL files,
executed in file-name order by a small Python runner. Its 14 checks were SQL files with
a severity written in a comment header. That runner reimplemented, badly, what dbt
provides:
- dependency order, which was encoded in file-name prefixes;
- grain and domain tests, all written by hand even when they were just "unique" or
  "one of these values";
- documentation and lineage, which did not exist at all.

## Decision

- **The `stg` and `mart` layers become a dbt project** in `catalog/dbt`, running on
  dbt-duckdb against the same file. Dependencies are declared with `ref()` and
  `source()`, not with file names.
- **Python keeps only what dbt cannot do:** pool CSVs whose schemas differ, scan the
  download directory, flatten JSON manifests, and move the finished file into place
  atomically.
- **Tests split by kind:**
  - Grain and domain rules become 25 generic tests declared in YAML: `unique`,
    `not_null`, `accepted_values`, `relationships`. Three hand-written checks
    (`labels_key_unique`, `file_paths_series_unique`, `boxes_known_class`) are removed
    in favour of these.
  - Domain rules that no generic test expresses stay as 11 singular tests, with severity
    and description in `_checks.yml`.
  - `store_failures` keeps every test's offending rows in `qa`.
- **`dbt run` then `dbt test`, not `dbt build`.** `build` skips models downstream of a
  failed test, which would hide the tables needed to investigate the failure.
- **The interface is unchanged:** a macro keeps the schema names `stg`, `mart` and `qa`,
  and staging models are aliased. `mart.dim_patient` and every example query keep
  working.
- **dbt runs in-process** (`dbtRunner`), so the CLI, tests and error handling stay in
  Python.
- **dbt-duckdb is an optional extra** (`.[catalog]`). Querying a built catalogue needs
  only duckdb, and the demo install pulls neither.

## Consequences

- **Parity checked:** the pre-dbt build and the dbt build produce identical mart tables,
  row for row, on the full collection.
- 36 tests instead of 14, all passing. A generic test is shown failing on an injected
  duplicate key.
- `python -m catalog docs` produces dbt's documentation site with the lineage graph.
- **A full build takes 9.7 s instead of 3.9 s**, the price of dbt's start-up and parsing.
  The catalogue suite in CI also runs dbt for real (19 tests, ~50 s locally).
- **Two traps found and handled,** both covered by tests or comments:
  - dbt-duckdb writes the database name, taken from the file stem, into every view. A
    build into a temporary file name broke all `stg` views once renamed, so the build
    now works under the final file name in a temporary directory.
  - dbt-duckdb keeps its connection open for the life of the process, so the build
    closes it explicitly.
- Adopting dbt was the stated follow-up of ADR 0009. The SQL itself barely changed: the
  `CREATE` statements disappeared and schema names became `ref()` and `source()` calls.
