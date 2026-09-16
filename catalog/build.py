"""Build the catalogue: load the raw sources, run the SQL layers, check, export.

The build is deliberately a function of the files on disk and nothing else. It never
reads a DICOM pixel, never downloads, and never writes outside ``config.CATALOG_DIR``.
Deleting the catalogue loses nothing: ``python -m catalog build`` recreates it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb

import config
from lineage import git_revision

log = logging.getLogger(__name__)

SQL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql")
LAYERS = ("staging", "marts")
SCHEMAS = ("raw", "stg", "mart", "qa")

# A TCIA series folder is named by its DICOM SeriesInstanceUID: digits and dots only.
# Anything else under the download directory (``duke_mri/``, the annotation CSVs) is
# not a DBT series and stays out of the disk inventory.
SERIES_FOLDER = re.compile(r"^[0-9]+(\.[0-9]+)+$")


@dataclass
class Sources:
    """Where each input lives. Defaults to the project layout; tests point elsewhere."""

    labels: list[str]
    boxes: list[str]
    file_paths: list[str]
    tcia_dir: str | None = None
    manifests: dict[str, str] = field(default_factory=dict)
    predictions: str | None = None

    @classmethod
    def from_config(cls) -> "Sources":
        return cls(
            labels=[config.DBT_LABELS_TRAIN, config.DBT_LABELS_VALIDATION,
                    config.DBT_LABELS_TEST],
            boxes=[config.DBT_BOXES_TRAIN, config.DBT_BOXES_VALIDATION,
                   config.DBT_BOXES_TEST],
            file_paths=[config.DBT_FILE_PATHS_TRAIN, config.DBT_FILE_PATHS_VALIDATION,
                        config.DBT_FILE_PATHS_TEST],
            tcia_dir=config.TCIA_DIR,
            manifests={
                "dbt": os.path.join(config.DBT_PREPROCESSED_DIR, "manifest.json"),
                "dbt_exams": os.path.join(config.DBT_EXAMS_PREPROCESSED_DIR, "manifest.json"),
            },
            predictions=config.EXAMCLF_PREDICTIONS,
        )


@dataclass
class CheckResult:
    name: str
    severity: str
    description: str
    failing_rows: int

    @property
    def passed(self) -> bool:
        return self.failing_rows == 0


@dataclass
class BuildResult:
    db_path: str
    row_counts: dict[str, int]
    checks: list[CheckResult]

    @property
    def failed_errors(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "error" and not c.passed]


# --- Raw layer ----------------------------------------------------------------

def _existing(paths, what):
    present = [p for p in paths if p and os.path.exists(p)]
    if not present:
        raise FileNotFoundError(
            f"no {what} table found (looked for {', '.join(paths)}). "
            "Fetch them with ExtractData.download_dbt_tables().")
    for missing in sorted(set(paths) - set(present)):
        log.warning(f"[catalog] {what}: {missing} not found, building without it")
    return [p.replace(os.sep, "/") for p in present]


def _load_csv_tables(con, sources: Sources) -> None:
    """The three BCS-DBT tables, every split pooled, source file kept per row.

    ``union_by_name`` because the splits do not share a schema: the train boxes table
    has no ``VolumeSlices`` column, the validation and test ones do.
    """
    for table, paths, what in (
        ("dbt_labels", sources.labels, "labels"),
        ("dbt_boxes", sources.boxes, "boxes"),
        ("dbt_file_paths", sources.file_paths, "file-paths"),
    ):
        con.execute(
            f"CREATE TABLE raw.{table} AS SELECT * FROM "
            "read_csv(?, header = true, union_by_name = true, filename = true)",
            [_existing(paths, what)],
        )
    # Present in the validation and test boxes tables only: a build from the train
    # split alone would otherwise fail on a column staging expects.
    con.execute("ALTER TABLE raw.dbt_boxes ADD COLUMN IF NOT EXISTS VolumeSlices INTEGER")


def _scan_disk(tcia_dir: str | None) -> list[tuple[str, int, int]]:
    """``(series_uid, n_files, n_bytes)`` for every series folder downloaded.

    Sizes come from ``stat`` only: listing ~1,000 folders takes well under a second,
    opening their 138 GB would not.
    """
    rows = []
    if not tcia_dir or not os.path.isdir(tcia_dir):
        return rows
    for entry in os.scandir(tcia_dir):
        if not (entry.is_dir() and SERIES_FOLDER.match(entry.name)):
            continue
        n_files = n_bytes = 0
        for root, _, files in os.walk(entry.path):
            for name in files:
                n_files += 1
                n_bytes += os.stat(os.path.join(root, name)).st_size
        rows.append((entry.name, n_files, n_bytes))
    return rows


def _load_disk(con, sources: Sources) -> None:
    con.execute("CREATE TABLE raw.disk_series "
                "(series_uid VARCHAR, n_files BIGINT, n_bytes BIGINT)")
    rows = _scan_disk(sources.tcia_dir)
    if rows:
        con.executemany("INSERT INTO raw.disk_series VALUES (?, ?, ?)", rows)


def _load_manifests(con, sources: Sources) -> None:
    """One row per preprocessing run, one row per case it wrote.

    The case fields are kept close to the JSON (shape as three columns, warnings
    counted) so the staging layer, not this loader, decides what they mean.
    """
    con.execute("""
        CREATE TABLE raw.manifest_runs (
            corpus VARCHAR, generated_at VARCHAR, git_revision VARCHAR, source VARCHAR,
            output_dir VARCHAR, n_cases BIGINT, n_validation_warnings BIGINT,
            parameters VARCHAR)""")
    con.execute("""
        CREATE TABLE raw.manifest_cases (
            corpus VARCHAR, series_uid VARCHAR, case_id VARCHAR, study_uid VARCHAR,
            view VARCHAR, label INTEGER, exam_status VARCHAR, lesion_class VARCHAR,
            n_slices INTEGER, height INTEGER, width INTEGER, dtype VARCHAR,
            lesion_voxels BIGINT, lesion_slices INTEGER, lesion_slice_fraction DOUBLE,
            mirrored BOOLEAN, n_warnings INTEGER)""")
    for corpus, path in sources.manifests.items():
        if not os.path.exists(path):
            log.warning(f"[catalog] manifest for {corpus} not found at {path}, skipping")
            continue
        with open(path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        con.execute(
            "INSERT INTO raw.manifest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [corpus, manifest.get("generated_at"), manifest.get("git_revision"),
             manifest.get("source"), manifest.get("output_dir"), manifest.get("n_cases"),
             len(manifest.get("validation_warnings") or []),
             json.dumps(manifest.get("parameters"), sort_keys=True)])
        rows = []
        for series_uid, case in (manifest.get("cases") or {}).items():
            shape = list(case.get("shape") or []) + [None, None, None]
            rows.append((
                corpus, series_uid, case.get("case_id"), case.get("study_uid"),
                case.get("view"), case.get("label"), case.get("exam_status"),
                case.get("lesion_class"), shape[0], shape[1], shape[2], case.get("dtype"),
                case.get("lesion_voxels"), case.get("lesion_slices"),
                case.get("lesion_slice_fraction"), case.get("mirrored"),
                len(case.get("warnings") or []),
            ))
        if rows:
            con.executemany(
                f"INSERT INTO raw.manifest_cases VALUES ({', '.join(['?'] * 17)})", rows)


def _load_predictions(con, sources: Sources) -> None:
    con.execute("CREATE TABLE raw.examclf_predictions "
                "(patient VARCHAR, label INTEGER, score DOUBLE, fold INTEGER)")
    if sources.predictions and os.path.exists(sources.predictions):
        con.execute(
            "INSERT INTO raw.examclf_predictions "
            "SELECT patient, label, score, fold FROM read_csv(?, header = true)",
            [sources.predictions.replace(os.sep, "/")])
    else:
        log.warning(f"[catalog] no exam classifier predictions at {sources.predictions}")


# --- SQL layers and checks ------------------------------------------------------

def _sql_files(subdir: str) -> list[str]:
    folder = os.path.join(SQL_DIR, subdir)
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".sql"))


def _run_layer(con, layer: str) -> None:
    for path in _sql_files(layer):
        with open(path, encoding="utf-8") as fh:
            con.execute(fh.read())


_HEADER = re.compile(r"^--\s*(severity|description):\s*(.+?)\s*$", re.MULTILINE)


def _run_checks(con) -> list[CheckResult]:
    """Each check is a query returning the rows that violate it; zero rows is a pass.

    The header of each file declares its severity. ``error`` means a downstream number
    would be wrong (a label that disagrees with its source); ``warn`` means something
    worth knowing that the pipeline already handles (a series not downloaded yet).
    """
    con.execute("""
        CREATE TABLE qa.check_results (
            check_name VARCHAR, severity VARCHAR, description VARCHAR,
            failing_rows BIGINT, passed BOOLEAN)""")
    results = []
    for path in _sql_files("checks"):
        with open(path, encoding="utf-8") as fh:
            sql = fh.read()
        header = dict(_HEADER.findall(sql))
        name = os.path.splitext(os.path.basename(path))[0]
        severity = header.get("severity", "error")
        if severity not in ("error", "warn"):
            raise ValueError(f"{path}: severity must be 'error' or 'warn', got {severity!r}")
        con.execute(f"CREATE TABLE qa.{name} AS {sql.strip().rstrip(';')}")
        failing = con.execute(f"SELECT count(*) FROM qa.{name}").fetchone()[0]
        result = CheckResult(name, severity, header.get("description", ""), failing)
        con.execute("INSERT INTO qa.check_results VALUES (?, ?, ?, ?, ?)",
                    [name, severity, result.description, failing, result.passed])
        results.append(result)
    return results


def _write_build_info(con, sources: Sources) -> None:
    con.execute("""
        CREATE TABLE main.build_info AS SELECT
            ?::TIMESTAMPTZ AS built_at, ? AS git_revision, ? AS sources""",
        [datetime.now(timezone.utc).isoformat(), git_revision(),
         json.dumps({"labels": sources.labels, "boxes": sources.boxes,
                     "file_paths": sources.file_paths, "tcia_dir": sources.tcia_dir,
                     "manifests": sources.manifests, "predictions": sources.predictions},
                    default=str)])


def _export_parquet(con, parquet_dir: str) -> None:
    os.makedirs(parquet_dir, exist_ok=True)
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'mart' ORDER BY table_name").fetchall()
    for (table,) in tables:
        target = os.path.join(parquet_dir, f"{table}.parquet").replace(os.sep, "/")
        con.execute(f"COPY mart.{table} TO '{target}' (FORMAT parquet)")


def build(db_path: str = config.CATALOG_DB, sources: Sources | None = None,
          parquet_dir: str | None = config.CATALOG_PARQUET_DIR) -> BuildResult:
    """Rebuild the catalogue from scratch and return what it holds.

    Written to a temporary file first and moved into place only once every layer has
    run: a build that fails halfway leaves the previous catalogue intact rather than a
    half-populated one that answers queries wrongly. Failing *checks* do not abort the
    build -- the catalogue is exactly what you need to investigate them -- they are
    returned, and the CLI turns error-severity failures into a non-zero exit.
    """
    sources = sources or Sources.from_config()
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".duckdb",
                                    dir=os.path.dirname(os.path.abspath(db_path)))
    os.close(fd)
    os.remove(tmp_path)  # DuckDB refuses to open an existing empty file

    try:
        con = duckdb.connect(tmp_path)
        try:
            for schema in SCHEMAS:
                con.execute(f"CREATE SCHEMA {schema}")
            _load_csv_tables(con, sources)
            _load_disk(con, sources)
            _load_manifests(con, sources)
            _load_predictions(con, sources)
            for layer in LAYERS:
                _run_layer(con, layer)
            checks = _run_checks(con)
            _write_build_info(con, sources)
            row_counts = {
                f"{schema}.{table}": con.execute(
                    f"SELECT count(*) FROM {schema}.{table}").fetchone()[0]
                for schema, table in con.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_schema IN ('raw', 'mart') "
                    "ORDER BY table_schema, table_name").fetchall()
            }
            if parquet_dir:
                _export_parquet(con, parquet_dir)
        finally:
            con.close()
        os.replace(tmp_path, db_path)  # atomic, and overwrites on Windows too
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    for name, n in row_counts.items():
        log.info(f"[catalog] {name}: {n} rows")
    return BuildResult(db_path, row_counts, checks)
