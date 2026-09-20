"""Build the catalogue: load the raw sources, run dbt, record the tests, export.

The build is deliberately a function of the files on disk and nothing else. It never
reads a DICOM pixel, never downloads, and never writes outside ``config.CATALOG_DIR``.
Deleting the catalogue loses nothing: ``python -m catalog build`` recreates it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

import duckdb

import config
from lineage import git_revision

log = logging.getLogger(__name__)

# The dbt project that builds the stg and mart layers and runs the tests.
DBT_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dbt")
# Created by the Python loader; dbt creates stg and mart itself.
RAW_SCHEMAS = ("raw", "qa")

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
                "dbt": os.path.join(config.DBT_SILVER_DIR, "manifest.json"),
                "dbt_exams": os.path.join(config.DBT_EXAMS_SILVER_DIR, "manifest.json"),
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


# --- dbt: staging, marts and tests ---------------------------------------------

class DbtError(RuntimeError):
    """A model failed to build, or a test could not run. Distinct from a test that ran
    and found rows: that is a data-quality result, returned rather than raised."""


def _release_dbt_connection() -> None:
    """Close the DuckDB connection dbt-duckdb caches for the life of the process.

    Run in-process, dbt keeps the database file open after `invoke` returns. The build
    then cannot reopen it to record results, and on Windows nothing else can open the
    catalogue until the interpreter exits. The adapter's own `close_all_connections`
    only drops the reference, so the environment is closed explicitly first.
    """
    from dbt.adapters.duckdb.connections import DuckDBConnectionManager
    from dbt.adapters.factory import reset_adapters

    env = DuckDBConnectionManager._ENV
    if env is not None:
        env.close()
    DuckDBConnectionManager.close_all_connections()
    reset_adapters()


def _dbt(command: str, db_path: str, target_dir: str, log_dir: str) -> list:
    """Run one dbt command (``"run"``, ``"test"``, ``"docs generate"``) against
    ``db_path`` in-process and return its node results."""
    try:
        from dbt.cli.main import dbtRunner
    except ImportError as exc:
        raise DbtError("the catalogue build needs dbt-duckdb: "
                       'pip install -e ".[catalog]"') from exc

    previous = os.environ.get("CATALOG_DB_PATH")
    os.environ["CATALOG_DB_PATH"] = db_path.replace(os.sep, "/")
    try:
        outcome = dbtRunner().invoke([
            *command.split(),
            "--project-dir", DBT_PROJECT_DIR,
            "--profiles-dir", DBT_PROJECT_DIR,
            "--target-path", target_dir,
            "--log-path", log_dir,
            "--quiet",
        ])
    finally:
        _release_dbt_connection()
        if previous is None:
            os.environ.pop("CATALOG_DB_PATH", None)
        else:
            os.environ["CATALOG_DB_PATH"] = previous

    if outcome.exception is not None:
        raise DbtError(f"dbt {command} crashed: {outcome.exception}") from outcome.exception
    if not hasattr(outcome.result, "results"):
        return []  # docs generate returns a catalogue artifact, not node results
    results = list(outcome.result.results)
    broken = [r for r in results if str(r.status) in ("error", "runtime error", "skipped")]
    if broken:
        details = "; ".join(f"{r.node.name}: {r.message}" for r in broken)
        raise DbtError(f"dbt {command}: {len(broken)} node(s) did not run -- {details}")
    return results


def _test_result(result) -> CheckResult:
    node = result.node
    severity = str(node.config.severity).lower()
    description = node.description
    if not description and getattr(node, "test_metadata", None):
        # Generic tests carry no description of their own; say what they assert.
        column = getattr(node, "column_name", None)
        model = node.attached_node.split(".")[-1] if node.attached_node else ""
        description = f"{node.test_metadata.name} on {model}" + (f".{column}" if column else "")
    failing = int(result.failures or 0)
    return CheckResult(node.name, "warn" if severity == "warn" else "error",
                       description, failing)


def _record_checks(con, checks: list[CheckResult]) -> None:
    con.execute("""
        CREATE TABLE qa.check_results (
            check_name VARCHAR, severity VARCHAR, description VARCHAR,
            failing_rows BIGINT, passed BOOLEAN)""")
    if checks:
        con.executemany("INSERT INTO qa.check_results VALUES (?, ?, ?, ?, ?)",
                        [[c.name, c.severity, c.description, c.failing_rows, c.passed]
                         for c in checks])


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
          parquet_dir: str | None = config.CATALOG_PARQUET_DIR,
          target_dir: str | None = None) -> BuildResult:
    """Rebuild the catalogue from scratch and return what it holds.

    1. Python loads the ``raw`` schema: the part dbt cannot do (scan a disk, flatten
       JSON manifests, pool CSVs whose schemas differ).
    2. ``dbt run`` builds ``stg`` and ``mart``; ``dbt test`` runs every generic and
       singular test, keeping offending rows in ``qa``.
    3. Python records the test outcomes in ``qa.check_results`` and exports the marts.

    ``run`` and ``test`` are separate on purpose: ``dbt build`` skips a model whose
    upstream test failed, which would hide exactly the tables needed to investigate.

    Everything is written to a temporary file and moved into place only once every
    step has run, so a build that fails halfway leaves the previous catalogue intact.
    Failing tests do not abort the build; they are returned, and the CLI turns
    error-severity failures into a non-zero exit.
    """
    sources = sources or Sources.from_config()
    catalog_dir = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(catalog_dir, exist_ok=True)
    target_dir = target_dir or os.path.join(catalog_dir, "dbt_target")
    log_dir = os.path.join(target_dir, "logs")
    # Built under its final file name, in a temporary directory beside it. The name
    # matters: DuckDB names the database after the file stem, and dbt-duckdb writes
    # that name into every view it creates (`"catalog".raw.dbt_labels`). A view built
    # in `tmpab12.duckdb` would point at a database that no longer exists once the file
    # is renamed -- measured, not guessed: every stg view failed to bind.
    tmp_dir = tempfile.mkdtemp(prefix=".building-", dir=catalog_dir)
    tmp_path = os.path.join(tmp_dir, os.path.basename(db_path))

    try:
        con = duckdb.connect(tmp_path)
        try:
            for schema in RAW_SCHEMAS:
                con.execute(f"CREATE SCHEMA {schema}")
            _load_csv_tables(con, sources)
            _load_disk(con, sources)
            _load_manifests(con, sources)
            _load_predictions(con, sources)
        finally:
            con.close()  # dbt opens the file itself, and DuckDB allows one writer

        _dbt("run", tmp_path, target_dir, log_dir)
        checks = [_test_result(r) for r in _dbt("test", tmp_path, target_dir, log_dir)]
        checks.sort(key=lambda c: c.name)

        con = duckdb.connect(tmp_path)
        try:
            _record_checks(con, checks)
            _write_build_info(con, sources)
            row_counts = {
                f"{schema}.{table}": con.execute(
                    f"SELECT count(*) FROM {schema}.{table}").fetchone()[0]
                for schema, table in con.execute(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_schema IN ('raw', 'mart') AND table_type = 'BASE TABLE' "
                    "ORDER BY table_schema, table_name").fetchall()
            }
            if parquet_dir:
                _export_parquet(con, parquet_dir)
        finally:
            con.close()
        os.replace(tmp_path, db_path)  # atomic, and overwrites on Windows too
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    for name, n in row_counts.items():
        log.info(f"[catalog] {name}: {n} rows")
    return BuildResult(db_path, row_counts, checks)


def generate_docs(db_path: str = config.CATALOG_DB, target_dir: str | None = None) -> str:
    """Generate the dbt documentation site (models, columns, tests, lineage graph).

    Reads the built catalogue for column types, so it runs after ``build``. Returns the
    target directory, which ``dbt docs serve`` serves.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"no catalogue at {db_path}; run `python -m catalog build`")
    target_dir = target_dir or os.path.join(os.path.dirname(os.path.abspath(db_path)),
                                            "dbt_target")
    _dbt("docs generate", db_path, target_dir, os.path.join(target_dir, "logs"))
    return target_dir
