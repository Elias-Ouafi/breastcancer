"""The DBT data chain as one runnable, resumable flow.

    python -m pipelines.dbt --dry-run            # the plan, computed offline, nothing run
    python -m pipelines.dbt                      # run what is missing
    python -m pipelines.dbt --from preprocess    # skip the network stages

Four stages, in dependency order:

    tables -> download -> preprocess -> catalog
    (9 CSVs)  (raw DICOM)  (2 corpora)   (DuckDB + dbt tests)

Why each stage decides from a plan rather than from "does the folder exist"
---------------------------------------------------------------------------
The DCE-MRI flow skips a stage when its output folder is non-empty. For this chain that
rule would be wrong in exactly the case that matters: the raw layer here grows over
time (normals on 2026-09-13, the test split on 2026-09-14), and a corpus built before a
download is a folder that exists and is stale. Skipping it silently is the failure this
orchestration exists to prevent.

So every stage first computes, **offline and from the collection's own tables**, what
should exist, compares it with what does, and runs only if something is missing:

* download: the series of the requested patients, read from the file-paths inventory,
  against the series folders on disk;
* preprocess: the series each corpus should contain given what is on disk, against the
  cases its manifest records;
* catalog: always rebuilt (~10 s), and its error-severity dbt tests fail the flow --
  a label that disagrees with its source is not something to hand to a model.

Retries are kept for the network stages only, as in the DCE-MRI flow. They are useful
now because a download that fails raises (``DownloadIncomplete``) instead of being
counted as a success, and a stalled request times out (``http_timeouts``) instead of
hanging for hours.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

from prefect import flow, get_run_logger, task

import config
import lineage
from logging_setup import setup_logging

log = logging.getLogger(__name__)

STAGES = ["tables", "download", "preprocess", "catalog"]
SPLITS = ("train", "validation", "test")
SERIES_FOLDER = re.compile(r"^[0-9]+(\.[0-9]+)+$")

TABLES = {
    "boxes": {"train": config.DBT_BOXES_TRAIN, "validation": config.DBT_BOXES_VALIDATION,
              "test": config.DBT_BOXES_TEST},
    "labels": {"train": config.DBT_LABELS_TRAIN, "validation": config.DBT_LABELS_VALIDATION,
               "test": config.DBT_LABELS_TEST},
    "file_paths": {"train": config.DBT_FILE_PATHS_TRAIN,
                   "validation": config.DBT_FILE_PATHS_VALIDATION,
                   "test": config.DBT_FILE_PATHS_TEST},
}
CORPORA = {
    "lesion": config.DBT_PREPROCESSED_DIR,
    "exam": config.DBT_EXAMS_PREPROCESSED_DIR,
}


def _logger():
    """Prefect's run logger inside a flow, the module logger outside it."""
    try:
        return get_run_logger()
    except Exception:  # pragma: no cover - outside a flow run
        return log


# --- The plan: pure functions over tables, disk and manifests ----------------------

def tables_for(kind, splits):
    return [TABLES[kind][split] for split in splits]


def all_tables():
    return [path for kind in TABLES.values() for path in kind.values()]


def _read(paths):
    import pandas as pd

    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        raise FileNotFoundError(f"none of {paths} exists; run the tables stage first")
    df = pd.concat(frames, ignore_index=True)
    if "View" in df.columns:
        df["View"] = df["View"].astype(str).str.strip().str.lower()
    return df


def inventory(splits=SPLITS):
    """series_uid -> (PatientID, StudyUID, View), from the file-paths tables."""
    df = _read(tables_for("file_paths", splits))
    uids = df["classic_path"].astype(str).str.replace("\\", "/").str.split("/").str[-2]
    return {uid: (row.PatientID, row.StudyUID, row.View)
            for uid, row in zip(uids, df.itertuples())}


def series_on_disk(tcia_dir=None):
    tcia_dir = tcia_dir or config.TCIA_DIR
    if not os.path.isdir(tcia_dir):
        return set()
    return {e.name for e in os.scandir(tcia_dir)
            if e.is_dir() and SERIES_FOLDER.match(e.name)}


def planned_patients(annotated_splits, max_normal_patients, seed):
    """Annotated patients of the requested splits, then the seeded sample of normals."""
    from ExtractData import choose_normal_patients

    boxes = _read(tables_for("boxes", annotated_splits))
    annotated = list(dict.fromkeys(boxes["PatientID"].tolist()))
    normals, _ = choose_normal_patients(tables_for("labels", SPLITS), max_normal_patients,
                                        seed)
    return annotated, normals


def planned_series(patients, inv):
    wanted = set(patients)
    return {uid for uid, (patient, _, _) in inv.items() if patient in wanted}


def expected_cases(corpus, corpus_splits, on_disk):
    """The series a corpus should hold, given what is on disk -- mirroring the rules of
    the function that builds it, not reimplementing its image processing."""
    inv = {uid: key for uid, key in inventory(corpus_splits).items() if uid in on_disk}
    if corpus == "exam":
        # preprocess_dbt_exams: listed in the inventory and a labels row with a flag set.
        labels = _read(tables_for("labels", SPLITS))
        flagged = labels[labels[["Normal", "Actionable", "Benign", "Cancer"]].sum(axis=1) > 0]
        keys = set(zip(flagged["PatientID"], flagged["StudyUID"], flagged["View"]))
    else:
        # preprocess_dbt_with_boxes: listed, and at least one box on that view.
        boxes = _read(tables_for("boxes", corpus_splits))
        keys = set(zip(boxes["PatientID"], boxes["StudyUID"], boxes["View"]))
    return {uid for uid, key in inv.items() if key in keys}


def manifest_cases(output_dir):
    manifest = lineage.read_manifest(output_dir) or {}
    return set((manifest.get("cases") or {}).keys())


# --- Stages ---------------------------------------------------------------------

@task(name="dbt-tables", retries=3, retry_delay_seconds=60)
def fetch_tables(force=False):
    """The nine BCS-DBT tables. Everything downstream plans from them."""
    logger = _logger()
    missing = [p for p in all_tables() if not os.path.exists(p)]
    if not missing and not force:
        logger.info("All %d tables present -- skipping.", len(all_tables()))
        return all_tables()

    from ExtractData import download_dbt_tables

    download_dbt_tables(overwrite=force)
    missing = [os.path.basename(p) for p in all_tables() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"tables still missing after download: {missing}")
    return all_tables()


@task(name="dbt-download", retries=2, retry_delay_seconds=300)
def download(annotated_splits=SPLITS, max_normal_patients=150, max_gb_added=50, seed=0,
             force=False):
    """Fetch the series of the planned patients that are not on disk yet.

    Retried: a failed series raises ``DownloadIncomplete`` once the run has fetched
    everything else, and the next attempt skips what is already on disk.
    """
    logger = _logger()
    annotated, normals = planned_patients(annotated_splits, max_normal_patients, seed)
    inv = inventory()
    on_disk = series_on_disk()
    missing = planned_series(annotated + normals, inv) - on_disk
    logger.info("Plan: %d annotated + %d normal patients; %d series planned, %d missing.",
                len(annotated), len(normals), len(planned_series(annotated + normals, inv)),
                len(missing))
    if not missing and not force:
        logger.info("Every planned series is on disk -- skipping download.")
        return {"missing_before": 0, "missing_after": 0}

    from ExtractData import download_dbt_series_for

    todo_annotated = sorted({inv[uid][0] for uid in missing} & set(annotated))
    todo_normals = sorted({inv[uid][0] for uid in missing} & set(normals))
    if force:
        todo_annotated, todo_normals = annotated, normals
    download_dbt_series_for(todo_annotated, max_gb_added=max_gb_added, raise_on_failure=True)
    download_dbt_series_for(todo_normals, max_gb_added=max_gb_added, raise_on_failure=True)

    still = planned_series(annotated + normals, inv) - series_on_disk()
    if still:
        logger.warning("%d planned series still missing (size cap reached?).", len(still))
    return {"missing_before": len(missing), "missing_after": len(still)}


@task(name="dbt-preprocess")
def preprocess(corpus_splits=("train", "validation"), force=False):
    """Bring both corpora up to date with the raw layer.

    Not retried: a failure here is a DICOM or an annotation problem, and running it
    again produces the same failure.
    """
    logger = _logger()
    on_disk = series_on_disk()
    report = {}
    for corpus, output_dir in CORPORA.items():
        expected = expected_cases(corpus, corpus_splits, on_disk)
        missing = expected - manifest_cases(output_dir)
        logger.info("%s corpus: %d series expected from disk, %d missing from its manifest.",
                    corpus, len(expected), len(missing))
        if not missing and not force:
            logger.info("%s corpus is up to date -- skipping.", corpus)
            report[corpus] = {"expected": len(expected), "missing": 0, "ran": False}
            continue

        from TransformData import preprocess_dbt_exams, preprocess_dbt_with_boxes

        paths = tables_for("file_paths", corpus_splits)
        boxes = tables_for("boxes", corpus_splits)
        if corpus == "exam":
            # Resumable: only the missing series are decoded, the manifest keeps the rest.
            preprocess_dbt_exams(labels_csv=tables_for("labels", SPLITS),
                                 file_paths_csv=paths, boxes_csv=boxes,
                                 output_dir=output_dir, skip_existing=not force)
        else:
            preprocess_dbt_with_boxes(boxes_csv=boxes, file_paths_csv=paths,
                                      output_dir=output_dir)
        left = expected - manifest_cases(output_dir)
        if left:
            logger.warning("%s corpus: %d expected series were not written (see the "
                           "preprocessing log for why).", corpus, len(left))
        report[corpus] = {"expected": len(expected), "missing": len(left), "ran": True}
    return report


@task(name="dbt-catalog")
def build_catalog():
    """Rebuild the metadata catalogue; fail the flow on an error-severity dbt test."""
    logger = _logger()
    from catalog.build import build

    result = build()
    failed = result.failed_errors
    logger.info("Catalogue: %d/%d dbt tests pass.",
                sum(c.passed for c in result.checks), len(result.checks))
    if failed:
        raise RuntimeError("catalogue tests failed: " + ", ".join(c.name for c in failed))
    return result.db_path


@flow(name="dbt-data-chain", log_prints=True)
def dbt_pipeline(start_at="tables", annotated_splits=SPLITS,
                 corpus_splits=("train", "validation"), max_normal_patients=150,
                 max_gb_added=50, seed=0, force=False):
    """Run the DBT chain from ``start_at`` to the catalogue, doing only what is missing."""
    logger = _logger()
    if start_at not in STAGES:
        raise ValueError(f"start_at must be one of {STAGES}; got {start_at!r}")
    todo = STAGES[STAGES.index(start_at):]
    logger.info("Stages to run: %s", " -> ".join(todo))

    if "tables" in todo:
        fetch_tables(force=force)
    if "download" in todo:
        download(annotated_splits, max_normal_patients, max_gb_added, seed, force=force)
    if "preprocess" in todo:
        preprocess(corpus_splits, force=force)
    return build_catalog()


# --- CLI ----------------------------------------------------------------------------

def _splits(text):
    values = tuple(s.strip() for s in text.split(",") if s.strip())
    unknown = set(values) - set(SPLITS)
    if unknown or not values:
        raise argparse.ArgumentTypeError(f"splits must be among {SPLITS}; got {text!r}")
    return values


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="start_at", default="tables", choices=STAGES)
    p.add_argument("--annotated-splits", type=_splits, default=SPLITS,
                   help="Splits whose annotated patients are downloaded (default: all three).")
    p.add_argument("--corpus-splits", type=_splits, default=("train", "validation"),
                   help="Splits the two corpora are built from (default: train,validation, "
                        "the corpora every published measurement used).")
    p.add_argument("--max-normal-patients", type=int, default=150)
    p.add_argument("--max-gb-added", type=float, default=50,
                   help="Cap on what one download call adds to the raw layer.")
    p.add_argument("--seed", type=int, default=0, help="Seed of the normal-patient sample.")
    p.add_argument("--force", action="store_true",
                   help="Re-run stages even when their plan says nothing is missing.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan, computed offline from tables, disk and manifests.")
    return p


def describe(args):
    """The plan, without running anything -- and without touching the network."""
    todo = STAGES[STAGES.index(args.start_at):]
    lines = [f"Would run {len(todo)} stage(s): {' -> '.join(todo)}", ""]
    missing_tables = [os.path.basename(p) for p in all_tables() if not os.path.exists(p)]

    if "tables" in todo:
        lines.append(f"  tables      {9 - len(missing_tables)}/9 present"
                     + (f" -> would fetch {missing_tables}" if missing_tables else
                        " -> skip" if not args.force else " -> refetch (--force)"))
    if missing_tables:
        lines += ["", "  The remaining stages plan from the tables: fetch them first."]
        return "\n".join(lines)

    on_disk = series_on_disk()
    if "download" in todo:
        annotated, normals = planned_patients(args.annotated_splits,
                                              args.max_normal_patients, args.seed)
        inv = inventory()
        planned = planned_series(annotated + normals, inv)
        missing = planned - on_disk
        patients = sorted({inv[uid][0] for uid in missing})
        lines.append(f"  download    {len(annotated)} annotated ({','.join(args.annotated_splits)})"
                     f" + {len(normals)} normal patients -> {len(planned)} series planned,"
                     f" {len(on_disk)} series on disk")
        lines.append(f"  {'':<11} -> {len(missing)} missing"
                     + (f" across {len(patients)} patient(s): {', '.join(patients[:6])}"
                        + ("..." if len(patients) > 6 else "") if missing else ", skip"))
    if "preprocess" in todo:
        for corpus, output_dir in CORPORA.items():
            expected = expected_cases(corpus, args.corpus_splits, on_disk)
            recorded = manifest_cases(output_dir)
            missing = expected - recorded
            lines.append(f"  preprocess  {corpus} corpus ({','.join(args.corpus_splits)}):"
                         f" {len(expected)} expected, {len(recorded)} in manifest"
                         f" -> {f'{len(missing)} to build' if missing else 'skip'}")
    lines.append(f"  catalog     always rebuilt -> {os.path.relpath(config.CATALOG_DB, config.ROOT)}")
    return "\n".join(lines)


if __name__ == "__main__":
    parsed = build_arg_parser().parse_args()
    setup_logging(logfile="dbt_pipeline.log")
    if parsed.dry_run:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(describe(parsed))
        sys.exit(0)
    dbt_pipeline(start_at=parsed.start_at, annotated_splits=parsed.annotated_splits,
                 corpus_splits=parsed.corpus_splits,
                 max_normal_patients=parsed.max_normal_patients,
                 max_gb_added=parsed.max_gb_added, seed=parsed.seed, force=parsed.force)
