"""The metadata catalogue, built from miniature BCS-DBT tables through the real dbt project.

The real build passes all 36 dbt tests, which proves nothing about the tests: a query
that can never return a row passes too. So each singular test that guards a label is
shown failing here on a fixture with exactly that defect injected, one generic grain
test is shown failing on a duplicated key, and the marts are checked against counts
worked out by hand.
"""
from __future__ import annotations

import json
import os

import duckdb
import pytest

from catalog.__main__ import main as catalog_main
from catalog.build import SERIES_FOLDER, Sources, build

# Three patients, one per split. P1 has a cancer box on one view; P2 is normal; P3 has
# a benign box. Series UIDs are dotted digits, as TCIA names its folders.
FILE_PATHS = {
    "train": [("P1", "S1", "lcc", "1.1"), ("P1", "S1", "lmlo", "1.2"),
              ("P1", "S1", "rcc", "1.3")],
    "validation": [("P2", "S2", "rcc", "2.1"), ("P2", "S2", "rmlo", "2.2")],
    "test": [("P3", "S3", "lcc", "3.1")],
}
STATUS = {"1.1": "cancer", "1.2": "cancer", "1.3": "normal",
          "2.1": "normal", "2.2": "normal", "3.1": "benign"}
BOXES = {
    "train": [("P1", "S1", "lcc", "cancer")],
    "validation": [],
    "test": [("P3", "S3", "lcc", "benign")],
}


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")


def _make_sources(root, *, exam_cases=None, lesion_cases=None, predictions=None,
                  on_disk=("1.1", "1.2", "1.3", "2.1", "2.2")):
    tcia = root / "tcia"
    tcia.mkdir()
    labels, boxes, paths = [], [], []
    for split, series in FILE_PATHS.items():
        p = tcia / f"BCS-DBT-file-paths-{split}-v2.csv"
        _write_csv(p, ["PatientID", "StudyUID", "View", "descriptive_path", "classic_path"],
                   [(pid, sid, view, "x", f"Breast-Cancer-Screening-DBT/{pid}/{sid}/{uid}/1-1.dcm")
                    for pid, sid, view, uid in series])
        paths.append(str(p))

        p = tcia / f"BCS-DBT-labels-{split}-PHASE-2.csv"
        rows = []
        for pid, sid, view, uid in series:
            s = STATUS[uid]
            rows.append((pid, sid, view, int(s == "normal"), int(s == "actionable"),
                         int(s == "benign"), int(s == "cancer")))
        _write_csv(p, ["PatientID", "StudyUID", "View", "Normal", "Actionable", "Benign",
                       "Cancer"], rows)
        labels.append(str(p))

        p = tcia / f"BCS-DBT-boxes-{split}.csv"
        header = ["PatientID", "StudyUID", "View", "Subject", "Slice", "X", "Y", "Width",
                  "Height", "Class", "AD"]
        extra = [] if split == "train" else ["VolumeSlices"]  # as published
        _write_csv(p, header + extra,
                   [(pid, sid, view, 0, 10, 5, 5, 20, 20, cls, 0, *([60] if extra else []))
                    for pid, sid, view, cls in BOXES[split]])
        boxes.append(str(p))

    for uid in on_disk:
        folder = tcia / uid
        folder.mkdir()
        (folder / "1-1.dcm").write_bytes(b"\0" * 1000)
    (tcia / "duke_mri").mkdir()  # not a DBT series folder: must be ignored

    def manifest(name, cases):
        path = root / f"{name}_manifest.json"
        path.write_text(json.dumps({
            "generated_at": "2026-09-13T00:00:00+00:00", "git_revision": "abc1234",
            "source": "data/raw_data/tcia", "output_dir": f"data/preprocessed_data/{name}",
            "parameters": {}, "n_cases": len(cases), "validation_warnings": [],
            "cases": cases,
        }))
        return str(path)

    if exam_cases is None:
        exam_cases = {
            uid: {"shape": [60, 384, 384], "case_id": pid, "study_uid": sid, "view": view,
                  "exam_status": STATUS[uid], "label": int(STATUS[uid] == "cancer"),
                  "warnings": [], "mirrored": False}
            for series in FILE_PATHS.values() for pid, sid, view, uid in series
            if uid in on_disk
        }
    if lesion_cases is None:
        lesion_cases = {"1.1": {"shape": [37, 200, 200], "case_id": "P1", "study_uid": "S1",
                                "view": "lcc", "lesion_class": "cancer", "label": 1,
                                "warnings": []}}
    pred_path = root / "cv_predictions.csv"
    _write_csv(pred_path, ["patient", "label", "score", "fold"],
               predictions if predictions is not None
               else [("P1", 1, 0.9, 0), ("P2", 0, 0.1, 1)])

    return Sources(labels=labels, boxes=boxes, file_paths=paths, tcia_dir=str(tcia),
                   manifests={"dbt": manifest("dbt", lesion_cases),
                              "dbt_exams": manifest("dbt_exams", exam_cases)},
                   predictions=str(pred_path))


def _build(tmp_path, **kwargs):
    sources = _make_sources(tmp_path, **kwargs)
    db = str(tmp_path / "catalog.duckdb")
    result = build(db_path=db, sources=sources, parquet_dir=str(tmp_path / "parquet"))
    return result, duckdb.connect(db, read_only=True)


def _failing(result):
    return {c.name for c in result.checks if not c.passed}


def test_clean_fixture_builds_and_passes_every_check(tmp_path):
    result, con = _build(tmp_path)
    assert _failing(result) == set()
    assert result.row_counts["mart.fct_series"] == 6
    assert result.row_counts["mart.dim_patient"] == 3
    assert result.row_counts["raw.disk_series"] == 5  # duke_mri/ is not a series
    con.close()


def test_series_uid_split_and_worst_status_are_derived(tmp_path):
    _, con = _build(tmp_path)
    rows = con.execute("""
        SELECT series_uid, split, view_position, laterality, view_status, n_cancer_boxes
        FROM mart.fct_series ORDER BY series_uid""").fetchall()
    assert rows[0] == ("1.1", "train", "cc", "l", "cancer", 1)
    assert {r[0]: r[1] for r in rows}["2.1"] == "validation"

    patients = dict(con.execute(
        "SELECT patient_id, status FROM mart.dim_patient").fetchall())
    # P1 has one normal view and two cancer views: read at its worst.
    assert patients == {"P1": "cancer", "P2": "normal", "P3": "benign"}
    con.close()


def test_disk_corpus_and_predictions_are_joined(tmp_path):
    _, con = _build(tmp_path)
    p1 = con.execute("""
        SELECT n_series_on_disk, fully_on_disk, n_series_in_exam_corpus,
               n_series_in_lesion_corpus, examclf_score
        FROM mart.dim_patient WHERE patient_id = 'P1'""").fetchone()
    assert p1 == (3, True, 3, 1, 0.9)
    p3 = con.execute("SELECT n_series_on_disk, examclf_score FROM mart.dim_patient "
                     "WHERE patient_id = 'P3'").fetchone()
    assert p3 == (0, None)
    con.close()


def test_train_boxes_without_volume_slices_column_still_load(tmp_path):
    _, con = _build(tmp_path)
    assert con.execute("SELECT volume_slices FROM stg.dbt_boxes WHERE split = 'train'"
                       ).fetchone() == (None,)
    con.close()


def test_exam_label_disagreeing_with_the_labels_table_fails(tmp_path):
    sources_cases = {
        "2.1": {"shape": [60, 384, 384], "case_id": "P2", "study_uid": "S2", "view": "rcc",
                "exam_status": "normal", "label": 1, "warnings": []},  # label says cancer
    }
    result, con = _build(tmp_path, exam_cases=sources_cases,
                         predictions=[("P2", 1, 0.5, 0)])
    assert "exam_corpus_label_matches_labels" in _failing(result)
    assert [c.name for c in result.failed_errors] == ["exam_corpus_label_matches_labels"]
    assert con.execute("SELECT series_uid FROM qa.exam_corpus_label_matches_labels"
                       ).fetchall() == [("2.1",)]
    con.close()


def test_lesion_label_disagreeing_with_the_boxes_fails(tmp_path):
    result, _ = _build(tmp_path, lesion_cases={
        "1.1": {"shape": [37, 200, 200], "case_id": "P1", "study_uid": "S1", "view": "lcc",
                "lesion_class": "benign", "label": 0, "warnings": []},  # box says cancer
    })
    assert "lesion_corpus_label_matches_boxes" in _failing(result)


def test_case_attached_to_the_wrong_patient_fails(tmp_path):
    result, _ = _build(tmp_path, lesion_cases={
        "1.1": {"shape": [37, 200, 200], "case_id": "P2", "study_uid": "S1", "view": "lcc",
                "lesion_class": "cancer", "label": 1, "warnings": []},
    })
    assert "corpus_cases_match_inventory" in _failing(result)


def test_prediction_for_a_patient_outside_the_corpus_fails(tmp_path):
    result, _ = _build(tmp_path, predictions=[("P1", 1, 0.9, 0), ("P9", 0, 0.2, 1)])
    assert "predictions_patients_in_exam_corpus" in _failing(result)


def test_prediction_label_disagreeing_with_the_corpus_fails(tmp_path):
    result, _ = _build(tmp_path, predictions=[("P1", 0, 0.9, 0)])
    assert "predictions_label_matches_corpus" in _failing(result)


def test_missing_download_is_a_warning_not_an_error(tmp_path):
    # 1.3 preprocessed but its raw folder is gone: rebuildable no more, labels still right.
    result, _ = _build(tmp_path, on_disk=("1.1", "1.2", "2.1", "2.2"), exam_cases={
        "1.3": {"shape": [60, 384, 384], "case_id": "P1", "study_uid": "S1", "view": "rcc",
                "exam_status": "normal", "label": 0, "warnings": []},
    }, predictions=[("P1", 0, 0.9, 0)])
    assert _failing(result) == {"corpus_series_on_disk"}
    assert result.failed_errors == []


def test_optional_sources_may_be_absent(tmp_path):
    sources = _make_sources(tmp_path)
    sources.manifests = {"dbt": str(tmp_path / "nope.json")}
    sources.predictions = str(tmp_path / "nope.csv")
    sources.tcia_dir = str(tmp_path / "no_downloads")
    result = build(db_path=str(tmp_path / "c.duckdb"), sources=sources, parquet_dir=None)
    assert result.row_counts["raw.manifest_cases"] == 0
    assert result.row_counts["mart.dim_patient"] == 3
    assert result.failed_errors == []


def test_missing_required_tables_name_the_fix(tmp_path):
    sources = _make_sources(tmp_path)
    sources.labels = [str(tmp_path / "absent.csv")]
    with pytest.raises(FileNotFoundError, match="download_dbt_tables"):
        build(db_path=str(tmp_path / "c.duckdb"), sources=sources, parquet_dir=None)


def test_a_failed_build_leaves_the_previous_catalogue_intact(tmp_path):
    db = str(tmp_path / "catalog.duckdb")
    sources = _make_sources(tmp_path)
    build(db_path=db, sources=sources, parquet_dir=None)
    before = os.path.getsize(db)

    sources.file_paths = [str(tmp_path / "absent.csv")]
    with pytest.raises(FileNotFoundError):
        build(db_path=db, sources=sources, parquet_dir=None)

    assert os.path.getsize(db) == before
    with duckdb.connect(db, read_only=True) as con:
        assert con.execute("SELECT count(*) FROM mart.fct_series").fetchone() == (6,)
    assert [f for f in os.listdir(tmp_path) if f.endswith(".duckdb")] == ["catalog.duckdb"]
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".building-")]


def test_staging_views_still_bind_after_the_catalogue_is_moved_into_place(tmp_path):
    """dbt-duckdb qualifies every view with the database name, which DuckDB takes from
    the file stem. A catalogue built under a temporary file name kept views pointing at
    that name once renamed, and every one of them failed to bind."""
    _, con = _build(tmp_path)
    views = con.execute("SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'stg' AND table_type = 'VIEW'").fetchall()
    assert len(views) == 7
    for (view,) in views:
        con.execute(f"SELECT count(*) FROM stg.{view}").fetchone()
    con.close()


def test_a_generic_dbt_test_fails_on_a_duplicated_key(tmp_path):
    """The grain tests declared in YAML must be able to fail, like the singular ones."""
    sources = _make_sources(tmp_path)
    with open(sources.labels[0], "a", encoding="utf-8", newline="\n") as fh:
        fh.write("P1,S1,lcc,0,0,0,1\n")  # the same (patient, study, view) twice
    result = build(db_path=str(tmp_path / "c.duckdb"), sources=sources, parquet_dir=None)
    failing = {c.name for c in result.failed_errors}
    assert "unique_stg_dbt_labels_patient_id_study_uid_view" in failing


def test_marts_are_exported_to_parquet(tmp_path):
    _build(tmp_path)
    exported = sorted(os.listdir(tmp_path / "parquet"))
    assert exported == ["collection_coverage.parquet", "corpus_summary.parquet",
                        "dim_patient.parquet", "fct_series.parquet"]
    n = duckdb.sql(f"SELECT count(*) FROM '{(tmp_path / 'parquet' / 'fct_series.parquet').as_posix()}'"
                   ).fetchone()[0]
    assert n == 6


def test_series_folder_pattern():
    assert SERIES_FOLDER.match("1.2.826.0.1.3680043.8.498.100")
    assert not SERIES_FOLDER.match("duke_mri")
    assert not SERIES_FOLDER.match("BCS-DBT-boxes-train.csv")


def test_example_queries_run_against_a_built_catalogue(tmp_path, capsys):
    """The documented queries are part of the interface: a renamed column breaks them."""
    _, con = _build(tmp_path)
    con.close()
    queries = os.path.join(os.path.dirname(os.path.dirname(__file__)), "catalog", "queries")
    files = sorted(f for f in os.listdir(queries) if f.endswith(".sql"))
    assert files
    for name in files:
        assert catalog_main(["--db", str(tmp_path / "catalog.duckdb"), "query",
                             "--file", os.path.join(queries, name)]) == 0


def test_cli_checks_exit_code_reflects_errors(tmp_path, capsys):
    _build(tmp_path, predictions=[("P1", 0, 0.9, 0)])
    assert catalog_main(["--db", str(tmp_path / "catalog.duckdb"), "checks"]) == 1
    assert "predictions_label_matches_corpus" in capsys.readouterr().out
