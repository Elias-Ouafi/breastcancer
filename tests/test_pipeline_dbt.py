"""Tests for the DBT flow's decisions, not for the compute it orchestrates.

What is worth testing cheaply is the plan: which series should be on disk, which cases
each corpus should hold, and therefore whether a stage runs. A wrong plan either
re-downloads or re-decodes hours of data, or -- worse -- skips a stage whose output is
stale. The heavy modules (ExtractData needs tcia_utils, TransformData needs pydicom,
the catalogue needs dbt) are replaced by fakes in ``sys.modules``, so this file runs
with Prefect and pandas only.
"""
from __future__ import annotations

import json
import os
import types

import numpy as np
import pytest

pytest.importorskip("prefect", reason="orchestration extra not installed")
pd = pytest.importorskip("pandas")

import config  # noqa: E402
import lineage  # noqa: E402
from pipelines import dbt as flow  # noqa: E402

# P1 cancer (train, 2 views, box on lcc), P2 normal (train, 1 view), P3 benign (test),
# P4 listed with no flag set (train). Series UIDs are dotted digits, as on disk.
INVENTORY = {
    "train": [("P1", "S1", "lcc", "1.1"), ("P1", "S1", "rcc", "1.2"),
              ("P2", "S2", "rmlo", "2.1"), ("P4", "S4", "lcc", "4.1")],
    "validation": [],
    "test": [("P3", "S3", "lcc", "3.1")],
}
STATUS = {"1.1": "cancer", "1.2": "cancer", "2.1": "normal", "3.1": "benign", "4.1": None}
BOXES = {"train": [("P1", "S1", "lcc", "cancer")], "validation": [],
         "test": [("P3", "S3", "lcc", "benign")]}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Nine tables, a download directory, two corpus folders -- all under tmp_path."""
    tables = {kind: {} for kind in flow.TABLES}
    for split in flow.SPLITS:
        rows = INVENTORY[split]
        p = tmp_path / f"paths-{split}.csv"
        pd.DataFrame([{"PatientID": pid, "StudyUID": sid, "View": view.upper(),
                       "descriptive_path": "x",
                       "classic_path": f"Breast-Cancer-Screening-DBT\\{pid}\\{sid}\\{uid}\\1-1.dcm"}
                      for pid, sid, view, uid in rows],
                     columns=["PatientID", "StudyUID", "View", "descriptive_path",
                              "classic_path"]).to_csv(p, index=False)
        tables["file_paths"][split] = str(p)

        p = tmp_path / f"labels-{split}.csv"
        pd.DataFrame([{"PatientID": pid, "StudyUID": sid, "View": view,
                       "Normal": int(STATUS[uid] == "normal"),
                       "Actionable": 0,
                       "Benign": int(STATUS[uid] == "benign"),
                       "Cancer": int(STATUS[uid] == "cancer")}
                      for pid, sid, view, uid in rows],
                     columns=["PatientID", "StudyUID", "View", "Normal", "Actionable",
                              "Benign", "Cancer"]).to_csv(p, index=False)
        tables["labels"][split] = str(p)

        p = tmp_path / f"boxes-{split}.csv"
        pd.DataFrame([{"PatientID": pid, "StudyUID": sid, "View": view, "Class": cls}
                      for pid, sid, view, cls in BOXES[split]],
                     columns=["PatientID", "StudyUID", "View", "Class"]).to_csv(p, index=False)
        tables["boxes"][split] = str(p)

    tcia = tmp_path / "tcia"
    tcia.mkdir()
    (tcia / "duke_mri").mkdir()  # not a DBT series folder
    corpora = {"lesion": str(tmp_path / "dbt"), "exam": str(tmp_path / "dbt_exams")}

    monkeypatch.setattr(flow, "TABLES", tables)
    monkeypatch.setattr(flow, "CORPORA", corpora)
    monkeypatch.setattr(config, "TCIA_DIR", str(tcia))

    calls = []
    fake_extract = types.ModuleType("ExtractData")
    fake_extract.choose_normal_patients = lambda labels, n, seed: (["P2"][:n], 1)
    fake_extract.download_dbt_tables = lambda overwrite=False: calls.append(("tables",))

    def fake_download(patients, max_gb_added=None, raise_on_failure=False):
        calls.append(("download", sorted(patients)))
        return 0

    fake_extract.download_dbt_series_for = fake_download
    monkeypatch.setitem(__import__("sys").modules, "ExtractData", fake_extract)
    return types.SimpleNamespace(tmp=tmp_path, tcia=tcia, corpora=corpora, calls=calls)


def put_on_disk(world, *uids):
    for uid in uids:
        (world.tcia / uid).mkdir()
        (world.tcia / uid / "1-1.dcm").write_bytes(b"\0")


def write_manifest(output_dir, uids):
    lineage.write_manifest(output_dir, source="x", parameters={},
                           cases={uid: {"label": 0} for uid in uids})


def put_in_silver(output_dir, *uids):
    """A corpus that really holds ``uids``: a readable volume each, and the manifest."""
    os.makedirs(output_dir, exist_ok=True)
    for uid in uids:
        np.savez_compressed(os.path.join(output_dir, f"{uid}.npz"),
                            volume=np.zeros((2, 4, 4), np.float16),
                            mask=np.zeros((2, 4, 4), np.uint8))
    write_manifest(output_dir, uids)


# --- stages and plan ------------------------------------------------------------

def test_stage_order_is_the_dependency_order():
    assert flow.STAGES == ["tables", "download", "preprocess", "purge", "catalog", "publish"]


def test_rejects_an_unknown_resume_point():
    with pytest.raises(ValueError, match="start_at"):
        flow.dbt_pipeline.fn(start_at="prepocess")


def test_the_inventory_reads_the_series_folder_from_classic_path(world):
    inv = flow.inventory()
    assert inv["1.1"] == ("P1", "S1", "lcc")  # view lower-cased, backslashes handled
    assert set(inv) == {"1.1", "1.2", "2.1", "4.1", "3.1"}


def test_only_series_folders_count_as_on_disk(world):
    put_on_disk(world, "1.1")
    assert flow.series_on_disk() == {"1.1"}


@pytest.mark.parametrize("text,ok", [("train,validation", True), ("test", True),
                                     ("train,dev", False), ("", False)])
def test_split_arguments_are_validated(text, ok):
    if ok:
        assert flow._splits(text)
    else:
        with pytest.raises(Exception):
            flow._splits(text)


# --- download -----------------------------------------------------------------------

def test_download_is_skipped_without_touching_the_network_when_the_plan_is_on_disk(world):
    put_on_disk(world, "1.1", "1.2", "2.1", "3.1")  # P1, P3 annotated; P2 the normal
    result = flow.download.fn(annotated_splits=flow.SPLITS, max_normal_patients=1)
    assert result == {"missing_before": 0, "missing_after": 0}
    assert world.calls == []


def test_download_fetches_only_the_patients_with_missing_series(world):
    put_on_disk(world, "1.1", "2.1")  # P1 lacks 1.2, P3 lacks 3.1, the normal P2 is complete
    result = flow.download.fn(annotated_splits=flow.SPLITS, max_normal_patients=1)
    assert world.calls == [("download", ["P1", "P3"]), ("download", [])]
    assert result["missing_before"] == 2


def test_a_failed_download_fails_the_task_so_prefect_retries(world, monkeypatch):
    class DownloadIncomplete(RuntimeError):
        pass

    def failing(patients, max_gb_added=None, raise_on_failure=False):
        assert raise_on_failure, "the flow must ask for failures to be raised"
        raise DownloadIncomplete("1 series failed")

    monkeypatch.setattr(__import__("sys").modules["ExtractData"], "download_dbt_series_for",
                        failing)
    with pytest.raises(DownloadIncomplete):
        flow.download.fn(annotated_splits=("train",), max_normal_patients=1)


# --- corpora ------------------------------------------------------------------------

def test_the_exam_corpus_expects_labelled_listed_series_on_disk(world):
    put_on_disk(world, "1.1", "1.2", "2.1", "4.1", "3.1")
    # 4.1 has no flag set; 3.1 is on disk but in the test split, outside the corpus.
    assert flow.expected_cases("exam", ("train", "validation"), flow.series_on_disk()) == {
        "1.1", "1.2", "2.1"}


def test_the_lesion_corpus_expects_only_views_carrying_a_box(world):
    put_on_disk(world, "1.1", "1.2", "2.1", "3.1")
    assert flow.expected_cases("lesion", ("train", "validation"), flow.series_on_disk()) == {
        "1.1"}
    assert flow.expected_cases("lesion", flow.SPLITS, flow.series_on_disk()) == {"1.1", "3.1"}


def _fake_transform(monkeypatch, calls):
    fake = types.ModuleType("TransformData")
    fake.preprocess_dbt_exams = lambda **kw: calls.append(("exam", kw["skip_existing"]))
    fake.preprocess_dbt_with_boxes = lambda **kw: calls.append(("lesion",))
    monkeypatch.setitem(__import__("sys").modules, "TransformData", fake)


def test_an_up_to_date_corpus_is_not_rebuilt(world, monkeypatch):
    put_on_disk(world, "1.1", "1.2", "2.1")
    write_manifest(world.corpora["exam"], ["1.1", "1.2", "2.1"])
    write_manifest(world.corpora["lesion"], ["1.1"])
    calls = []
    _fake_transform(monkeypatch, calls)

    report = flow.preprocess.fn(corpus_splits=("train", "validation"))

    assert calls == []
    assert report["exam"] == {"expected": 3, "missing": 0, "ran": False}


def test_a_stale_exam_corpus_is_resumed_not_rebuilt(world, monkeypatch):
    """The failure this flow exists to prevent: a folder that exists and is stale."""
    put_on_disk(world, "1.1", "1.2", "2.1")
    write_manifest(world.corpora["exam"], ["1.1", "1.2"])  # 2.1 downloaded afterwards
    write_manifest(world.corpora["lesion"], ["1.1"])
    calls = []
    _fake_transform(monkeypatch, calls)

    report = flow.preprocess.fn(corpus_splits=("train", "validation"))

    assert calls == [("exam", True)]  # skip_existing: only the new series is decoded
    assert report["exam"]["ran"] and report["lesion"]["ran"] is False


# --- catalogue ----------------------------------------------------------------------

def _fake_catalog(monkeypatch, failed):
    fake = types.ModuleType("catalog.build")
    check = types.SimpleNamespace(name="exam_corpus_label_matches_labels", passed=not failed)
    fake.build = lambda: types.SimpleNamespace(
        db_path="catalog.duckdb", checks=[check], failed_errors=[check] if failed else [])
    monkeypatch.setitem(__import__("sys").modules, "catalog.build", fake)


def test_the_catalogue_stage_fails_the_flow_on_an_error_test(monkeypatch):
    _fake_catalog(monkeypatch, failed=True)
    with pytest.raises(RuntimeError, match="exam_corpus_label_matches_labels"):
        flow.build_catalog.fn()


def test_the_catalogue_stage_returns_the_database_when_tests_pass(monkeypatch):
    _fake_catalog(monkeypatch, failed=False)
    assert flow.build_catalog.fn() == "catalog.duckdb"


# --- publish ----------------------------------------------------------------------

def test_publish_is_skipped_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("BREASTCANCER_S3_ENDPOINT", raising=False)
    assert flow.publish.fn() is None


def test_publish_syncs_to_the_configured_bucket(monkeypatch):
    monkeypatch.setenv("BREASTCANCER_S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setenv("BREASTCANCER_S3_BUCKET", "lake")
    seen = {}
    fake = types.ModuleType("objectstore.sync")
    fake.DEFAULT_BUCKET = "breastcancer"
    fake.client_from_env = lambda endpoint: seen.setdefault("endpoint", endpoint)
    fake.publishable_files = lambda: ["file"]
    fake.run_sync = lambda client, bucket, files: seen.update(bucket=bucket, files=files) or \
        types.SimpleNamespace(summary=lambda: {"new": 1})
    monkeypatch.setitem(__import__("sys").modules, "objectstore.sync", fake)

    assert flow.publish.fn() == {"new": 1}
    assert seen == {"endpoint": "http://127.0.0.1:9000", "bucket": "lake", "files": ["file"]}


def test_starting_at_publish_does_not_rebuild_the_catalogue(monkeypatch):
    calls = []
    monkeypatch.setattr(flow, "build_catalog", lambda: calls.append("catalog"))
    monkeypatch.setattr(flow, "publish", lambda: calls.append("publish"))
    flow.dbt_pipeline.fn(start_at="publish")
    assert calls == ["publish"]


# --- dry run ------------------------------------------------------------------------

def _args(**overrides):
    args = flow.build_arg_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_the_dry_run_plans_from_tables_disk_and_manifests(world):
    put_on_disk(world, "1.1", "2.1")
    write_manifest(world.corpora["exam"], ["1.1"])
    plan = flow.describe(_args(max_normal_patients=1))

    assert "9/9 present -> skip" in plan
    assert "2 missing across 2 patient(s): P1, P3" in plan
    assert "exam corpus (train,validation): 2 expected, 1 in manifest -> 1 to build" in plan
    assert "lesion corpus (train,validation): 1 expected, 0 in manifest -> 1 to build" in plan
    assert "publish" in plan
    assert world.calls == []  # no network, no download


def test_the_dry_run_stops_at_the_tables_when_they_are_missing(world):
    os.remove(flow.TABLES["labels"]["test"])
    plan = flow.describe(_args())
    assert "8/9 present" in plan and "fetch them first" in plan


def test_the_manifest_reader_tolerates_a_corpus_never_built(world):
    assert flow.manifest_cases(world.corpora["exam"]) == set()
    json.dumps(sorted(flow.manifest_cases(world.corpora["exam"])))


# --- purge: bronze is a transit zone ------------------------------------------------

CORPUS_SPLITS = ("train", "validation")


def test_a_series_is_purgeable_only_once_every_corpus_reading_it_holds_it(world):
    """1.1 is read by both corpora (it carries a box); 1.2 and 2.1 only by the exam one.

    The exam corpus alone is complete, so purging on it would delete 1.1 before the
    lesion corpus had decoded it -- the failure the "every reader" rule exists for.
    """
    put_on_disk(world, "1.1", "1.2", "2.1")
    put_in_silver(world.corpora["exam"], "1.1", "1.2", "2.1")

    plan = flow.purgeable_series(CORPUS_SPLITS, flow.series_on_disk())
    assert set(plan) == {"1.2", "2.1"}
    assert plan["1.2"] == [os.path.join(world.corpora["exam"], "1.2.npz")]

    put_in_silver(world.corpora["lesion"], "1.1")
    plan = flow.purgeable_series(CORPUS_SPLITS, flow.series_on_disk())
    assert set(plan) == {"1.1", "1.2", "2.1"}
    assert plan["1.1"] == [os.path.join(world.corpora["lesion"], "1.1.npz"),
                           os.path.join(world.corpora["exam"], "1.1.npz")]


def test_a_series_no_corpus_reads_is_never_purgeable(world):
    """4.1 has no labels flag and 3.1 is in the test split: neither was processed, so
    deleting them would lose data that exists nowhere else."""
    put_on_disk(world, "4.1", "3.1")
    put_in_silver(world.corpora["exam"], "1.1")
    put_in_silver(world.corpora["lesion"], "1.1")
    assert flow.purgeable_series(CORPUS_SPLITS, flow.series_on_disk()) == {}


def test_a_manifest_entry_without_its_volume_does_not_count_as_held(world):
    put_on_disk(world, "2.1")
    write_manifest(world.corpora["exam"], ["2.1"])  # the manifest promises, no .npz exists
    assert flow.purgeable_series(CORPUS_SPLITS, flow.series_on_disk()) == {}


def test_purge_deletes_the_promoted_series_and_keeps_the_rest(world):
    put_on_disk(world, "1.1", "1.2", "2.1", "4.1")
    put_in_silver(world.corpora["exam"], "1.1", "1.2", "2.1")

    report = flow.purge.fn(corpus_splits=CORPUS_SPLITS)

    # 1.1 waits for the lesion corpus; 4.1 was never processed.
    assert flow.series_on_disk() == {"1.1", "4.1"}
    assert report["purged"] == 2 and report["purgeable"] == 2
    assert report["freed_bytes"] == 2 and report["kept"] is False


def test_keep_bronze_deletes_nothing(world):
    put_on_disk(world, "1.2", "2.1")
    put_in_silver(world.corpora["exam"], "1.2", "2.1")

    report = flow.purge.fn(corpus_splits=CORPUS_SPLITS, keep_bronze=True)

    assert flow.series_on_disk() == {"1.2", "2.1"}
    assert report == {"purgeable": 2, "purged": 0, "freed_bytes": 0, "kept": True}


def test_purge_refuses_a_series_whose_volume_is_unreadable(world):
    put_on_disk(world, "2.1")
    put_in_silver(world.corpora["exam"], "2.1")
    open(os.path.join(world.corpora["exam"], "2.1.npz"), "wb").close()  # emptied afterwards

    report = flow.purge.fn(corpus_splits=CORPUS_SPLITS)

    assert flow.series_on_disk() == {"2.1"}
    assert report["purged"] == 0


def test_download_counts_the_series_already_in_silver_as_present(world):
    """After a purge nothing is in bronze. Reading that as "nothing downloaded" would
    fetch the whole collection again on every run."""
    put_in_silver(world.corpora["exam"], "1.1", "1.2", "2.1")
    put_in_silver(world.corpora["lesion"], "1.1")
    assert flow.series_on_disk() == set()

    result = flow.download.fn(annotated_splits=CORPUS_SPLITS, max_normal_patients=1)

    assert result == {"missing_before": 0, "missing_after": 0}
    assert world.calls == []


def test_the_dry_run_reports_what_the_purge_would_delete(world):
    put_on_disk(world, "1.2", "2.1")
    put_in_silver(world.corpora["exam"], "1.2", "2.1")

    plan = flow.describe(_args(max_normal_patients=1))
    assert "purge       2 of 2 bronze series" in plan and "-> 2 to delete" in plan
    assert "-> kept (--keep-bronze)" in flow.describe(
        _args(max_normal_patients=1, keep_bronze=True))
    assert flow.series_on_disk() == {"1.2", "2.1"}  # a dry run deletes nothing
