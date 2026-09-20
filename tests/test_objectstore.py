"""Publishing to object storage, against an in-memory S3 (moto).

The properties that matter are the ones a second run relies on: nothing unchanged is
sent again, a changed file is, the MD5 comparison still holds for multipart uploads
(whose ETag is not an MD5), planning never writes, and nothing in the bucket is ever
deleted.
"""
from __future__ import annotations

import json
import os

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from objectstore.sync import (  # noqa: E402
    MD5_METADATA_KEY,
    LocalFile,
    plan_sync,
    publishable_files,
    run_sync,
)

BUCKET = "test-bucket"


@pytest.fixture
def s3(monkeypatch):
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(var, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with moto.mock_aws():
        yield boto3.client("s3", region_name="us-east-1")


@pytest.fixture
def layers(tmp_path):
    """A miniature data tree: two tables, a manifest, a Parquet table, two series."""
    tcia = tmp_path / "tcia"
    tcia.mkdir()
    (tcia / "BCS-DBT-labels-train-v2.csv").write_bytes(b"PatientID,View\nP1,lcc\n")
    (tcia / "BCS-DBT-boxes-train.csv").write_bytes(b"PatientID,View\nP1,lcc\n")
    (tcia / "Annotation_Boxes.xlsx").write_bytes(b"not a BCS-DBT table")
    for uid in ("1.2.3", "1.2.4"):
        (tcia / uid).mkdir()
        (tcia / uid / "1-1.dcm").write_bytes(uid.encode() * 100)
    (tcia / "duke_mri").mkdir()
    corpus = tmp_path / "dbt_exams"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(json.dumps({"n_cases": 1}))
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    (parquet / "dim_patient.parquet").write_bytes(b"PAR1 fake")

    def files(dicom_sample=0):
        return publishable_files(dicom_sample=dicom_sample, tcia_dir=str(tcia),
                                 corpora={"dbt_exams": str(corpus)},
                                 parquet_dir=str(parquet))
    files.tcia = tcia
    return files


def keys(files):
    return sorted(f.key for f in files)


def test_the_keys_mirror_the_local_layers(layers):
    assert keys(layers()) == [
        "bronze/tcia/tables/BCS-DBT-boxes-train.csv",
        "bronze/tcia/tables/BCS-DBT-labels-train-v2.csv",
        "gold/catalog/dim_patient.parquet",
        "silver/dbt_exams/manifest.json",
    ]


def test_dicom_is_published_only_for_the_requested_sample(layers):
    assert [k for k in keys(layers(dicom_sample=1)) if "series" in k] == [
        "bronze/tcia/series/1.2.3/1-1.dcm"]


def test_planning_against_a_missing_bucket_sends_nothing_and_creates_nothing(s3, layers):
    plan = plan_sync(s3, BUCKET, layers())
    assert plan.summary()["new"] == 4
    assert BUCKET not in [b["Name"] for b in s3.list_buckets()["Buckets"]]


def test_a_second_sync_uploads_nothing(s3, layers):
    first = run_sync(s3, BUCKET, layers())
    second = run_sync(s3, BUCKET, layers())
    assert first.summary()["new"] == 4
    assert second.summary() == {"new": 0, "changed": 0, "unchanged": 4, "remote_only": 0,
                                "bytes_to_upload": 0}


def test_a_changed_file_is_uploaded_again(s3, layers):
    run_sync(s3, BUCKET, layers())
    (layers.tcia / "BCS-DBT-labels-train-v2.csv").write_bytes(b"PatientID,View\nP1,rcc\n")
    plan = run_sync(s3, BUCKET, layers())
    assert [f.key for f in plan.upload_changed] == ["bronze/tcia/tables/BCS-DBT-labels-train-v2.csv"]
    body = s3.get_object(Bucket=BUCKET, Key="bronze/tcia/tables/BCS-DBT-labels-train-v2.csv")
    assert body["Body"].read() == b"PatientID,View\nP1,rcc\n"


def test_same_size_different_content_is_detected(s3, layers):
    """A size comparison alone would miss this edit: the MD5 is what catches it."""
    run_sync(s3, BUCKET, layers())
    (layers.tcia / "BCS-DBT-labels-train-v2.csv").write_bytes(b"PatientID,View\nP2,lcc\n")
    assert plan_sync(s3, BUCKET, layers()).summary()["changed"] == 1


def test_multipart_uploads_are_compared_by_stored_md5(s3, tmp_path, monkeypatch):
    """Above the multipart threshold the ETag is not an MD5; the metadata is."""
    import objectstore.sync as sync

    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(6 * 1024 * 1024))
    files = [LocalFile(str(big), "bronze/big.bin", big.stat().st_size)]
    monkeypatch.setattr(sync, "MULTIPART_THRESHOLD", 5 * 1024 * 1024)

    run_sync(s3, BUCKET, files, multipart_threshold=5 * 1024 * 1024)
    head = s3.head_object(Bucket=BUCKET, Key="bronze/big.bin")
    assert "-" in head["ETag"]  # a multipart ETag, not an MD5
    assert MD5_METADATA_KEY in head["Metadata"]
    assert plan_sync(s3, BUCKET, files).summary()["unchanged"] == 1


def test_objects_only_in_the_bucket_are_reported_and_kept(s3, layers):
    run_sync(s3, BUCKET, layers())
    s3.put_object(Bucket=BUCKET, Key="bronze/tcia/tables/old-table.csv", Body=b"x")
    plan = run_sync(s3, BUCKET, layers())
    assert plan.remote_only == ["bronze/tcia/tables/old-table.csv"]
    s3.head_object(Bucket=BUCKET, Key="bronze/tcia/tables/old-table.csv")  # still there


def test_each_sync_is_recorded_in_the_bucket(s3, layers):
    run_sync(s3, BUCKET, layers())
    record = json.loads(s3.get_object(Bucket=BUCKET, Key="_meta/last_sync.json")["Body"].read())
    assert record["new"] == 4 and len(record["keys"]) == 4
    assert "synced_at" in record and "git_revision" in record


def test_the_cli_plans_and_syncs(s3, layers, monkeypatch, capsys):
    import objectstore.__main__ as cli

    monkeypatch.setattr(cli, "client_from_env", lambda endpoint=None: s3)
    monkeypatch.setattr(cli, "publishable_files", lambda dicom_sample=0: layers(dicom_sample))
    assert cli.main(["--bucket", BUCKET, "plan"]) == 0
    assert "would upload 4 new" in capsys.readouterr().out
    assert cli.main(["--bucket", BUCKET, "sync"]) == 0
    assert "uploaded 4 new" in capsys.readouterr().out
    assert cli.main(["--bucket", BUCKET, "sync", "--dicom-sample", "2"]) == 0
    assert "uploaded 2 new + 0 changed" in capsys.readouterr().out  # only the two series
