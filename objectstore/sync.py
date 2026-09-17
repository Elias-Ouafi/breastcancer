"""Plan and run an idempotent upload of local files to an S3-compatible bucket."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import config

log = logging.getLogger(__name__)

# Above this size boto3 uploads in parts, and the object's ETag stops being the file's
# MD5. The MD5 is therefore also stored as object metadata, which is what is compared
# for large files -- listing gives the ETag for free, metadata costs one HEAD request.
MULTIPART_THRESHOLD = 64 * 1024 * 1024
MD5_METADATA_KEY = "md5"

ENV_ENDPOINT = "BREASTCANCER_S3_ENDPOINT"
ENV_BUCKET = "BREASTCANCER_S3_BUCKET"
DEFAULT_BUCKET = "breastcancer"


@dataclass(frozen=True)
class LocalFile:
    path: str
    key: str
    size: int


@dataclass
class Plan:
    upload_new: list[LocalFile] = field(default_factory=list)
    upload_changed: list[LocalFile] = field(default_factory=list)
    unchanged: list[LocalFile] = field(default_factory=list)
    remote_only: list[str] = field(default_factory=list)

    @property
    def to_upload(self) -> list[LocalFile]:
        return self.upload_new + self.upload_changed

    @property
    def bytes_to_upload(self) -> int:
        return sum(f.size for f in self.to_upload)

    def summary(self) -> dict:
        return {"new": len(self.upload_new), "changed": len(self.upload_changed),
                "unchanged": len(self.unchanged), "remote_only": len(self.remote_only),
                "bytes_to_upload": self.bytes_to_upload}


# --- What to publish --------------------------------------------------------------

def _series_folders(tcia_dir):
    if not os.path.isdir(tcia_dir):
        return []
    return sorted(e.name for e in os.scandir(tcia_dir)
                  if e.is_dir() and e.name.replace(".", "").isdigit() and "." in e.name)


def publishable_files(dicom_sample: int = 0, tcia_dir: str | None = None,
                      corpora: dict | None = None,
                      parquet_dir: str | None = None) -> list[LocalFile]:
    """Local files to publish, with their bucket keys. Missing sources are skipped."""
    tcia_dir = tcia_dir or config.TCIA_DIR
    corpora = corpora or {"dbt": config.DBT_PREPROCESSED_DIR,
                          "dbt_exams": config.DBT_EXAMS_PREPROCESSED_DIR}
    parquet_dir = parquet_dir or config.CATALOG_PARQUET_DIR
    files = []

    def add(path, key):
        if os.path.isfile(path):
            files.append(LocalFile(path, key, os.path.getsize(path)))

    if os.path.isdir(tcia_dir):
        for name in sorted(os.listdir(tcia_dir)):
            if name.startswith("BCS-DBT-") and name.endswith(".csv"):
                add(os.path.join(tcia_dir, name), f"raw/tcia/tables/{name}")
    for corpus, folder in corpora.items():
        add(os.path.join(folder, "manifest.json"), f"preprocessed/{corpus}/manifest.json")
    if os.path.isdir(parquet_dir):
        for name in sorted(os.listdir(parquet_dir)):
            if name.endswith(".parquet"):
                add(os.path.join(parquet_dir, name), f"curated/catalog/{name}")
    for uid in _series_folders(tcia_dir)[:dicom_sample]:
        for root, _, names in os.walk(os.path.join(tcia_dir, uid)):
            for name in sorted(names):
                rel = os.path.relpath(os.path.join(root, name), tcia_dir).replace(os.sep, "/")
                add(os.path.join(root, name), f"raw/tcia/series/{rel}")
    return files


# --- Client -------------------------------------------------------------------------

def client_from_env(endpoint: str | None = None):
    """A boto3 S3 client for the configured endpoint.

    Credentials come from the standard AWS variables (AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY) or any other source boto3 reads -- never from this code.
    Without an endpoint, boto3 talks to AWS S3 itself.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError('object storage needs boto3: pip install -e ".[storage]"') from exc
    endpoint = endpoint or os.environ.get(ENV_ENDPOINT) or None
    return boto3.client(
        "s3", endpoint_url=endpoint,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(connect_timeout=10, read_timeout=120, retries={"max_attempts": 5}),
    )


def ensure_bucket(client, bucket: str) -> None:
    from botocore.exceptions import ClientError

    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)
        log.info("Created bucket %s", bucket)


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_objects(client, bucket: str, prefixes: set[str]) -> dict[str, dict]:
    """Size and ETag of every object under ``prefixes``; empty for a bucket not created yet
    (planning must not create it)."""
    from botocore.exceptions import ClientError

    objects = {}
    paginator = client.get_paginator("list_objects_v2")
    try:
        for prefix in sorted(prefixes):
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    objects[obj["Key"]] = {"size": obj["Size"],
                                           "etag": obj["ETag"].strip('"')}
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchBucket":
            raise
    return objects


def _same_content(client, bucket, local: LocalFile, remote: dict) -> bool:
    if remote["size"] != local.size:
        return False
    md5 = _md5(local.path)
    if local.size < MULTIPART_THRESHOLD:
        return remote["etag"] == md5
    head = client.head_object(Bucket=bucket, Key=local.key)
    return head.get("Metadata", {}).get(MD5_METADATA_KEY) == md5


def plan_sync(client, bucket: str, files: list[LocalFile]) -> Plan:
    """Compare local files with the bucket, sending nothing."""
    prefixes = {f.key.split("/", 1)[0] + "/" for f in files} or {""}
    remote = _remote_objects(client, bucket, prefixes)
    plan = Plan()
    local_keys = set()
    for f in files:
        local_keys.add(f.key)
        if f.key not in remote:
            plan.upload_new.append(f)
        elif _same_content(client, bucket, f, remote[f.key]):
            plan.unchanged.append(f)
        else:
            plan.upload_changed.append(f)
    plan.remote_only = sorted(k for k in remote if k not in local_keys
                              and not k.startswith("_meta/"))
    return plan


def run_sync(client, bucket: str, files: list[LocalFile],
             multipart_threshold: int = MULTIPART_THRESHOLD) -> Plan:
    """Upload what the plan says is new or changed, then record the sync in the bucket."""
    from boto3.s3.transfer import TransferConfig

    ensure_bucket(client, bucket)
    plan = plan_sync(client, bucket, files)
    transfer = TransferConfig(multipart_threshold=multipart_threshold)
    for f in plan.to_upload:
        client.upload_file(f.path, bucket, f.key, Config=transfer,
                           ExtraArgs={"Metadata": {MD5_METADATA_KEY: _md5(f.path)}})
        log.info("uploaded %s (%d bytes)", f.key, f.size)

    from lineage import git_revision

    record = {
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        **plan.summary(),
        "keys": sorted(f.key for f in files),
    }
    client.put_object(Bucket=bucket, Key="_meta/last_sync.json",
                      Body=json.dumps(record, indent=2).encode("utf-8"),
                      ContentType="application/json")
    if plan.remote_only:
        log.warning("%d object(s) exist only in the bucket; left in place.",
                    len(plan.remote_only))
    return plan
