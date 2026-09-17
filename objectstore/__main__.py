"""Command line for object storage publishing.

    python -m objectstore plan [--dicom-sample N]
    python -m objectstore sync [--dicom-sample N]
    python -m objectstore ls [--prefix curated/]

The endpoint and bucket come from BREASTCANCER_S3_ENDPOINT (e.g. http://127.0.0.1:9000
for the MinIO service of docker-compose.yml) and BREASTCANCER_S3_BUCKET (default
"breastcancer"); credentials from the standard AWS environment variables.
"""
from __future__ import annotations

import argparse
import os
import sys

from logging_setup import setup_logging
from objectstore.sync import (
    DEFAULT_BUCKET,
    ENV_BUCKET,
    client_from_env,
    plan_sync,
    publishable_files,
    run_sync,
)


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m objectstore",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default=None, help="overrides BREASTCANCER_S3_ENDPOINT")
    parser.add_argument("--bucket", default=os.environ.get(ENV_BUCKET, DEFAULT_BUCKET))
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("plan", "compare local files with the bucket, send nothing"),
                            ("sync", "upload new and changed files, delete nothing")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dicom-sample", type=int, default=0,
                       help="also publish the raw DICOM of the first N series")
    p_ls = sub.add_parser("ls", help="list objects in the bucket")
    p_ls.add_argument("--prefix", default="")
    args = parser.parse_args(argv)

    setup_logging(logfile="objectstore.log")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    client = client_from_env(args.endpoint)

    if args.command == "ls":
        paginator = client.get_paginator("list_objects_v2")
        total = 0
        for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
            for obj in page.get("Contents", []):
                total += obj["Size"]
                print(f"{obj['Size']:>12}  {obj['Key']}")
        print(f"{_mb(total)} under s3://{args.bucket}/{args.prefix}")
        return 0

    files = publishable_files(dicom_sample=args.dicom_sample)
    if args.command == "plan":
        plan = plan_sync(client, args.bucket, files)  # never creates the bucket
    else:
        plan = run_sync(client, args.bucket, files)

    s = plan.summary()
    verb = "would upload" if args.command == "plan" else "uploaded"
    print(f"{len(files)} local file(s): {verb} {s['new']} new + {s['changed']} changed "
          f"({_mb(s['bytes_to_upload'])}), {s['unchanged']} unchanged, "
          f"{s['remote_only']} only in the bucket (kept) -> s3://{args.bucket}/")
    for f in plan.to_upload[:20]:
        print(f"  {'+' if f in plan.upload_new else '~'} {f.key}  ({_mb(f.size)})")
    if len(plan.to_upload) > 20:
        print(f"  ... {len(plan.to_upload) - 20} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
