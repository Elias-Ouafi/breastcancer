"""Command line of the nnU-Net MRI corpus.

    python -m mri_nnunet build                    # ingest + process every patient (hours)
    python -m mri_nnunet build --patients Breast_MRI_037 --limit 3   # try it on a few first
    python -m mri_nnunet spacing                  # box distribution and the spacing it implies
    python -m mri_nnunet export                   # the nnUNet_raw dataset, in gold
    python -m mri_nnunet qc --n 10                # histograms, overlays, aggregate statistics

Install with ``pip install -e ".[data,nnunet]"`` (or ``.[all]``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import config
from logging_setup import setup_logging

from . import settings as settings_module


def build_arg_parser():
    p = argparse.ArgumentParser(prog="mri_nnunet", description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None, help="YAML parameters (default: mri_nnunet/config.yaml)")
    p.add_argument("--silver-dir", default=config.DCE_MRI_NNUNET_SILVER_DIR)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="ingest then process")
    b.add_argument("--raw-dir", default=os.path.join(config.TCIA_DIR, "duke_mri"))
    b.add_argument("--boxes", default=config.MRI_ANNOTATION_BOXES)
    b.add_argument("--patients", nargs="*", default=None)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--force", action="store_true", help="redo cases whose parameters did not change")
    b.add_argument("--ingest-only", action="store_true",
                   help="only the lossless DICOM -> native NIfTI copy (all that needs bronze)")

    sub.add_parser("spacing", help="box-size distribution and the chosen spacing")

    e = sub.add_parser("export", help="write the nnUNet_raw dataset")
    e.add_argument("--nnunet-raw", default=config.NNUNET_RAW_DIR)

    q = sub.add_parser("qc", help="QC report on random cases")
    q.add_argument("--n", type=int, default=10)
    q.add_argument("--out", default=None, help="default: <silver-dir>/qc")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    setup_logging(logfile="mri_nnunet.log")
    settings = settings_module.load(args.config)

    if args.command == "build":
        from . import pipeline

        counts = pipeline.build(args.raw_dir, args.boxes, args.silver_dir, settings,
                                patients=args.patients, limit=args.limit, force=args.force,
                                ingest_only=args.ingest_only)
        print(json.dumps(counts, indent=2))
    elif args.command == "spacing":
        from . import spacing

        print(json.dumps(spacing.describe(args.silver_dir, settings), indent=2))
    elif args.command == "export":
        from . import export

        folder, cases = export.export(args.silver_dir, args.nnunet_raw, settings)
        print(f"{len(cases)} cases -> {folder}")
    elif args.command == "qc":
        from . import qc

        out = args.out or os.path.join(args.silver_dir, "qc")
        summary = qc.run(args.silver_dir, out, settings, n=args.n)
        print(json.dumps({k: v for k, v in summary.items() if k != "scanners"}, indent=2, default=str))
        print(f"report in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
