"""The DCE-MRI pipeline as one runnable, resumable flow.

    python -m pipelines.dce_mri --max-patients 20
    python -m pipelines.dce_mri --from preprocess     # skip the 60 GB download
    python -m pipelines.dce_mri --dry-run             # print the plan, run nothing

Four stages, in the only order that works:

    download -> preprocess -> train -> evaluate
    (raw_data)  (preprocessed_data)   (models/)

Why an orchestrator rather than a shell script
----------------------------------------------
The stages are slow and unequal: the download is hours of network I/O that fails on
individual series, preprocessing is CPU-bound over ~200 patients, training is hours of
GPU. A shell script that dies in stage three has thrown away stages one and two, and
tells you nothing about which patient broke.

So each stage is a Prefect task, which buys three things the script cannot:

* **Retries where failure is transient.** Only the download retries -- a TCIA series
  fetch fails on timeouts. Re-running a *training* step that crashed would just burn
  another hour reaching the same exception, so it does not retry.
* **Idempotence.** Every stage checks its own output first and skips when it is already
  there. Re-running the flow after a crash resumes instead of redoing, and ``--force``
  is the explicit way to say "do it again anyway".
* **A run record.** Which stage ran, how long it took, what it emitted -- visible in
  the Prefect UI or the logs, rather than reconstructed from shell history.

The tasks are thin: they call the same functions the CLI entry points call, so the
pipeline and the manual commands can never drift apart.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from glob import glob

from prefect import flow, get_run_logger, task

import config
from logging_setup import setup_logging

log = logging.getLogger(__name__)

STAGES = ["download", "preprocess", "train", "evaluate"]


def _logger():
    """Prefect's run logger inside a flow, the module logger outside it.

    Lets the same task body be called from a test without a flow context.
    """
    try:
        return get_run_logger()
    except Exception:  # pragma: no cover - outside a flow run
        return log


def _count_npz(directory):
    return len(glob(os.path.join(directory, "*.npz")))


@task(name="download-dce-mri", retries=3, retry_delay_seconds=60)
def download(max_patients, max_gb, force=False):
    """Fetch Duke-Breast-Cancer-MRI series into the raw layer.

    Retries because this is the one stage whose failures are transient: TCIA drops
    connections mid-series. ``download_dce_mri_series`` already skips series present on
    disk, so a retry resumes rather than restarting.
    """
    logger = _logger()
    target = os.path.join(config.TCIA_DIR, "duke_mri")
    existing = len(os.listdir(target)) if os.path.isdir(target) else 0

    if existing and not force:
        logger.info("Raw layer already holds %d series in %s -- skipping download "
                    "(use --force to re-fetch).", existing, target)
        return target

    from ExtractData import download_dce_mri_series

    logger.info("Downloading up to %s patients (cap %s GB) into %s",
                max_patients, max_gb, target)
    download_dce_mri_series(max_patients=max_patients, download_dir=target, max_gb=max_gb)
    return target


@task(name="preprocess-dce-mri")
def preprocess(raw_dir, boxes_path, force=False):
    """Build subtraction volumes + box masks, one ``.npz`` per patient.

    No retry: a failure here is a bad DICOM or a missing annotation, and running it
    again produces the same failure.
    """
    logger = _logger()
    out_dir = config.DCE_MRI_PREPROCESSED_DIR
    existing = _count_npz(out_dir)

    if existing and not force:
        logger.info("%d preprocessed volumes already in %s -- skipping.", existing, out_dir)
        return out_dir

    if not os.path.exists(boxes_path):
        raise FileNotFoundError(
            f"Annotation boxes not found at {boxes_path}. Fetch Annotation_Boxes.xlsx "
            "from TCIA into the raw layer first."
        )

    from TransformData import preprocess_dce_mri_with_boxes

    logger.info("Preprocessing %s -> %s (post-contrast phase 2, full frame)", raw_dir, out_dir)
    # crop=False is deliberate: cropping to the lesion ROI made the task artificially
    # easy and produced the confidence-always-1.0 bug (plan.md section 4.2).
    preprocess_dce_mri_with_boxes(root_dir=raw_dir, boxes_path=boxes_path,
                                  output_dir=out_dir, post_phase_rank=2, crop=False)
    logger.info("Wrote %d volumes.", _count_npz(out_dir))
    return out_dir


@task(name="train-unet")
def train_unet(data_dir, epochs, force=False):
    """Train the segmentation U-Net, writing checkpoint + metrics to ``models/``."""
    logger = _logger()
    out_dir = config.DCE_MRI_MODEL_DIR
    checkpoint = config.DCE_MRI_UNET_CKPT

    if os.path.exists(checkpoint) and not force:
        logger.info("Checkpoint already at %s -- skipping training (use --force to retrain).",
                    checkpoint)
        return checkpoint

    from imaging.train import build_arg_parser
    from imaging.train import train as run_training

    args = build_arg_parser().parse_args([
        "--data-dir", data_dir,
        "--output-dir", out_dir,
        "--slice-bank", config.SLICE_BANK_DIR,
        "--epochs", str(epochs),
    ])
    logger.info("Training for %d epochs on %d volumes.", epochs, _count_npz(data_dir))
    run_training(args)
    return checkpoint


@task(name="evaluate-unet")
def evaluate_unet(data_dir, checkpoint):
    """Score the held-out split and write the report the README's numbers come from.

    Always runs: it is minutes, not hours, and it is the step whose output is quoted
    publicly -- silently reusing a stale report is exactly the failure worth avoiding.
    """
    logger = _logger()
    from imaging.evaluate import build_arg_parser, run

    args = build_arg_parser().parse_args([
        "--data-dir", data_dir,
        "--checkpoint", checkpoint,
        "--output-dir", config.DCE_MRI_MODEL_DIR,
    ])
    run(args)
    report = os.path.join(config.DCE_MRI_MODEL_DIR, "eval_report.json")
    logger.info("Evaluation report: %s", report)
    return report


@flow(name="dce-mri-pipeline", log_prints=True)
def dce_mri_pipeline(max_patients=200, max_gb=80, epochs=25, boxes_path=None,
                     start_at="download", force=False):
    """Run the DCE-MRI pipeline from ``start_at`` to the end.

    Returns the path of the evaluation report, which is the pipeline's real output:
    the checkpoint alone is a claim, the report is the evidence for it.
    """
    logger = _logger()
    if start_at not in STAGES:
        raise ValueError(f"start_at must be one of {STAGES}; got {start_at!r}")
    todo = STAGES[STAGES.index(start_at):]
    logger.info("Stages to run: %s", " -> ".join(todo))

    boxes_path = boxes_path or config.MRI_ANNOTATION_BOXES
    raw_dir = os.path.join(config.TCIA_DIR, "duke_mri")
    data_dir = config.DCE_MRI_PREPROCESSED_DIR
    checkpoint = config.DCE_MRI_UNET_CKPT

    if "download" in todo:
        raw_dir = download(max_patients, max_gb, force=force)
    if "preprocess" in todo:
        data_dir = preprocess(raw_dir, boxes_path, force=force)
    if "train" in todo:
        checkpoint = train_unet(data_dir, epochs, force=force)
    if "evaluate" in todo:
        return evaluate_unet(data_dir, checkpoint)
    return checkpoint


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-patients", type=int, default=200)
    p.add_argument("--max-gb", type=int, default=80,
                   help="Stop downloading once the raw layer reaches this size.")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--boxes", default=None,
                   help=f"Annotation table (default: {config.MRI_ANNOTATION_BOXES})")
    p.add_argument("--from", dest="start_at", default="download", choices=STAGES,
                   help="Start at this stage instead of the beginning.")
    p.add_argument("--force", action="store_true",
                   help="Re-run stages whose output already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the stages that would run, and what each would read and write.")
    return p


def _describe(args):
    """The plan, without running it -- the cheap way to check paths before an hours-long run."""
    todo = STAGES[STAGES.index(args.start_at):]
    boxes = args.boxes or config.MRI_ANNOTATION_BOXES
    raw = os.path.join(config.TCIA_DIR, "duke_mri")
    io = {
        "download": (f"TCIA ({args.max_patients} patients, cap {args.max_gb} GB)", raw),
        "preprocess": (f"{raw} + {boxes}", config.DCE_MRI_PREPROCESSED_DIR),
        "train": (config.DCE_MRI_PREPROCESSED_DIR, config.DCE_MRI_UNET_CKPT),
        "evaluate": (config.DCE_MRI_UNET_CKPT, os.path.join(config.DCE_MRI_MODEL_DIR,
                                                            "eval_report.json")),
    }
    lines = [f"Would run {len(todo)} stage(s): {' -> '.join(todo)}", ""]
    for stage in todo:
        src, dst = io[stage]
        lines.append(f"  {stage:<11} {src}")
        lines.append(f"  {'':<11}   -> {dst}")
    if not args.force:
        lines += ["", "  Stages whose output already exists will be skipped (--force overrides)."]
    return "\n".join(lines)


if __name__ == "__main__":
    parsed = build_arg_parser().parse_args()
    setup_logging(logfile="dce_mri_pipeline.log")
    if parsed.dry_run:
        print(_describe(parsed))
        sys.exit(0)
    dce_mri_pipeline(max_patients=parsed.max_patients, max_gb=parsed.max_gb,
                     epochs=parsed.epochs, boxes_path=parsed.boxes,
                     start_at=parsed.start_at, force=parsed.force)
