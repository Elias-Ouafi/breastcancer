"""Regenerate the three curated demo cases shipped in ``data/curated_data/demo_cases/``.

The cases in git are slim (a 25-slice slab, 4-5 MB each) so the demo works straight out
of a clone. This script rebuilds them from the full preprocessed DCE-MRI volumes,
which is what you need when the pinned slices change, when the preprocessing is
redone, or to verify that what is committed matches what the pipeline produces.

    python scripts/make_demo_cases.py                       # rebuild
    python scripts/make_demo_cases.py --verify              # rebuild + score them
    python scripts/make_demo_cases.py --source-dir <dir>    # other volume source

Requires the full volumes (``data/preprocessed_data/dce_mri_p2/``, produced by
``TransformData.preprocess_dce_mri_with_boxes`` with ``post_phase_rank=2`` and
``crop=False``). Those are patient data and stay out of git -- which is exactly why
the *outputs* are committed instead.

The (patient, slice) pairs below are not arbitrary: for every one of the 186
patients the model was scored on the slice where the ground-truth mask is largest,
and these are the three best by real IoU (plan.md §4.2). The slice is human-picked
because automatic slice selection does not work yet -- a documented limitation, not
a hidden shortcut.
"""
from __future__ import annotations

import argparse
import os
import sys

# Importable as `python scripts/make_demo_cases.py` from anywhere in the repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from TransformData import make_demo_case  # noqa: E402

# (output name, patient id, pinned slice, IoU verified on that slice -- plan.md §4.2)
DEMO_CASES = [
    ("demo_1_Breast_MRI_135.npz", "Breast_MRI_135", 52, 0.830),
    ("demo_2_Breast_MRI_105.npz", "Breast_MRI_105", 62, 0.738),
    ("demo_3_Breast_MRI_079.npz", "Breast_MRI_079", 104, 0.728),
]

DEFAULT_SOURCE_DIR = config.DCE_MRI_PREPROCESSED_DIR
DEFAULT_OUT_DIR = config.DEMO_CASES_DIR


def build(source_dir=DEFAULT_SOURCE_DIR, out_dir=DEFAULT_OUT_DIR, slim=True):
    """Write every case in :data:`DEMO_CASES`. Returns the list of output paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for out_name, patient, slice_index, iou in DEMO_CASES:
        source = os.path.join(source_dir, f"{patient}.npz")
        if not os.path.exists(source):
            raise FileNotFoundError(
                f"Missing source volume {source!r}. Rebuilding demo cases needs the "
                "full preprocessed volumes (not in git -- patient data); regenerate "
                "them with TransformData.preprocess_dce_mri_with_boxes."
            )
        out_path = os.path.join(out_dir, out_name)
        make_demo_case(source, out_path, slice_index, slim=slim)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"{out_path}  <- {patient} slice {slice_index} "
              f"(IoU {iou:.3f}, {size_mb:.2f} MB)")
        written.append(out_path)
    return written


def verify(paths):
    """Score each rebuilt case with the real backend and print the result.

    A rebuilt case must reproduce the numbers the demo is presented with; inference
    is deterministic, so any drift here means the source volumes or the checkpoint
    changed.
    """
    from inference import DEFAULT_MRI_UNET_CKPT, load_unet, predict_dce_mri

    model, device = load_unet(DEFAULT_MRI_UNET_CKPT)
    for path in paths:
        r = predict_dce_mri(path, model=model, device=device)
        print(f"{os.path.basename(path)}: detected={r['lesion_detected']} "
              f"confidence={r['confidence']:.4f} slice={r['best_slice']}/{r['n_slices']} "
              f"box={r['box_xywh']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR,
                        help=f"Full preprocessed volumes (default: {DEFAULT_SOURCE_DIR})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Where to write the cases (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--full", action="store_true",
                        help="Keep the whole volume (~30 MB/case) instead of one slice")
    parser.add_argument("--verify", action="store_true",
                        help="Score the rebuilt cases (needs torch + the checkpoint)")
    args = parser.parse_args(argv)

    written = build(args.source_dir, args.out_dir, slim=not args.full)
    if args.verify:
        print()
        verify(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
