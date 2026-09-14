"""Flat, memory-mappable slice bank for the exam-level decision head.

Why this exists
----------------
``TransformData.preprocess_dbt_exams`` writes every listed DBT series at one fixed
geometry (384x384 in plane, every slice kept, no cropping) -- see plan.md, "Corpus
DBT à deux classes". That corpus is what an exam-level cancer / no-cancer decision
needs, but it cannot be loaded the way a lesion-cropped corpus can, entirely
decompressed into memory: a lesion crop holds ~5 slices, while a full exam holds
24-114 slices (mean 68), and 870 of
them at 384x384 in float16 come to ~17.6 GB -- re-inflating that every epoch would
starve training on I/O the same way it did for DCE-MRI (see ``imaging.slicebank``).

The bank pays the decompression cost once, and downsamples further to `image_size`
(224 by default). 224 = 32 * 7, so five stride-2 pooling stages (the architecture
this package already uses, see ``sliceclf.SliceClassifier``) divide it evenly with
no rounding. The resize is area-averaged rather than sampled, the same reasoning
``TransformData.resize_in_plane`` documents for the larger first step (2457 -> 384):
picking one row in a downsampling stride is how a small bright mass is lost, while
averaging keeps its energy.

Layout written to ``out_dir``:
  volumes.npy  (N, S, S) float16 -- z-normalised image slices
  index.npz    -- per slice: case_id (patient), exam_id (source file -- the unit a
                  MIL bag is built over, since one .npz is one view), label (that
                  exam's 0/1 cancer target, broadcast to every one of its slices),
                  has_lesion (whether an annotated box paints this slice)
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

try:  # allow both "python -m imaging.examclf" and direct execution
    from .dataset import default_patient_key
except ImportError:  # pragma: no cover
    from dataset import default_patient_key

VOLUMES_FILE = "volumes.npy"
INDEX_FILE = "index.npz"


def exam_id_for_path(path):
    """The bag identifier for one ``.npz``: one file is one exam (a single view)."""
    return os.path.splitext(os.path.basename(str(path)))[0]


def _resize_stack(volume, size):
    """``(D, H, W)`` -> ``(D, size, size)``, area-averaged (downsampling only)."""
    img = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[:, None]
    resized = F.interpolate(img, size=(size, size), mode="area")
    return resized[:, 0].numpy().astype(np.float16)


def build_exam_bank(npz_paths, out_dir, image_size=224, force=False):
    """Materialise ``npz_paths`` into a flat memmapped exam bank under ``out_dir``.

    Every path must carry a ``label`` (0/1): that is the exam-level target
    ``preprocess_dbt_exams`` writes, and there is nothing to train this head on
    without it. Re-uses an existing bank unless ``force`` is set or the stored
    geometry/source list no longer matches what was asked for.

    Returns ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, INDEX_FILE)
    paths = [str(p) for p in npz_paths]

    if not force and os.path.exists(index_path):
        with np.load(index_path, allow_pickle=True) as idx:
            same_geometry = int(idx["image_size"]) == image_size
            same_sources = list(idx["source_paths"]) == paths
        if same_geometry and same_sources:
            return out_dir  # already built for exactly this request

    # Pass 1 -- read only the mask (for depth and has_lesion) and the label, cheap
    # relative to the volume, to size the bank before allocating it.
    per_file_depth, case_ids, exam_ids, labels, has_lesion = [], [], [], [], []
    for p in paths:
        with np.load(p) as data:
            if "label" not in data.files:
                raise KeyError(
                    f"{os.path.basename(p)} carries no `label`: the exam-level "
                    "decision head needs the cancer/no-cancer target that "
                    "TransformData.preprocess_dbt_exams stores. Re-run it.")
            mask = data["mask"]
            label = int(data["label"])
        depth = mask.shape[0]
        exam = exam_id_for_path(p)
        per_file_depth.append(depth)
        case_ids.extend([default_patient_key(p)] * depth)
        exam_ids.extend([exam] * depth)
        labels.extend([label] * depth)
        has_lesion.extend((mask.reshape(depth, -1) > 0).any(axis=1).tolist())

    total = int(sum(per_file_depth))
    volumes = np.lib.format.open_memmap(
        os.path.join(out_dir, VOLUMES_FILE), mode="w+", dtype=np.float16,
        shape=(total, image_size, image_size))

    # Pass 2 -- inflate each volume exactly once and write its resized slices.
    offset = 0
    for i, (p, depth) in enumerate(zip(paths, per_file_depth), start=1):
        with np.load(p) as data:
            vol = data["volume"]
        volumes[offset:offset + depth] = _resize_stack(vol, image_size)
        offset += depth
        if i % 50 == 0 or i == len(paths):
            log.info(f"  exam bank: {i}/{len(paths)} exams -> {offset}/{total} slices")

    volumes.flush()
    np.savez(index_path,
             case_id=np.array(case_ids),
             exam_id=np.array(exam_ids),
             label=np.array(labels, dtype=np.uint8),
             has_lesion=np.array(has_lesion, dtype=bool),
             source_paths=np.array(paths),
             image_size=np.array(image_size))
    return out_dir


class ExamBank:
    """Read-only view of a bank :func:`build_exam_bank` wrote.

    Groups slice rows by ``exam_id`` once at construction (``O(N)``, done a single
    time rather than per lookup), so ``rows_for`` and the per-exam ``exam_label`` /
    ``exam_case`` maps used by training and evaluation are O(1) afterwards.
    """

    def __init__(self, bank_dir):
        self.bank_dir = bank_dir
        with np.load(os.path.join(bank_dir, INDEX_FILE), allow_pickle=True) as idx:
            self.case_id = idx["case_id"]
            self.exam_id = idx["exam_id"]
            self.label = idx["label"]
            self.has_lesion = idx["has_lesion"]
            self.image_size = int(idx["image_size"])
        # Memmaps are opened lazily, per process: a DataLoader worker on Windows is
        # spawned rather than forked, and an already-open np.memmap does not survive
        # being pickled to it (the same reason imaging.slicebank defers this).
        self._volumes = None

        self._exam_rows = {}
        for i, exam in enumerate(self.exam_id):
            self._exam_rows.setdefault(exam, []).append(i)
        self._exam_rows = {e: np.array(rows, dtype=np.int64)
                           for e, rows in self._exam_rows.items()}
        self.exam_label = {e: int(self.label[rows[0]])
                           for e, rows in self._exam_rows.items()}
        self.exam_case = {e: str(self.case_id[rows[0]])
                          for e, rows in self._exam_rows.items()}

    @property
    def volumes(self):
        if self._volumes is None:
            self._volumes = np.load(os.path.join(self.bank_dir, VOLUMES_FILE),
                                    mmap_mode="r")
        return self._volumes

    def unique_exams(self):
        return list(self._exam_rows)

    def rows_for(self, exam_id):
        return self._exam_rows[exam_id]
