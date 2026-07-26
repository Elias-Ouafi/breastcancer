"""Flat, memory-mappable slice bank for fast 2D training.

Why this exists
---------------
``MRISliceDataset`` reads slices straight out of the compressed ``.npz`` volumes.
That is fine for the small cropped DBT volumes it was written for, but it collapses
on the full-frame DCE-MRI set: one ``.npz`` costs ~0.2 s to inflate, a training
epoch draws ~10k slices in shuffled order across ~130 volumes, and the in-memory LRU
cache only holds a handful of them. Nearly every access therefore re-inflates a whole
volume to read a single slice -- measured at ~17 min/epoch on an RTX 5060 whose GPU
sat at 5-28% utilisation, i.e. the model was starved by I/O, not compute.

The bank fixes that by paying the decompression cost **once**: every slice is
inflated, resized to the training resolution, and written into a flat ``.npy``
memmap. Training then reads a slice with a single page-cached disk seek and no
decompression, so random shuffling costs the same as sequential access.

Storing at the training resolution (256x256 by default) rather than native
(448x448 / 512x512, which vary across patients) is also what shrinks the bank enough
to sit in the OS page cache: ~4 GB for 30k slices instead of ~15 GB. The resize is
the same bilinear/nearest pair ``MRISliceDataset`` applied per access anyway, so the
tensors the model sees are unchanged -- it is just computed once instead of once per
epoch.

Layout written to ``out_dir``:
  volumes.npy  (N, S, S) float16  -- z-normalised image slices
  masks.npy    (N, S, S) uint8    -- binary lesion masks
  index.npz                       -- case_id / has_lesion / source path per slice
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:  # allow both "python -m imaging.train" and direct script execution
    from .dataset import _gamma_jitter, _random_affine, default_patient_key
except ImportError:  # pragma: no cover
    from dataset import _gamma_jitter, _random_affine, default_patient_key

VOLUMES_FILE = "volumes.npy"
MASKS_FILE = "masks.npy"
INDEX_FILE = "index.npz"


def _resize_pair(volume, mask, size):
    """Resize a whole ``(D, H, W)`` volume/mask pair to ``(D, size, size)``.

    Bilinear for intensities, nearest for the mask so it stays binary -- the same
    convention ``MRISliceDataset.__getitem__`` uses per slice.
    """
    img = torch.from_numpy(volume.astype(np.float32))[:, None]  # (D,1,H,W)
    msk = torch.from_numpy(mask.astype(np.float32))[:, None]
    img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    msk = F.interpolate(msk, size=(size, size), mode="nearest")
    return img[:, 0].numpy().astype(np.float16), (msk[:, 0].numpy() > 0.5).astype(np.uint8)


def build_slice_bank(npz_paths, out_dir, image_size=256, force=False):
    """Materialise ``npz_paths`` into a flat memmapped slice bank under ``out_dir``.

    Returns ``out_dir``. Re-uses an existing bank unless ``force`` is set or the
    stored geometry/source list no longer matches what was asked for, so repeated
    training runs pay the conversion cost only once.
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

    # Pass 1 -- read only the mask of each volume to size the bank and record, per
    # slice, its patient (for leakage-free splitting) and whether it holds lesion.
    per_file_depth, case_ids, has_lesion = [], [], []
    for p in paths:
        with np.load(p) as data:
            mask = data["mask"]
        depth = mask.shape[0]
        per_file_depth.append(depth)
        case_ids.extend([default_patient_key(p)] * depth)
        has_lesion.extend((mask.reshape(depth, -1) > 0).any(axis=1).tolist())

    total = int(sum(per_file_depth))
    volumes = np.lib.format.open_memmap(
        os.path.join(out_dir, VOLUMES_FILE), mode="w+", dtype=np.float16,
        shape=(total, image_size, image_size))
    masks = np.lib.format.open_memmap(
        os.path.join(out_dir, MASKS_FILE), mode="w+", dtype=np.uint8,
        shape=(total, image_size, image_size))

    # Pass 2 -- inflate each volume exactly once and write its resized slices.
    offset = 0
    for i, (p, depth) in enumerate(zip(paths, per_file_depth), start=1):
        with np.load(p) as data:
            vol = data["volume"]
            msk = data["mask"]
        vol_r, msk_r = _resize_pair(vol, msk, image_size)
        volumes[offset:offset + depth] = vol_r
        masks[offset:offset + depth] = msk_r
        offset += depth
        if i % 20 == 0 or i == len(paths):
            print(f"  slice bank: {i}/{len(paths)} volumes -> {offset}/{total} slices")

    volumes.flush()
    masks.flush()
    np.savez(index_path,
             case_id=np.array(case_ids),
             has_lesion=np.array(has_lesion, dtype=bool),
             source_paths=np.array(paths),
             image_size=np.array(image_size))
    return out_dir


class SliceBankDataset(Dataset):
    """Serves 2D slices from a :func:`build_slice_bank` bank.

    Same contract as ``MRISliceDataset`` (returns ``(image, mask)`` float tensors of
    shape ``(1, S, S)``, optional augmentation) but backed by a memmap, so a shuffled
    epoch does no decompression.

    ``case_ids`` restricts the dataset to one split's patients; the balanced
    ``positive_only``/``neg_per_pos`` sampling mirrors ``MRISliceDataset`` so metrics
    stay comparable across the two backends.
    """

    def __init__(self, bank_dir, case_ids=None, positive_only=True, neg_per_pos=None,
                 seed=0, augment=False):
        self.bank_dir = bank_dir
        with np.load(os.path.join(bank_dir, INDEX_FILE), allow_pickle=True) as idx:
            self.all_case_ids = idx["case_id"]
            self.all_has_lesion = idx["has_lesion"]
            self.image_size = int(idx["image_size"])
        # Memmaps are opened lazily, per process. DataLoader workers on Windows are
        # spawned (not forked), so the dataset is pickled to each one -- and an open
        # np.memmap does not survive that (UnpicklingError: pickle data was
        # truncated). Keeping them None here means each worker opens its own handle
        # on first access, which is also the cheaper arrangement on Linux.
        self._volumes = None
        self._masks = None

        self.augment = augment
        self._rng = np.random.default_rng(seed)

        if case_ids is None:
            candidates = np.arange(len(self.all_case_ids))
        else:
            wanted = set(case_ids)
            candidates = np.array([i for i, c in enumerate(self.all_case_ids) if c in wanted],
                                  dtype=np.int64)

        positives = candidates[self.all_has_lesion[candidates]]
        negatives = candidates[~self.all_has_lesion[candidates]]

        if neg_per_pos is not None:
            n_neg = min(len(negatives), int(neg_per_pos * len(positives)))
            if n_neg and len(negatives):
                picked = self._rng.choice(len(negatives), size=n_neg, replace=False)
                negatives = negatives[picked]
            else:
                negatives = negatives[:0]
            index = np.concatenate([positives, negatives])
            self._rng.shuffle(index)
        elif positive_only:
            index = positives
        else:
            index = np.concatenate([positives, negatives])
        self.index = index

    @property
    def volumes(self):
        if self._volumes is None:
            self._volumes = np.load(os.path.join(self.bank_dir, VOLUMES_FILE), mmap_mode="r")
        return self._volumes

    @property
    def masks(self):
        if self._masks is None:
            self._masks = np.load(os.path.join(self.bank_dir, MASKS_FILE), mmap_mode="r")
        return self._masks

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        img = torch.from_numpy(np.asarray(self.volumes[j], dtype=np.float32))[None]
        msk = torch.from_numpy(np.asarray(self.masks[j], dtype=np.float32))[None]

        if self.augment:
            if self._rng.random() < 0.5:
                img, msk = torch.flip(img, dims=[2]), torch.flip(msk, dims=[2])
            if self._rng.random() < 0.5:
                img, msk = torch.flip(img, dims=[1]), torch.flip(msk, dims=[1])
            img, msk = _random_affine(img, msk, self._rng)
            img = _gamma_jitter(img, self._rng)

        return img, msk


def case_ids_for_paths(npz_paths):
    """Patient keys for a list of ``.npz`` paths, matching the bank's ``case_id``."""
    return [default_patient_key(p) for p in npz_paths]
