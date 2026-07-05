"""Dataset and patient-level splitting for MRI lesion segmentation.

Reads the compressed ``.npz`` files produced by
``TransformData.save_preprocessed`` (keys: ``volume``, ``mask``, ``crop_offset``)
and serves 2D axial slices as ``(image, mask)`` tensor pairs for a 2D U-Net.

Splitting is done at the *case* level (all slices of one case stay in the same
split) to avoid optimistic leakage between train/val/test.
"""
from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def default_patient_key(path):
    """Group key that keeps all slices/series of one case in the same split.

    Preprocessing stores the real patient identifier as ``case_id`` inside the
    ``.npz`` (several series/views can belong to one patient); it is used here so a
    patient never straddles the train/val/test boundary. Files without ``case_id``
    (older MRI outputs) fall back to the filename stem.
    """
    try:
        with np.load(path) as data:
            if "case_id" in data.files:
                case_id = str(data["case_id"])
                if case_id:
                    return case_id
    except Exception:
        pass
    return os.path.splitext(os.path.basename(path))[0]


def split_npz_by_patient(data_dir, val_frac=0.15, test_frac=0.15, seed=42,
                         patient_key=default_patient_key):
    """Split ``data_dir/*.npz`` into (train, val, test) lists of file paths.

    Whole cases (groups) are assigned to a single split. With very few cases the
    val/test fractions are honoured only as far as leaving at least one training
    case; a warning is emitted when a split ends up empty.
    """
    paths = sorted(glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No .npz files found in {data_dir!r}. "
            "Run TransformData.process_all_mri_data first to generate them."
        )

    groups = OrderedDict()
    for p in paths:
        groups.setdefault(patient_key(p), []).append(p)

    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    # Always keep at least one training case.
    n_val = min(n_val, max(0, n - n_test - 1))
    n_test = min(n_test, max(0, n - n_val - 1))

    test_keys = keys[:n_test]
    val_keys = keys[n_test:n_test + n_val]
    train_keys = keys[n_test + n_val:]

    def collect(ks):
        return [p for k in ks for p in groups[k]]

    train, val, test = collect(train_keys), collect(val_keys), collect(test_keys)
    if not val:
        warnings.warn("Validation split is empty (too few cases); metrics will be NaN.")
    if not test:
        warnings.warn("Test split is empty (too few cases).")
    return train, val, test


class MRISliceDataset(Dataset):
    """Serves 2D axial slices from a list of preprocessed ``.npz`` volumes.

    Parameters
    ----------
    npz_paths : list[str]
        Preprocessed volume files to draw slices from.
    image_size : int
        Slices are resized to ``(image_size, image_size)`` (bilinear for the image,
        nearest for the mask). Must be divisible by 16 for the U-Net.
    positive_only : bool
        If True, only slices whose mask contains lesion voxels are indexed. Useful
        for training on the (cropped) region of interest; set False for validation
        and test so background slices are also evaluated.
    cache_size : int
        Number of decoded volumes kept in an in-memory LRU cache to avoid
        re-reading a file for every slice.
    """

    def __init__(self, npz_paths, image_size=256, positive_only=True, neg_per_pos=None,
                 cache_size=8, seed=0):
        self.paths = list(npz_paths)
        self.image_size = image_size
        self.positive_only = positive_only
        self.neg_per_pos = neg_per_pos
        self._rng = np.random.default_rng(seed)
        self._cache_size = cache_size
        self._cache = OrderedDict()
        self.index = self._build_index()

    def _build_index(self):
        positives, negatives = [], []
        for p in self.paths:
            with np.load(p) as data:
                mask = data["mask"]
                depth = mask.shape[0]
                has_lesion = mask.reshape(depth, -1).any(axis=1)
            for z in range(depth):
                (positives if has_lesion[z] else negatives).append((p, z))

        # Balanced sampling: all lesion slices + a bounded number of background
        # slices per lesion slice. This prevents the model from collapsing to the
        # trivial "predict nothing" (or "predict everything") solution that a raw
        # positive-only or all-slices index encourages.
        if self.neg_per_pos is not None:
            n_neg = min(len(negatives), int(self.neg_per_pos * len(positives)))
            if n_neg and negatives:
                picked = self._rng.choice(len(negatives), size=n_neg, replace=False)
                negatives = [negatives[i] for i in picked]
            else:
                negatives = []
            index = positives + negatives
            self._rng.shuffle(index)
            return index

        if self.positive_only:
            return positives
        return positives + negatives

    def _load(self, path):
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        with np.load(path) as data:
            volume = data["volume"].astype(np.float32)
            mask = (data["mask"] > 0).astype(np.float32)
        self._cache[path] = (volume, mask)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return volume, mask

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        path, z = self.index[i]
        volume, mask = self._load(path)

        img = torch.from_numpy(volume[z])[None, None]  # (1, 1, H, W)
        msk = torch.from_numpy(mask[z])[None, None]
        size = (self.image_size, self.image_size)
        img = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        msk = F.interpolate(msk, size=size, mode="nearest")
        return img[0], msk[0]  # each (1, H, W)
