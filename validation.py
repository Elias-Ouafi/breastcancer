"""What a preprocessed volume has to look like before it is written to disk.

Preprocessing turns DICOM series into ``.npz`` volumes that everything downstream
trusts blindly: the trainer reshapes them, the evaluator scores against their masks,
the app renders them. A malformed volume does not crash there -- it trains a model
on nonsense, or produces a metric that looks plausible and is not. The cost of
finding that out is a training run.

So the checks live at the single point where a volume is written
(``TransformData.save_preprocessed``) and fail loudly, naming the case.

The thresholds are calibrated on this dataset rather than guessed
--------------------------------------------------------------------
The failure worth catching is the one that already happened: ``crop=True`` recentred
each volume on the lesion ROI, producing tiny volumes on which the model scored a
confidence of 1.0000 on 9 test patients out of 9 (plan.md section 4.2). The obvious
signal -- "the lesion occupies too much of the volume" -- turns out **not** to work:
measured over the 28 test patients, the fraction of lesion-bearing slices runs from
3.3% to 50.9% (median 11.9%) on legitimate full-frame data, so a threshold there
would fire on real patients.

The dimensions do separate the two cleanly. Full-frame volumes in this collection are
448x448 or 512x512 in plane; the cropped volumes from the bug were of the order of
45x72x70. ``MIN_IN_PLANE`` sits at 128 -- far below anything legitimate, far above
anything cropped.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Below this in-plane size a volume cannot be a full-frame breast MRI; it is a crop.
# Real data sits at 448-512, so this leaves a factor of 3.5 of headroom.
MIN_IN_PLANE = 128


class VolumeValidationError(ValueError):
    """A preprocessed volume violates a contract the rest of the pipeline relies on."""


def validate_volume_and_mask(volume, mask, case_id="<unknown>", expect_full_frame=True):
    """Check one volume/mask pair. Raises on a broken contract, warns on a smell.

    Returns the list of warnings raised, so a caller can count them across a run.

    ``expect_full_frame=False`` turns off the crop check, for the DBT pipeline and the
    demo cases, where a cropped volume is the intended output rather than an accident.
    """
    volume = np.asarray(volume)
    mask = np.asarray(mask)
    where = f"{case_id}: "

    # --- Contracts. Breaking any of these makes the file unusable downstream. ---
    if volume.ndim != 3:
        raise VolumeValidationError(
            f"{where}volume must be 3D (depth, height, width), got shape {volume.shape}")

    if mask.shape != volume.shape:
        raise VolumeValidationError(
            f"{where}mask shape {mask.shape} does not match volume shape {volume.shape}; "
            "they are indexed together at training time")

    if any(dimension == 0 for dimension in volume.shape):
        raise VolumeValidationError(f"{where}volume has an empty axis: shape {volume.shape}")

    # float16 overflows to inf around 65504, which is exactly what an unnormalised
    # intensity does when cast down. Catching it here beats catching it as a NaN loss.
    finite = np.isfinite(volume)
    if not finite.all():
        bad = int((~finite).sum())
        raise VolumeValidationError(
            f"{where}volume holds {bad} non-finite value(s) out of {volume.size}. "
            "Intensity normalisation failed, or an unnormalised volume overflowed float16")

    unique = np.unique(mask)
    if not np.isin(unique, (0, 1)).all():
        raise VolumeValidationError(
            f"{where}mask must be binary, found values {unique[:8].tolist()}"
            f"{' ...' if unique.size > 8 else ''}")

    # --- Smells. Real, but not a reason to throw away the file. ---
    warnings = []

    if float(np.ptp(volume.astype(np.float32))) == 0.0:
        warnings.append(f"{where}volume is constant — nothing to learn from this series")

    in_plane = min(volume.shape[1], volume.shape[2])
    if expect_full_frame and in_plane < MIN_IN_PLANE:
        warnings.append(
            f"{where}in-plane size is {volume.shape[1]}x{volume.shape[2]}, below the "
            f"{MIN_IN_PLANE} px expected of a full-frame study. This is the signature of "
            "cropping to the lesion ROI, which makes localisation artificially easy "
            "(plan.md section 4.2). Pass crop=False, or expect_full_frame=False if the "
            "crop is intended")

    lesion_voxels = int((mask > 0).sum())
    if lesion_voxels == 0:
        warnings.append(f"{where}mask is empty — no lesion to localise in this volume")

    for message in warnings:
        log.warning(message)
    return warnings


def summarise(volume, mask):
    """Descriptive stats for the lineage manifest. Assumes validation already passed."""
    volume = np.asarray(volume)
    mask = np.asarray(mask) > 0
    positive_slices = int(mask.reshape(mask.shape[0], -1).any(axis=1).sum())
    return {
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "lesion_voxels": int(mask.sum()),
        "lesion_slices": positive_slices,
        "lesion_slice_fraction": round(positive_slices / volume.shape[0], 4),
    }
