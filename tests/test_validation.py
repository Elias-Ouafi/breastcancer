"""The validator has to catch the failures it claims to, and stay quiet otherwise.

A check that never fires is indistinguishable from no check at all, and one that fires
on good data gets switched off within a week. So each test here either builds the
specific corruption the check exists for, or builds a legitimate volume and asserts
silence.
"""
from __future__ import annotations

import numpy as np
import pytest

from validation import MIN_IN_PLANE, VolumeValidationError, summarise, validate_volume_and_mask


def _volume(depth=20, height=448, width=448, seed=0):
    """A plausible full-frame study: z-normalised noise at the real in-plane size."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((depth, height, width)).astype(np.float16)


def _mask(shape, slices=(4, 8), box=(100, 140)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[slices[0]:slices[1], box[0]:box[1], box[0]:box[1]] = 1
    return mask


def test_a_legitimate_volume_passes_silently():
    volume = _volume()
    assert validate_volume_and_mask(volume, _mask(volume.shape), "case") == []


# --- Contracts: these make the file unusable, so they must raise ---------------

def test_a_2d_volume_is_rejected():
    with pytest.raises(VolumeValidationError, match="must be 3D"):
        validate_volume_and_mask(np.zeros((448, 448), np.float16), np.zeros((448, 448), np.uint8))


def test_mask_and_volume_shapes_must_agree():
    """They are indexed together at training time; a mismatch mislabels every slice."""
    volume = _volume(depth=20)
    with pytest.raises(VolumeValidationError, match="does not match volume shape"):
        validate_volume_and_mask(volume, np.zeros((19, 448, 448), np.uint8), "case")


def test_non_finite_intensities_are_rejected():
    """float16 overflows to inf near 65504 -- what an unnormalised volume does."""
    volume = _volume()
    volume[3, 10, 10] = np.inf
    with pytest.raises(VolumeValidationError, match="non-finite"):
        validate_volume_and_mask(volume, _mask(volume.shape), "case")


def test_a_non_binary_mask_is_rejected():
    volume = _volume()
    mask = _mask(volume.shape)
    mask[5, 105, 105] = 7
    with pytest.raises(VolumeValidationError, match="binary"):
        validate_volume_and_mask(volume, mask, "case")


def test_an_empty_axis_is_rejected():
    with pytest.raises(VolumeValidationError, match="empty axis"):
        validate_volume_and_mask(np.zeros((0, 448, 448), np.float16),
                                 np.zeros((0, 448, 448), np.uint8))


def test_the_error_names_the_case():
    """Across 186 patients, an error that does not say which one is barely usable."""
    with pytest.raises(VolumeValidationError, match="Breast_MRI_042"):
        validate_volume_and_mask(np.zeros((4, 4), np.float16), np.zeros((4, 4), np.uint8),
                                 case_id="Breast_MRI_042")


# --- Smells: real, but not a reason to throw the file away --------------------

def test_a_lesion_roi_crop_is_flagged():
    """The plan.md section 4.2 failure: cropping made localisation artificially easy."""
    volume = _volume(depth=45, height=72, width=70)
    warnings = validate_volume_and_mask(volume, _mask(volume.shape, box=(10, 30)), "case")
    assert any("in-plane" in w for w in warnings), warnings


def test_an_intended_crop_is_not_flagged():
    """The DBT pipeline and the demo cases crop on purpose; nagging them is noise."""
    volume = _volume(depth=45, height=72, width=70)
    warnings = validate_volume_and_mask(volume, _mask(volume.shape, box=(10, 30)), "case",
                                        expect_full_frame=False)
    assert not any("in-plane" in w for w in warnings), warnings


def test_the_crop_threshold_clears_real_data_by_a_wide_margin():
    """Real studies are 448-512 in plane. The threshold must not creep up to meet them."""
    assert MIN_IN_PLANE < 448 / 3


def test_an_empty_mask_is_flagged_but_allowed():
    volume = _volume()
    warnings = validate_volume_and_mask(volume, np.zeros(volume.shape, np.uint8), "case")
    assert any("empty" in w for w in warnings), warnings


def test_a_constant_volume_is_flagged():
    volume = np.full((20, 448, 448), 0.5, dtype=np.float16)
    warnings = validate_volume_and_mask(volume, _mask(volume.shape), "case")
    assert any("constant" in w for w in warnings), warnings


# --- Summary feeds the lineage manifest ---------------------------------------

def test_summarise_counts_lesion_slices_not_just_voxels():
    volume = _volume(depth=20)
    stats = summarise(volume, _mask(volume.shape, slices=(4, 8), box=(100, 110)))
    assert stats["shape"] == [20, 448, 448]
    assert stats["lesion_slices"] == 4
    assert stats["lesion_voxels"] == 4 * 10 * 10
    assert stats["lesion_slice_fraction"] == 0.2
