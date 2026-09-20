"""The pure steps of the nnU-Net MRI corpus: boxes -> pseudo-masks, normalisation, spacing.

numpy only, so these run in CI where SimpleITK is not installed. The two tests the task
asks for by name are here: the box -> mask conversion, and the normalisation inside the mask
(mean ~ 0, std ~ 1 there, exactly 0 outside).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mri_nnunet import steps

RAS = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
LPS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _ellipsoid(cols=(11, 50), rows=(11, 50), z=range(10, 30), spacing=(1.0, 1.0, 1.0),
               origin=(0.0, 0.0, 0.0), direction=LPS):
    return steps.box_to_ellipsoid(cols, rows, list(z), origin, spacing, direction)


# --------------------------------------------------------------- box -> pseudo-mask

def test_the_ellipsoid_is_inscribed_in_the_box_and_weighs_pi_over_six_of_it():
    ell = _ellipsoid()
    shape = (48, 64, 64)
    pseudo = steps.ellipsoid_mask(shape, (0, 0, 0), (1, 1, 1), LPS, ell)
    box = steps.ellipsoid_mask(shape, (0, 0, 0), (1, 1, 1), LPS, ell, box=True)

    assert (pseudo <= box).all(), "a pseudo-mask voxel lies outside its own box"
    assert box.sum() == 40 * 40 * 20  # 1-based inclusive: cols 11..50, rows 11..50, 20 slices
    ratio = steps.box_volume_ratio(box.sum(), pseudo.sum())
    assert ratio == pytest.approx(1 / steps.INSCRIBED_ELLIPSOID_RATIO, rel=0.05)


def test_the_centre_is_inside_and_the_box_corners_are_not():
    pseudo = steps.ellipsoid_mask((48, 64, 64), (0, 0, 0), (1, 1, 1), LPS, _ellipsoid())
    assert pseudo[19, 30, 30] == 1          # z 10..29 -> centre 19.5; cols/rows 10..49 -> 29.5
    for z, y, x in [(10, 10, 10), (29, 49, 49), (10, 49, 10), (29, 10, 49)]:
        assert pseudo[z, y, x] == 0


def test_a_one_voxel_box_is_not_lost():
    ell = _ellipsoid(cols=(30, 30), rows=(30, 30), z=[20])
    pseudo = steps.ellipsoid_mask((48, 64, 64), (0, 0, 0), (1, 1, 1), LPS, ell)
    assert pseudo.sum() >= 1 and pseudo[20, 29, 29] == 1


def test_the_boxes_1_based_inclusive_convention_is_applied():
    """TCIA counts from 1 and includes the last row: cols 1..1 is voxel index 0."""
    ell = _ellipsoid(cols=(1, 1), rows=(1, 1), z=[0])
    box = steps.ellipsoid_mask((8, 8, 8), (0, 0, 0), (1, 1, 1), LPS, ell, box=True)
    assert box.sum() == 1 and box[0, 0, 0] == 1


def test_a_reversed_box_is_refused():
    with pytest.raises(ValueError, match="ends before it starts"):
        steps.box_to_ellipsoid((20, 10), (5, 9), [1, 2], (0, 0, 0), (1, 1, 1), LPS)


def test_the_pseudo_mask_lands_on_the_same_physical_place_on_a_flipped_grid():
    """RAS reorientation flips x and y: the mask must follow the anatomy, not the indices.

    The same physical grid, read once with the native LPS axes and once with the RAS ones
    (x and y reversed), gives masks that are exact mirror images of each other.
    """
    shape = (24, 32, 40)                                  # (z, y, x)
    spacing = (1.0, 1.0, 2.0)
    ell = steps.box_to_ellipsoid((8, 20), (10, 22), [8, 9, 10, 11, 12], (0, 0, 0), spacing, LPS)
    native = steps.ellipsoid_mask(shape, (0, 0, 0), spacing, LPS, ell)
    # The RAS grid holds the same voxel centres, walked from the other end of x and y.
    origin_ras = (39 * 1.0, 31 * 1.0, 0.0)
    flipped = steps.ellipsoid_mask(shape, origin_ras, spacing, RAS, ell)

    assert native.sum() > 0
    assert np.array_equal(flipped, native[:, ::-1, ::-1])


def test_the_pseudo_mask_volume_does_not_depend_on_the_resampling():
    """Anisotropic native grid vs 1 mm isotropic: the lesion keeps its volume in mm3."""
    ell = steps.box_to_ellipsoid((11, 40), (11, 40), list(range(10, 22)), (0, 0, 0), (1.0, 1.0, 2.0), LPS)
    native = steps.ellipsoid_mask((40, 64, 64), (0, 0, 0), (1.0, 1.0, 2.0), LPS, ell)
    iso = steps.ellipsoid_mask((80, 64, 64), (0, 0, 0), (1.0, 1.0, 1.0), LPS, ell)
    assert native.sum() * 2.0 == pytest.approx(iso.sum() * 1.0, rel=0.05)
    expected = 4 / 3 * math.pi * float(np.prod(ell["semi_axes"]))
    assert iso.sum() == pytest.approx(expected, rel=0.05)


def test_an_oblique_acquisition_still_gets_an_ellipsoid_of_the_right_volume():
    """A 2 degree tilt (Breast_MRI_001 has one): mask rasterised on an aligned grid, same volume."""
    tilt = math.radians(2.0)
    c, s = math.cos(tilt), math.sin(tilt)
    oblique = (c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c)
    ell = steps.box_to_ellipsoid((11, 40), (11, 40), list(range(10, 30)), (0, 0, 0), (1.0, 1.0, 1.0), oblique)
    pseudo = steps.ellipsoid_mask((64, 96, 96), (-10.0, 0.0, -10.0), (1.0, 1.0, 1.0), LPS, ell)
    expected = 4 / 3 * math.pi * float(np.prod(ell["semi_axes"]))
    assert pseudo.sum() == pytest.approx(expected, rel=0.05)


def test_a_box_off_the_grid_gives_an_empty_mask_not_an_error():
    ell = _ellipsoid()
    assert steps.ellipsoid_mask((4, 4, 4), (500.0, 500.0, 500.0), (1, 1, 1), LPS, ell).sum() == 0


def _contrast_case():
    shape = (20, 30, 30)
    organ = np.ones(shape, np.uint8)
    box = np.zeros(shape, np.uint8)
    box[4:16, 6:24, 6:24] = 1
    pseudo = steps.ellipsoid_mask(shape, (0, 0, 0), (1, 1, 1), LPS,
                                  steps.box_to_ellipsoid((7, 24), (7, 24), list(range(4, 16)),
                                                         (0, 0, 0), (1, 1, 1), LPS))
    noise = np.random.default_rng(0).normal(0, 1, shape).astype(np.float32)
    return noise, pseudo, box, organ


def test_a_box_on_an_enhancing_lesion_stands_out_and_a_box_on_tissue_does_not():
    noise, pseudo, box, organ = _contrast_case()
    on_lesion = noise + 5.0 * pseudo                     # the lesion enhances by 5 sd
    assert steps.enhancement_contrast(on_lesion, pseudo, box, organ) > 4.0
    assert abs(steps.enhancement_contrast(noise, pseudo, box, organ)) < 0.5


def test_the_contrast_does_not_read_an_empty_corner_as_a_loose_box():
    """The index this replaces read 0 on a perfect box: the corners of a box are empty by geometry."""
    noise, pseudo, box, organ = _contrast_case()
    perfect = noise + 5.0 * pseudo                       # bright exactly in the ellipsoid, dark corners
    assert steps.enhancement_contrast(perfect, pseudo, box, organ) > 4.0


def test_the_contrast_is_nan_without_a_pseudo_mask_or_without_a_reference():
    noise, pseudo, box, organ = _contrast_case()
    assert math.isnan(steps.enhancement_contrast(noise, np.zeros_like(pseudo), box, organ))
    assert math.isnan(steps.enhancement_contrast(noise, pseudo, box, np.zeros_like(organ)))


# ------------------------------------------------------------------- normalisation

def _case(seed=0, shape=(20, 40, 40)):
    rng = np.random.default_rng(seed)
    volume = rng.normal(500.0, 80.0, size=shape).astype(np.float32)
    mask = np.zeros(shape, np.uint8)
    mask[4:16, 8:32, 8:32] = 1
    volume[mask == 0] = rng.uniform(0, 3000, size=int((mask == 0).sum()))  # a background with its own scale
    return volume, mask


def test_the_zscore_is_zero_mean_unit_std_inside_the_mask_and_zero_outside():
    volume, mask = _case()
    out, (mean, std) = steps.zscore_in_mask(volume, mask)

    inside = out[mask > 0]
    assert inside.mean() == pytest.approx(0.0, abs=1e-5)
    assert inside.std() == pytest.approx(1.0, abs=1e-3)
    assert (out[mask == 0] == 0).all()
    assert mean == pytest.approx(volume[mask > 0].mean(), rel=1e-6)
    assert std == pytest.approx(volume[mask > 0].std(), rel=1e-6)


def test_the_normalisation_ignores_the_background_entirely():
    volume, mask = _case()
    louder = volume.copy()
    louder[mask == 0] *= 100.0
    a, _ = steps.zscore_in_mask(volume, mask)
    b, _ = steps.zscore_in_mask(louder, mask)
    assert np.array_equal(a, b), "the background leaked into the statistics"


def test_the_clip_uses_the_percentiles_of_the_mask_not_of_the_volume():
    volume, mask = _case()
    clipped, low, high = steps.clip_percentiles(volume, mask, 0.5, 99.5)
    assert low == pytest.approx(np.percentile(volume[mask > 0], 0.5))
    assert high == pytest.approx(np.percentile(volume[mask > 0], 99.5))
    assert low != pytest.approx(np.percentile(volume, 0.5))
    assert clipped.min() == pytest.approx(low) and clipped.max() == pytest.approx(high)


def test_the_documented_order_clip_then_zscore_keeps_an_outlier_from_setting_the_scale():
    volume, mask = _case()
    volume[8, 20, 20] = 1e6                       # one hot voxel inside the mask
    raw, _ = steps.zscore_in_mask(volume, mask)
    clipped, _, _ = steps.clip_percentiles(volume, mask, 0.5, 99.5)
    tamed, _ = steps.zscore_in_mask(clipped, mask)
    assert raw[mask > 0].max() > 50               # unclipped: the outlier is many sigmas out
    assert tamed[mask > 0].max() < 10             # clipped first: it no longer dominates


@pytest.mark.parametrize("function", [
    lambda v, m: steps.zscore_in_mask(v, m),
    lambda v, m: steps.clip_percentiles(v, m, 0.5, 99.5),
    lambda v, m: steps.bbox_slices(m),
])
def test_an_empty_mask_is_an_error_not_a_nan_volume(function):
    volume, _ = _case()
    with pytest.raises(ValueError, match="empty"):
        function(volume, np.zeros(volume.shape, np.uint8))


def test_the_crop_is_the_mask_bounding_box_plus_a_margin_clamped_to_the_volume():
    mask = np.zeros((20, 40, 40), np.uint8)
    mask[4:16, 8:32, 8:32] = 1
    z, y, x = steps.bbox_slices(mask, margin=0)
    assert (z, y, x) == (slice(4, 16), slice(8, 32), slice(8, 32))
    z, y, x = steps.bbox_slices(mask, margin=8)
    assert (z, y, x) == (slice(0, 20), slice(0, 40), slice(0, 40))     # clamped, never negative


# ------------------------------------------------------------------------- spacing

def test_a_thick_slice_axis_keeps_its_native_spacing():
    assert steps.choose_spacing((0.8, 0.8, 1.1), 1.0, anisotropy_ratio=2.0) == (1.0, 1.0, 1.0)
    assert steps.choose_spacing((0.7, 0.7, 5.0), 1.0, anisotropy_ratio=2.0) == (1.0, 1.0, 5.0)


def test_the_spacing_follows_the_small_lesions_not_the_median():
    smallest = np.array([8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0])   # mm
    spacing, coverage = steps.auto_spacing(smallest, min_voxels=8, percentile=10, floor_mm=0.5, ceil_mm=1.5)
    median_choice = float(np.median(smallest)) / 8
    assert spacing < median_choice                         # the median would starve the small ones
    assert spacing == pytest.approx(np.percentile(smallest, 10) / 8)
    assert coverage >= 0.9                                 # >= 8 voxels on the smallest axis for 90% of them


def test_the_spacing_is_held_inside_its_bounds():
    tiny = np.array([1.0, 1.5, 2.0])
    huge = np.array([200.0, 300.0])
    assert steps.auto_spacing(tiny, 8, 10, 0.5, 1.5)[0] == 0.5
    assert steps.auto_spacing(huge, 8, 10, 0.5, 1.5)[0] == 1.5


def test_the_smallest_axis_is_measured_in_millimetres():
    extents = np.array([[11.0, 13.0, 8.0], [40.0, 30.0, 20.0]])
    assert steps.smallest_axis_mm(extents, (1.0, 1.0, 1.1)).tolist() == pytest.approx([8.8, 20.0 * 1.1])
