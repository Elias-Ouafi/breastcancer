"""The pure, numpy-only steps of the nnU-Net MRI corpus.

Everything here is a function of arrays and numbers -- no file, no SimpleITK, no clock --
so each step can be tested in isolation, and so the tests that matter most (the box to
pseudo-mask conversion, the normalisation inside the mask) run in CI, where SimpleITK is
not installed. The geometric work that needs an image library (DICOM with its geometry,
N4, registration, resampling) lives in ``sitk_io.py`` and only calls these.

Arrays are ``(z, y, x)``, as SimpleITK's ``GetArrayFromImage`` returns them. Geometry is
passed as plain tuples in SimpleITK's convention: ``origin`` and ``spacing`` in ``(x, y, z)``
order, ``direction`` a row-major flattened 3x3 matrix whose *columns* are the physical
directions of the ``x``, ``y`` and ``z`` index axes.
"""
from __future__ import annotations

import math

import numpy as np

# The volume of an ellipsoid inscribed in its bounding box, relative to that box: pi / 6.
# It is what a pseudo-mask built from a box must weigh, so any other ratio is information.
INSCRIBED_ELLIPSOID_RATIO = math.pi / 6.0


# --------------------------------------------------------------------------- geometry

def direction_matrix(direction):
    """The 3x3 matrix of a SimpleITK direction tuple; columns are the index axes."""
    return np.asarray(direction, dtype=float).reshape(3, 3)


def index_to_physical(index_xyz, origin, spacing, direction):
    """Physical point(s) of continuous index(es) ``(x, y, z)``; shape ``(..., 3)``."""
    index_xyz = np.asarray(index_xyz, dtype=float)
    return np.asarray(origin, float) + (index_xyz * np.asarray(spacing, float)) @ direction_matrix(direction).T


def physical_to_index(points, origin, spacing, direction):
    """Continuous ``(x, y, z)`` index(es) of physical point(s); the inverse of the above."""
    delta = np.asarray(points, dtype=float) - np.asarray(origin, float)
    return (delta @ np.linalg.inv(direction_matrix(direction)).T) / np.asarray(spacing, float)


def box_to_ellipsoid(cols, rows, z_indices, origin, spacing, direction):
    """The ellipsoid inscribed in one annotation box, as physical-space parameters.

    ``cols`` and ``rows`` are the box's ``(start, end)`` in **1-based inclusive** voxels, the
    convention TCIA publishes (``TransformData._read_mri_boxes``). ``z_indices`` are the
    0-based indices, in the image's own slice order, of the slices the box covers -- what
    the caller obtained by mapping the published slice numbers through the series (the
    published order is by ``InstanceNumber``, which runs *against* the spatial order in this
    collection). The geometry is that image's, before any resampling.

    The ellipsoid touches the middle of every face: its semi-axes are half the box's extent,
    edges included, so a one-voxel box has a half-voxel semi-axis and is not lost.

    Returns ``{"center", "axes", "semi_axes", "box_voxels"}``: the centre (mm), the box's
    axes as columns (they are the image's index axes, which is what makes it a box), the
    semi-axes (mm) along them, and the box size in voxels ``(x, y, z)``.
    """
    spacing = np.asarray(spacing, dtype=float)
    z_lo, z_hi = int(min(z_indices)), int(max(z_indices))
    lo = np.array([cols[0] - 1, rows[0] - 1, z_lo], dtype=float)
    hi = np.array([cols[1] - 1, rows[1] - 1, z_hi], dtype=float)
    if np.any(hi < lo):
        raise ValueError(f"box ends before it starts: cols={cols} rows={rows} z={z_lo}..{z_hi}")
    extent_voxels = hi - lo + 1.0
    return {
        "center": index_to_physical((lo + hi) / 2.0, origin, spacing, direction),
        "axes": direction_matrix(direction).copy(),
        "semi_axes": extent_voxels * spacing / 2.0,
        "box_voxels": extent_voxels,
    }


def ellipsoid_mask(shape_zyx, origin, spacing, direction, ellipsoid, box=False):
    """Rasterise an ellipsoid (or, with ``box=True``, its bounding box) on a grid.

    Evaluated in the ellipsoid's own frame, so the target grid may be flipped, permuted or
    tilted relative to the image the box was drawn on: that is exactly the case after the
    reorientation to RAS and the resampling. Only the block of the grid that can contain
    the shape is evaluated, so a small lesion on a large grid costs almost nothing.
    Returns a ``uint8`` array of ``shape_zyx``.
    """
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    centre, axes, semi = ellipsoid["center"], ellipsoid["axes"], ellipsoid["semi_axes"]

    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    corner_points = centre + (corners * semi) @ axes.T
    corner_index = physical_to_index(corner_points, origin, spacing, direction)
    size_xyz = np.array(shape_zyx[::-1])
    lo = np.maximum(np.floor(corner_index.min(axis=0)).astype(int) - 1, 0)
    hi = np.minimum(np.ceil(corner_index.max(axis=0)).astype(int) + 2, size_xyz)
    if np.any(hi <= lo):
        return mask

    xs, ys, zs = (np.arange(lo[a], hi[a]) for a in range(3))
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)  # (nx, ny, nz, 3)
    local = (index_to_physical(grid, origin, spacing, direction) - centre) @ axes  # box axes
    scaled = local / semi
    inside = (np.abs(scaled) <= 1.0 + 1e-9).all(axis=-1) if box else (scaled ** 2).sum(axis=-1) <= 1.0 + 1e-9
    mask[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = inside.transpose(2, 1, 0)
    return mask


def box_volume_ratio(box_voxels, pseudo_voxels):
    """``box volume / pseudo-mask volume`` -- 6/pi = 1.91 for an ellipsoid inscribed in a box."""
    return float(box_voxels) / float(pseudo_voxels) if pseudo_voxels else float("inf")


def ring_around(box, ring_voxels):
    """The shell of ``ring_voxels`` (z, y, x) around the bounding box of ``box``, box excluded."""
    ring = np.zeros(box.shape, dtype=bool)
    if not box.any():
        return ring
    lo = [int(np.where(box.any(axis=tuple(a for a in range(3) if a != k)))[0][0]) for k in range(3)]
    hi = [int(np.where(box.any(axis=tuple(a for a in range(3) if a != k)))[0][-1]) + 1 for k in range(3)]
    window = tuple(slice(max(0, lo[k] - int(ring_voxels[k])), hi[k] + int(ring_voxels[k])) for k in range(3))
    ring[window] = True
    ring &= box == 0
    return ring


def enhancement_contrast(enhancement, pseudo, box, organ, ring_voxels):
    """How far the pseudo-mask stands out from the tissue around its box, in standard deviations.

    ``(mean inside the pseudo-mask - mean of the ring) / std of the ring`` on an enhancement
    volume, the ring being the organ within ``ring_voxels`` of the box (box excluded). A lesion
    enhances more than what surrounds it, so a box on it scores clearly positive; a misplaced or
    far too large box scores near zero.

    Two choices, both measured on the 186 built cases against the same box mirrored into the
    other breast (DOCUMENTATION.md §4.21):

    * the **enhancement is raw** post minus pre, not the difference of two channels z-scored
      separately: that difference compresses the scale (median 0.75 sd on the true boxes) and
      put 65 % of them under a 1.0 sd threshold;
    * the **reference is local**, not the whole organ: the organ holds the heart and the chest
      wall, which enhance strongly and inflate the reference's spread.

    A warning signal about the annotation, not a measurement of it: ``nan`` when there is no
    inside or too small a ring.
    """
    inside = enhancement[(pseudo > 0) & (organ > 0)]
    reference = enhancement[ring_around(box, ring_voxels) & (organ > 0)]
    if inside.size == 0 or reference.size < 2:
        return float("nan")
    return float((inside.mean() - reference.mean()) / (reference.std() + 1e-8))


# --------------------------------------------------------------------- intensity steps

def clip_percentiles(volume, mask, low, high):
    """Clip to the ``[low, high]`` percentiles **of the voxels inside the mask**.

    Per case and per channel, and taken inside the organ so that the air around it -- most of
    a breast MRI -- does not decide where the tissue's range starts. Returns a float32 copy
    and the two bounds.
    """
    inside = volume[mask > 0]
    if inside.size == 0:
        raise ValueError("the mask is empty: nothing to take percentiles of")
    lo, hi = np.percentile(inside, [low, high])
    return np.clip(volume, lo, hi).astype(np.float32), float(lo), float(hi)


def zscore_in_mask(volume, mask):
    """Z-score with the mean and std of the voxels inside the mask; **zero outside it**.

    The stats come from the mask alone (never from the background, never from the whole
    dataset, so nothing is estimated on a test case), and the outside is set to 0 -- the
    new mean -- so it carries no intensity of its own. Returns the volume and ``(mean, std)``.
    """
    inside = volume[mask > 0].astype(np.float64)
    if inside.size == 0:
        raise ValueError("the mask is empty: nothing to normalise")
    mean, std = float(inside.mean()), float(inside.std())
    out = np.zeros(volume.shape, dtype=np.float32)
    out[mask > 0] = ((volume[mask > 0] - mean) / (std + 1e-8)).astype(np.float32)
    return out, (mean, std)


def bbox_slices(mask, margin=0):
    """The ``(z, y, x)`` slices of the mask's bounding box, ``margin`` voxels added, clamped."""
    points = np.argwhere(mask > 0)
    if points.size == 0:
        raise ValueError("the mask is empty: no bounding box")
    lo = np.maximum(points.min(axis=0) - margin, 0)
    hi = np.minimum(points.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


# ---------------------------------------------------------------------------- spacing

def choose_spacing(native_spacing, target_mm, anisotropy_ratio):
    """Per-axis spacing ``(x, y, z)`` to resample to.

    ``target_mm`` everywhere, **except** an axis whose native spacing is more than
    ``anisotropy_ratio`` times the finest in-plane spacing: that axis keeps its native value.
    Upsampling a thick-slice axis invents nothing, and nnU-Net handles anisotropy itself
    (it does not resample the thick axis with the same interpolation), so the spacing is not
    forced to the target there.
    """
    native = np.asarray(native_spacing, dtype=float)
    finest = float(native[:2].min())
    out = np.full(3, float(target_mm))
    for axis in range(3):
        if native[axis] > anisotropy_ratio * finest:
            out[axis] = native[axis]
    return tuple(float(v) for v in out)


def smallest_axis_mm(box_extent_voxels, spacing):
    """Length in mm of a lesion's shortest box axis, for each row of ``box_extent_voxels``."""
    return (np.asarray(box_extent_voxels, float) * np.asarray(spacing, float)).min(axis=1)


def auto_spacing(smallest_axes_mm, min_voxels, percentile, floor_mm, ceil_mm):
    """The isotropic spacing that keeps ``min_voxels`` on the smallest lesion axis.

    Not the median of the dataset, which is what nnU-Net picks by default and which would
    resample a 1 cm lesion to five voxels: the spacing follows the **small** lesions. It is
    ``percentile``-th percentile of the smallest axes (a low one, so most lesions are covered)
    divided by ``min_voxels``, held inside ``[floor_mm, ceil_mm]``. Returns the spacing and
    the share of lesions that end up with at least ``min_voxels`` voxels on that axis.
    """
    axes = np.asarray(smallest_axes_mm, dtype=float)
    raw = float(np.percentile(axes, percentile)) / float(min_voxels)
    spacing = float(min(max(raw, floor_mm), ceil_mm))
    coverage = float((axes / spacing >= min_voxels).mean())
    return spacing, coverage
