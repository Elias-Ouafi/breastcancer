"""The image-library layer: everything that needs SimpleITK.

Reading DICOM *with its geometry* (not just pixels, which is all ``TransformData`` keeps),
N4, the body mask, rigid registration and resampling. Each function returns what it did
alongside its result, so the per-case log can say which parameters were applied and what
happened rather than only that a step ran. Nothing here decides anything; the parameters
come from ``config.yaml`` and the order from ``pipeline.py``.

SimpleITK is imported at module level: this module is only imported by the stages that need
it, and the tests that reach it skip when it is absent.
"""
from __future__ import annotations

import math
import os

import numpy as np
import SimpleITK as sitk

# NIfTI RAS+: array axes increase toward Right, Anterior, Superior. Verified against nibabel
# (`aff2axcodes`): ITK's default LPS direction writes as ('L', 'P', 'S'), and this direction,
# diag(-1, -1, 1) in LPS, writes as ('R', 'A', 'S').
RAS_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)


# --------------------------------------------------------------------------- reading

def dicom_files(folder):
    """The ``.dcm`` files of a series folder, in SimpleITK's spatial order."""
    return list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder))


def slice_positions(files):
    """``(instance_numbers, z_positions)`` per file, in the order of ``files``.

    Read from the headers (no pixels), because the two orders differ in this collection:
    the published boxes count slices by ``InstanceNumber``, the image is laid out by position.
    """
    import pydicom

    instance, position = [], []
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        instance.append(int(ds.InstanceNumber))
        position.append(float(ds.ImagePositionPatient[2]))
    return np.array(instance), np.array(position)


def slice_spacing_deviation(positions):
    """Largest relative deviation of the slice gaps from their median; 0 for a complete series.

    A missing slice doubles one gap; a resorted or duplicated one makes it zero. Either shows
    here and would otherwise reach the resampler as a silently distorted volume.
    """
    gaps = np.abs(np.diff(np.sort(np.asarray(positions, float))))
    if gaps.size == 0:
        return float("inf")
    median = float(np.median(gaps))
    return float(np.abs(gaps - median).max() / median) if median > 0 else float("inf")


def read_series(folder):
    """One DICOM series as ``(image, files)``, geometry included, pixel type preserved."""
    files = dicom_files(folder)
    if not files:
        raise ValueError(f"no DICOM files in {folder}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    return reader.Execute(), files


def series_tags(path):
    """The registry fields of one series, from the header of one of its files."""
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True)

    def tag(name):
        value = getattr(ds, name, None)
        return "" if value is None else str(value).strip()

    return {
        "scanner": " ".join(part for part in (tag("Manufacturer"), tag("ManufacturerModelName")) if part),
        "field_strength_t": tag("MagneticFieldStrength"),
        "site": tag("InstitutionName"),
        "study_date": tag("StudyDate"),
        "series_description": tag("SeriesDescription"),
    }


def to_ras(image):
    """Reorient to the canonical RAS orientation, without resampling (axes are permuted/flipped)."""
    return sitk.DICOMOrient(image, "RAS")


def write_nifti(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(image, path, useCompression=True)


def read_nifti(path):
    return sitk.ReadImage(path)


def same_pixels(a, b):
    """True when two images hold the same pixels on the same physical grid.

    This is the check that lets bronze be deleted after ingestion: the NIfTI is read back from
    disk and compared with what was read from the DICOM. Same pixel type, same values, same
    geometry -- not "the file exists" and not "it opens".
    """
    return (a.GetPixelID() == b.GetPixelID() and a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-6)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-6)
            and np.array_equal(sitk.GetArrayViewFromImage(a), sitk.GetArrayViewFromImage(b)))


def to_array(image):
    return sitk.GetArrayFromImage(image)


def geometry(image):
    """``(origin, spacing, direction)`` of an image, as plain tuples."""
    return image.GetOrigin(), image.GetSpacing(), image.GetDirection()


# ---------------------------------------------------------------------------- steps

def n4_correct(image, cfg):
    """N4 bias-field correction; returns ``(corrected float32 image, info)``."""
    f32 = sitk.Cast(image, sitk.sitkFloat32)
    shrink = [int(cfg["shrink_factor"])] * 3
    small = sitk.Shrink(f32, shrink)
    filt = sitk.N4BiasFieldCorrectionImageFilter()
    filt.SetMaximumNumberOfIterations([int(n) for n in cfg["iterations"]])
    filt.SetConvergenceThreshold(float(cfg["convergence_threshold"]))
    if cfg.get("use_foreground_mask", True):
        foreground = sitk.OtsuThreshold(small, 0, 1, 128)  # below the threshold -> 0, above -> 1
        filt.Execute(small, foreground)
    else:
        filt.Execute(small)
    log_bias = filt.GetLogBiasFieldAsImage(f32)
    corrected = f32 / sitk.Exp(log_bias)
    return corrected, {"shrink_factor": shrink[0], "iterations": list(cfg["iterations"])}


_THRESHOLDS = {
    "otsu": sitk.OtsuThresholdImageFilter,
    "huang": sitk.HuangThresholdImageFilter,
    "li": sitk.LiThresholdImageFilter,
}


def body_mask(image, cfg):
    """Threshold + morphology + the large components + holes filled; ``(uint8 mask, info)``.

    The threshold must separate the body from the air. Otsu does not always: on a volume that
    is mostly air with a bright tail (fat, enhancing gland) it splits the *tissue* into dim and
    bright and keeps only the bright part -- measured on Breast_MRI_017, 1.6 % of the field of
    view kept, and on _118, the two breasts cut apart and the lesion-free one kept. Li's
    minimum cross-entropy threshold lands between air and tissue on every case checked; Huang
    does too but spills into the air noise on some (DOCUMENTATION.md §4.20).

    Every component at least ``keep_components_above`` times the size of the largest is kept,
    so two breasts that the threshold separates both survive; small islands still go.
    """
    f32 = sitk.Cast(image, sitk.sitkFloat32)
    method = cfg.get("threshold", "otsu")
    thresholder = _THRESHOLDS[method]()
    thresholder.SetInsideValue(0)       # below the threshold -> 0, above -> 1
    thresholder.SetOutsideValue(1)
    thresholder.SetNumberOfHistogramBins(int(cfg["histogram_bins"]))
    mask = thresholder.Execute(f32)
    spacing = image.GetSpacing()

    def radius(mm):
        return [max(1, int(round(mm / s))) for s in spacing]

    mask = sitk.BinaryMorphologicalClosing(mask, radius(cfg["closing_radius_mm"]), sitk.sitkBall)
    mask = sitk.BinaryFillhole(mask)
    mask = sitk.BinaryMorphologicalOpening(mask, radius(cfg["opening_radius_mm"]), sitk.sitkBall)
    components = sitk.RelabelComponent(sitk.ConnectedComponent(mask), sortByObjectSize=True)
    labels = sitk.GetArrayViewFromImage(components)
    sizes = np.bincount(labels.ravel())[1:]
    kept = 0
    if sizes.size:
        kept = int((sizes >= float(cfg.get("keep_components_above", 1.0)) * sizes[0]).sum())
    # Relabelled by decreasing size, so the kept components are labels 1..kept.
    mask = sitk.BinaryFillhole(sitk.Cast(sitk.BinaryThreshold(components, 1, max(kept, 1)) if kept
                                         else components * 0, sitk.sitkUInt8))
    voxels = int(sitk.GetArrayViewFromImage(mask).sum())
    # The air/tissue level the lesion guard measures against (pipeline.process_case). Always
    # Li's, whatever builds the mask, so the guard does not inherit the mask's own mistake: an
    # Otsu threshold set too high would call the tissue it cut "air".
    if method == "li":
        tissue = float(thresholder.GetThreshold())
    else:
        li = sitk.LiThresholdImageFilter()
        li.SetNumberOfHistogramBins(int(cfg["histogram_bins"]))
        li.Execute(f32)
        tissue = float(li.GetThreshold())
    return mask, {"threshold_method": method, "threshold": round(float(thresholder.GetThreshold()), 3),
                  "tissue_threshold": round(tissue, 3),
                  "components_kept": kept, "voxels": voxels,
                  "fraction_of_fov": voxels / float(np.prod(image.GetSize()))}


def register_rigid(fixed, moving, fixed_mask, cfg, seed):
    """Rigid (Euler 3D) registration of ``moving`` onto ``fixed``.

    Returns ``(transform, info)``. ``transform`` maps points of ``fixed`` to points of
    ``moving``, which is what ``sitk.Resample(moving, grid, transform)`` wants.

    A transform is **accepted only if it earns it**. Two refusals, both returning the identity
    with ``info["accepted"] = False`` and the reason: the translation or rotation exceeds the
    configured bound, or the correlation between ``fixed`` and the resampled ``moving`` inside
    the mask does not rise by at least ``min_ncc_gain`` over the identity's. The second is the
    one that matters here: for a pre/post pair of the same session the identity is already
    good, and an optimiser will happily find a shift that scores better on its own metric and
    worse on the anatomy (measured: mutual information lowered the correlation in 7 of 8 real
    cases). ``info`` carries both correlations so the decision can be audited.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    method = sitk.ImageRegistrationMethod()
    if cfg.get("metric", "correlation") == "mattes_mi":
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(cfg["histogram_bins"]))
    else:
        method.SetMetricAsCorrelation()
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(float(cfg["sampling_percentage"]), int(seed))
    method.SetMetricFixedMask(fixed_mask)
    method.SetInterpolator(sitk.sitkLinear)
    # Regular-step descent: the step is halved each time the gradient changes direction, so it
    # settles instead of oscillating. A fixed-step descent diverged on a 4.5 mm test shift
    # (22.8 mm found), which is exactly what the acceptance gate below exists to catch.
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(cfg["learning_rate"]), minStep=float(cfg["min_step"]),
        numberOfIterations=int(cfg["iterations"]), relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([int(s) for s in cfg["shrink_factors"]])
    method.SetSmoothingSigmasPerLevel([float(s) for s in cfg["smoothing_sigmas"]])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    initial = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    method.SetInitialTransform(initial, inPlace=False)

    transform = method.Execute(fixed_f, moving_f)
    params = [float(p) for p in transform.GetParameters()]
    rotation_deg = math.degrees(math.sqrt(sum(p * p for p in params[:3])))
    translation_mm = math.sqrt(sum(p * p for p in params[3:6]))
    info = {"metric": float(method.GetMetricValue()), "rotation_deg": round(rotation_deg, 3),
            "translation_mm": round(translation_mm, 3), "stop": method.GetOptimizerStopConditionDescription(),
            "accepted": True}
    identity = sitk.Transform(3, sitk.sitkIdentity)
    if translation_mm > cfg["max_translation_mm"] or rotation_deg > cfg["max_rotation_deg"]:
        info["accepted"] = False
        info["reason"] = (f"transform out of bounds ({translation_mm:.1f} mm > {cfg['max_translation_mm']} "
                          f"or {rotation_deg:.1f} deg > {cfg['max_rotation_deg']}); identity kept")
        return identity, info

    inside = sitk.GetArrayFromImage(fixed_mask) > 0
    reference = sitk.GetArrayFromImage(fixed_f)
    before = _correlation(reference, sitk.GetArrayFromImage(moving_f), inside)
    after = _correlation(reference, sitk.GetArrayFromImage(
        sitk.Resample(moving_f, fixed_f, transform, sitk.sitkLinear, 0.0)), inside)
    info.update(ncc_identity=round(before, 4), ncc_registered=round(after, 4))
    if after < before + float(cfg.get("min_ncc_gain", 0.0)):
        info["accepted"] = False
        info["reason"] = (f"no gain: correlation {before:.3f} (identity) -> {after:.3f} (registered); "
                          "identity kept")
        return identity, info
    return transform, info


def _correlation(a, b, mask):
    """Pearson correlation of two arrays over the voxels of ``mask``."""
    x, y = a[mask].astype(np.float64), b[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))


def target_grid(image, spacing):
    """A RAS-aligned grid covering ``image``'s whole field of view, at ``spacing`` (x, y, z).

    Built in physical (LPS) space from the eight corners of ``image``, so it holds an oblique
    or flipped acquisition without cropping it -- the crop is a later, separate step. Index
    axes point to R, A, S (direction ``RAS_DIRECTION``): the origin is the corner with the
    largest x and y (LPS) and the smallest z.
    """
    size = image.GetSize()
    corners = [image.TransformIndexToPhysicalPoint((i, j, k))
               for i in (0, size[0] - 1) for j in (0, size[1] - 1) for k in (0, size[2] - 1)]
    corners = np.array(corners)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    new_size = [int(math.ceil((hi[a] - lo[a]) / spacing[a])) + 1 for a in range(3)]
    grid = sitk.Image(new_size, sitk.sitkFloat32)
    grid.SetSpacing(tuple(float(s) for s in spacing))
    grid.SetDirection(RAS_DIRECTION)
    grid.SetOrigin((float(hi[0]), float(hi[1]), float(lo[2])))
    return grid


def resample(image, grid, transform=None, order=3):
    """Resample ``image`` onto ``grid``: B-spline of the given order for intensities.

    ``transform`` (e.g. the registration) is applied *in the same interpolation*, so a
    registered channel is interpolated once, not once for the alignment and once for the grid.
    Order 3 can overshoot below zero near sharp edges; the percentile clip that follows takes
    the overshoot with the rest of the tails.
    """
    interpolators = {1: sitk.sitkLinear, 3: sitk.sitkBSpline}   # linear: the contrast index only
    if order not in interpolators:
        raise ValueError(f"interpolation order {order} is not configured (1 or 3)")
    return sitk.Resample(sitk.Cast(image, sitk.sitkFloat32), grid,
                         transform if transform is not None else sitk.Transform(3, sitk.sitkIdentity),
                         interpolators[order], 0.0, sitk.sitkFloat32)


def resample_labels(mask, grid, transform=None):
    """Nearest-neighbour resampling for a label image (never interpolated)."""
    return sitk.Resample(mask, grid,
                         transform if transform is not None else sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def image_like(array_zyx, grid, dtype=np.float32):
    """A SimpleITK image with ``array``'s data and ``grid``'s geometry."""
    image = sitk.GetImageFromArray(np.ascontiguousarray(array_zyx.astype(dtype)))
    image.CopyInformation(grid)
    return image


def crop_image(image, slices_zyx):
    """Crop an image to numpy-style ``(z, y, x)`` slices, keeping its physical geometry."""
    z, y, x = slices_zyx
    return image[x.start:x.stop, y.start:y.stop, z.start:z.stop]
