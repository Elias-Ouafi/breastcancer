import logging
import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
import pydicom

import config
import lineage
import validation

log = logging.getLogger(__name__)


def load_dicom_volume(dicom_dir):
    """
    Load a 3D volume from a folder of DICOM slices.
    Sorts slices by InstanceNumber.
    """
    dicom_files = sorted(
        [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')],
        key=lambda f: int(pydicom.dcmread(os.path.join(dicom_dir, f)).InstanceNumber)
    )
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)).pixel_array for f in dicom_files]
    volume = np.stack(slices, axis=0)  # Shape: (depth, height, width)
    return volume

def normalize_intensity(volume, low_pct=1.0, high_pct=99.0):
    """Clip to the [low_pct, high_pct] intensity percentiles, then z-normalise.

    Plain global z-normalisation (subtract mean, divide by std over the whole
    volume) is skewed in mammography/MRI by the large air/background region:
    the mean and std mostly describe background, not tissue, so the useful
    intensity range gets compressed. Clipping outliers (background floor, any
    saturated pixels) first anchors the standardisation to the tissue range.
    """
    lo, hi = np.percentile(volume, [low_pct, high_pct])
    clipped = np.clip(volume, lo, hi)
    return (clipped - clipped.mean()) / (clipped.std() + 1e-8)


def create_mask(volume_shape, bbox):
    """
    Create a binary mask from bounding box coordinates.
    bbox keys: Start Slice, End Slice, Start Row, End Row, Start Column, End Column
    """
    mask = np.zeros(volume_shape, dtype=np.uint8)
    z0, z1 = bbox['Start Slice'], bbox['End Slice']
    y0, y1 = bbox['Start Row'], bbox['End Row']
    x0, x1 = bbox['Start Column'], bbox['End Column']
    mask[z0:z1, y0:y1, x0:x1] = 1
    return mask


def crop_to_roi(volume, mask, margin=16):
    """Crop `volume` and `mask` to the mask's bounding box plus a voxel `margin`.

    Segmentation masks are almost entirely background, so storing the full volume
    wastes space. When the mask is empty we cannot infer a region of interest, so
    the arrays are returned unchanged.
    Returns (cropped_volume, cropped_mask, offset) where `offset` is the (z, y, x)
    index of the crop origin in the original volume, so the crop can be located
    back in the full image later.
    """
    if not mask.any():
        return volume, mask, (0, 0, 0)

    nonzero = np.argwhere(mask)
    start = np.maximum(nonzero.min(axis=0) - margin, 0)
    end = np.minimum(nonzero.max(axis=0) + margin + 1, mask.shape)

    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
    return volume[slices], mask[slices], tuple(int(s) for s in start)


def save_preprocessed(patient_id, volume, mask, output_dir, dtype=np.float16, crop=True,
                      case_id=None, summary=None, expect_lesion=True):
    """Save a preprocessed volume + mask as a single compressed .npz file.

    Three levers keep the files small:
      - `crop`: keep only the region of interest around the segmentation.
      - `dtype`: store intensities as float16 (half the size of float32); the
        precision loss is negligible for z-normalised MRI data.
      - `np.savez_compressed`: zlib-compresses the arrays; the mostly-empty mask
        shrinks by orders of magnitude.

    `case_id` is the real patient identifier used to group files for a leakage-free
    train/val/test split (several series/views can belong to one patient). It is
    stored inside the .npz; when omitted the filename (`patient_id`) is used.

    Every write goes through `validation.validate_volume_and_mask` first: this is the
    one place all four preprocessing paths funnel through, so it is the only place a
    check has to be written to cover them all. A broken volume raises here rather than
    surfacing hours later as a NaN loss. Pass `summary` a dict to collect the
    per-case stats for the lineage manifest.

    `expect_lesion=False` is for a volume with no annotation (an exam nobody has read yet): its
    mask is empty by construction, and each one reporting an empty mask as a smell would bury
    the warnings that mean something.
    """
    os.makedirs(output_dir, exist_ok=True)

    offset = (0, 0, 0)
    if crop:
        volume, mask, offset = crop_to_roi(volume, mask)

    volume = volume.astype(dtype)
    mask = mask.astype(np.uint8)

    # Validate after the cast: float16 overflow is one of the failures being caught,
    # and it does not exist until the cast has happened.
    key = str(case_id) if case_id is not None else str(patient_id)
    warnings = validation.validate_volume_and_mask(
        volume, mask, case_id=key, expect_full_frame=not crop, expect_lesion=expect_lesion)
    # The summary is keyed by output file, not by case: a patient can contribute several
    # series, and keying by `case_id` made each series overwrite the previous
    # one -- a manifest claiming 72 cases for 147 files, and losing 3 of its 5
    # validation warnings with them. The patient grouping stays readable as `case_id`
    # inside each entry, which is what a leakage-free split needs.
    if summary is not None:
        entry = validation.summarise(volume, mask)
        entry["case_id"] = key
        entry["warnings"] = warnings
        summary[str(patient_id)] = entry

    arrays = {
        "volume": volume,
        "mask": mask,
        "crop_offset": np.asarray(offset, dtype=np.int32),
        "case_id": np.asarray(key),
    }

    out_path = os.path.join(output_dir, f"{patient_id}.npz")
    np.savez_compressed(out_path, **arrays)
    return out_path


def _is_silver_volume(npz_path):
    """True when ``npz_path`` is a readable volume: the check that lets bronze be deleted.

    Opening the archive reads its central directory, not the arrays, so a truncated or
    corrupt file fails here at the cost of a stat and not of a decode. "Exists and is not
    empty" was not enough: a write interrupted halfway leaves a non-empty file.
    """
    try:
        if not os.path.isfile(npz_path) or os.path.getsize(npz_path) == 0:
            return False
        with np.load(npz_path) as archive:
            return "volume" in archive.files and "mask" in archive.files
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def purge_bronze_series(folder, silver_files, bronze_root=config.TCIA_DIR):
    """Delete the bronze series ``folder`` once it is safely held in silver.

    The medallion rule: raw bytes are kept only until they are in silver, so the data
    ends up held once. This is the one place that deletes them, and it is destructive --
    the DICOM series is not recoverable except by downloading it again -- so it refuses
    rather than guesses:

    * ``folder`` must be a direct child of ``bronze_root``; anything else raises
      ``ValueError`` (a wrong path here must never reach ``rmtree``);
    * every path of ``silver_files`` must be a readable ``.npz`` holding a volume and a
      mask (:func:`_is_silver_volume`), and there must be at least one. Pass one file per
      corpus that reads the series: purging on the first corpus alone would strand the
      others.

    Returns the number of bytes freed, or 0 when nothing was deleted (silver not ready,
    folder already gone, or the deletion failed part-way -- the next run retries, the
    silver file being already safe).
    """
    root = os.path.normcase(os.path.realpath(bronze_root))
    target = os.path.normcase(os.path.realpath(folder))
    if os.path.dirname(target) != root:
        raise ValueError(f"refusing to delete {folder!r}: not a direct child of the "
                         f"bronze root {bronze_root!r}")
    if not os.path.isdir(folder):
        return 0

    silver_files = list(silver_files)
    not_ready = [f for f in silver_files if not _is_silver_volume(f)]
    if not silver_files or not_ready:
        log.warning(f"Keeping {folder} in bronze: silver is not ready "
                    f"({not_ready or 'no silver file given'}).")
        return 0

    n_bytes = sum(os.stat(os.path.join(d, f)).st_size
                  for d, _, files in os.walk(folder) for f in files)
    try:
        shutil.rmtree(folder)
    except OSError as e:
        log.error(f"Failed to remove {folder}: {e}")
        return 0
    log.info(f"Purged bronze {folder} ({n_bytes / 1e6:.0f} MB), held in silver.")
    return n_bytes


def extract_patient_ids(root_dir=config.TCIA_DIR):
    """Walk `root_dir` and return the set of PatientIDs found in the DICOM files."""
    patient_ids = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                filepath = os.path.join(dirpath, filename)
                try:
                    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                    patient_ids.add(ds.PatientID)
                except Exception as e:
                    log.info(f"Error reading {filepath}: {e}")
    return patient_ids


# --------------------------------------------------------------------------- #
# DCE-MRI (Duke-Breast-Cancer-MRI): multiphase subtraction + box annotations
# --------------------------------------------------------------------------- #

# Duke-Breast-Cancer-MRI mixes two SeriesDescription conventions across its cohort
# (protocol/scanner varied over the years the collection was acquired):
#   A) "ax dyn pre", "ax dyn 1st pass" .. "ax dyn 4th pass"       (spelled-out pass)
#   B) "ax 3d dyn" (bare = pre-contrast), "Ph1/ax 3d dyn" .. "Ph4/..." (Ph-prefixed)
# Verified on a 10-patient sample: exactly 5/10 use each convention, so both must be
# handled or half the cohort silently loses its DCE series.
_DCE_PASS_WORDS = [("1st", 1), ("2nd", 2), ("3rd", 3), ("4th", 4)]
_DCE_PH_PREFIX_RE = re.compile(r"\bph\s*(\d)\b")


def _dce_phase_rank(series_description):
    """Return the phase rank (0=pre-contrast, 1..4=post-contrast passes) for a
    Duke-Breast-Cancer-MRI ``SeriesDescription``, or None if it doesn't match the
    dynamic-contrast protocol (e.g. the unrelated "ax t1 tse +c" or SEG series)."""
    desc = (series_description or "").lower()
    m = _DCE_PH_PREFIX_RE.search(desc)
    if m:
        return int(m.group(1))
    for key, rank in _DCE_PASS_WORDS:
        if key in desc:
            return rank
    if "pre" in desc:
        return 0
    if "dyn" in desc:
        # Convention B's pre-contrast series carries no "Ph"/"pass"/"pre" marker at
        # all -- a bare dynamic series is the baseline by elimination.
        return 0
    return None


def group_dce_series_by_patient(root_dir):
    """Walk ``root_dir``'s series subfolders (one ``SeriesInstanceUID`` per folder,
    as written by :func:`ExtractData.download_dce_mri_series`) and group them by
    ``PatientID -> {phase_rank: folder_path}``.

    Only folders whose ``SeriesDescription`` matches the DCE dynamic protocol are
    kept; unrelated series (T1 TSE, SEG) are ignored here.
    """
    groups = {}
    for name in sorted(os.listdir(root_dir)):
        folder = os.path.join(root_dir, name)
        if not os.path.isdir(folder):
            continue
        dcm_files = [f for f in os.listdir(folder) if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue
        ds = pydicom.dcmread(os.path.join(folder, dcm_files[0]), stop_before_pixels=True)
        rank = _dce_phase_rank(getattr(ds, "SeriesDescription", ""))
        if rank is None:
            continue
        pid = getattr(ds, "PatientID", None)
        if pid is None:
            continue
        groups.setdefault(pid, {})[rank] = folder
    return groups


def _load_series_volume(dicom_dir):
    """Load one DICOM series folder into a ``(depth, H, W)`` float32 array, sorted
    by ``InstanceNumber`` (slice order within the series)."""
    dcm_files = sorted(
        (f for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm")),
        key=lambda f: int(pydicom.dcmread(os.path.join(dicom_dir, f),
                                          stop_before_pixels=True).InstanceNumber),
    )
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)).pixel_array for f in dcm_files]
    return np.stack(slices, axis=0).astype(np.float32)


def _read_mri_boxes(boxes_path):
    """Read the Duke-Breast-Cancer-MRI ``Annotation_Boxes`` file (``.xlsx`` or the
    ``.csv`` produced by :func:`ExtractData.clean_mri_annotation`) into a DataFrame.

    Expected columns (1-indexed, inclusive, as published by TCIA): ``Patient ID``,
    ``Start Row``, ``End Row``, ``Start Column``, ``End Column``, ``Start Slice``,
    ``End Slice``. Column names are normalised (stripped) so minor header variants
    still match.
    """
    if str(boxes_path).lower().endswith(".xlsx"):
        df = pd.read_excel(boxes_path)
    else:
        df = pd.read_csv(boxes_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def dce_subtraction(pre, post):
    """Build the enhancement volume of a DCE-MRI pair: ``post - pre``, clipped at 0,
    then z-normalised.

    Both DCE preprocessing paths call this, so an exam prepared for inference cannot
    end up on a different intensity scale from the corpus the checkpoint was trained
    on. A test pins that equality.
    """
    return normalize_intensity(np.clip(post - pre, a_min=0, a_max=None))


def _with_previous_cases(output_dir, summary):
    """``summary`` plus the manifest entries of volumes this pass did not rewrite.

    A manifest describes the corpus, not the last run. It used to be rewritten from the
    cases of the pass alone, which was harmless while every pass re-read all of bronze.
    Bronze is now purged once a series is in silver, so a second pass only sees what is
    left -- often nothing -- and would replace a 186-case manifest with an empty one
    while the 186 volumes sit beside it. An entry is carried only when its ``.npz`` is
    still there, so a volume deleted by hand does not linger in the record.
    """
    previous = (lineage.read_manifest(output_dir) or {}).get("cases") or {}
    carried = {case: entry for case, entry in previous.items()
               if case not in summary
               and os.path.exists(os.path.join(output_dir, f"{case}.npz"))}
    return {**carried, **summary}


def preprocess_dce_mri_exams(root_dir,
                             output_dir=config.DCE_MRI_SILVER_DIR,
                             post_phase_rank=2):
    """Preprocess DCE-MRI exams that carry no annotation, ready for inference.

    :func:`preprocess_dce_mri_with_boxes` skips every patient absent from the
    annotation table, so it cannot prepare an exam nobody has read yet -- which is
    every new exam. This path asks only for what a subtraction needs: the
    pre-contrast series and the chosen post-contrast pass.

    The mask is empty and the frame is never cropped. Both follow from having no
    annotation: there is no lesion to crop to, and full frame is the geometry the
    served checkpoint was trained on, so a crop would hand the model a picture it has
    never seen.

    Returns ``(saved, skipped)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    groups = group_dce_series_by_patient(root_dir)

    saved, skipped, summary = 0, 0, {}
    for pid, phases in groups.items():
        if 0 not in phases or post_phase_rank not in phases:
            log.warning(f"[MRI] {pid}: missing pre or post-phase {post_phase_rank} series, skipping.")
            skipped += 1
            continue

        pre = _load_series_volume(phases[0])
        post = _load_series_volume(phases[post_phase_rank])
        if pre.shape != post.shape:
            log.warning(f"[MRI] {pid}: phase shape mismatch {pre.shape} vs {post.shape}, skipping.")
            skipped += 1
            continue

        subtraction = dce_subtraction(pre, post)
        save_preprocessed(pid, subtraction, np.zeros(subtraction.shape, dtype=np.uint8),
                          output_dir, crop=False, case_id=pid, summary=summary,
                          expect_lesion=False)
        saved += 1
        log.info(f"[MRI] {pid}: {subtraction.shape[0]} slices -> saved (no annotation).")

    cases = _with_previous_cases(output_dir, summary)
    lineage.write_manifest(
        output_dir,
        source=root_dir,
        parameters={
            "pipeline": "preprocess_dce_mri_exams",
            "post_phase_rank": post_phase_rank,
            "crop": False,
            "annotated": False,
            "skipped": skipped,
        },
        cases=cases,
        warnings=[w for case in cases.values() for w in case.get("warnings", [])],
    )

    log.info(f"[MRI] Saved {saved} unannotated exam(s), skipped {skipped}.")
    return saved, skipped


def preprocess_dce_mri_with_boxes(root_dir, boxes_path,
                                  output_dir=config.DCE_MRI_SILVER_DIR,
                                  post_phase_rank=2, crop=False):
    """Preprocess Duke-Breast-Cancer-MRI series into subtraction volumes + box masks.

    For each patient with both a pre-contrast (rank 0) and the chosen post-contrast
    pass (``post_phase_rank``) series available, this:

    ``post_phase_rank`` defaults to 2 (the *second* post-contrast pass). Zhou et al.,
    "U-Net breast lesion segmentations for breast dynamic contrast-enhanced MRI"
    (https://pmc.ncbi.nlm.nih.gov/articles/PMC10658935/) compared subtraction inputs
    head-to-head on this exact task and found second-post-contrast subtraction
    significantly better than first (DSC p<0.05, for both 2D and 3D U-Nets) -- the
    second pass sits nearer peak enhancement, so malignant uptake stands out more
    against background parenchyma. Verified here that both passes are available for
    the same 186/189 patients, so the switch costs no sample size. Pass
    ``post_phase_rank=1`` to reproduce the earlier first-pass behaviour.

    1. Loads both phases (:func:`_load_series_volume`) -- they are acquired in the
       same session without repositioning, so no inter-phase registration is applied
       for this first pass.
    2. Computes the enhancement volume with :func:`dce_subtraction` -- the same
       convention every DCE-MRI path shares, so the resulting ``.npz`` is a
       drop-in for the existing ``imaging/`` training/inference code.
    3. Builds a binary lesion mask from the matching row(s) in the TCIA annotation
       boxes file via :func:`create_mask` (1-indexed bounds are converted to the
       0-indexed slicing ``create_mask``/numpy expect).

    ``crop`` defaults to ``False`` because that is the corpus the served checkpoint
    was trained on -- measured, not assumed: every volume under
    ``data/silver/dce_mri_p2/`` carries ``crop_offset == (0, 0, 0)`` at
    full 512x512. Cropping to the lesion ROI is what made the task artificially easy
    and produced the confidence-always-1.0 bug (DOCUMENTATION.md section 4.2), so the
    default used to contradict every caller in the repository.

    Patients with mismatched phase shapes (rare acquisition inconsistencies) or no
    matching box row are skipped -- for an exam that has no annotation at all, use
    :func:`preprocess_dce_mri_exams`. Returns ``(saved, skipped)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    boxes = _read_mri_boxes(boxes_path)
    groups = group_dce_series_by_patient(root_dir)

    saved, skipped, summary = 0, 0, {}
    for pid, phases in groups.items():
        if 0 not in phases or post_phase_rank not in phases:
            log.warning(f"[MRI] {pid}: missing pre or post-phase {post_phase_rank} series, skipping.")
            skipped += 1
            continue

        rows = boxes[boxes["Patient ID"] == pid]
        if rows.empty:
            skipped += 1
            continue

        pre = _load_series_volume(phases[0])
        post = _load_series_volume(phases[post_phase_rank])
        if pre.shape != post.shape:
            log.warning(f"[MRI] {pid}: phase shape mismatch {pre.shape} vs {post.shape}, skipping.")
            skipped += 1
            continue

        subtraction = dce_subtraction(pre, post)

        mask = np.zeros(subtraction.shape, dtype=np.uint8)
        for _, r in rows.iterrows():
            bbox = {
                # TCIA boxes are 1-indexed inclusive; create_mask/numpy slicing is
                # 0-indexed exclusive on the end, so subtract 1 only from the starts.
                "Start Slice": int(r["Start Slice"]) - 1, "End Slice": int(r["End Slice"]),
                "Start Row": int(r["Start Row"]) - 1, "End Row": int(r["End Row"]),
                "Start Column": int(r["Start Column"]) - 1, "End Column": int(r["End Column"]),
            }
            mask = np.logical_or(mask, create_mask(subtraction.shape, bbox)).astype(np.uint8)

        save_preprocessed(pid, subtraction, mask, output_dir, crop=crop, case_id=pid,
                          summary=summary)
        saved += 1
        log.info(f"[MRI] {pid}: {len(rows)} box(es), {int(mask.sum())} lesion voxels -> saved.")

    # Written last, and only on a completed pass: a folder with no manifest is a
    # folder whose run was interrupted, which is exactly what you want to know.
    cases = _with_previous_cases(output_dir, summary)
    lineage.write_manifest(
        output_dir,
        source=root_dir,
        parameters={
            "pipeline": "preprocess_dce_mri_with_boxes",
            "post_phase_rank": post_phase_rank,
            "crop": crop,
            "boxes": lineage.relative_path(boxes_path),
            "skipped": skipped,
        },
        cases=cases,
        warnings=[w for case in cases.values() for w in case.get("warnings", [])],
    )

    log.info(f"[MRI] Saved {saved} patients, skipped {skipped}.")
    return saved, skipped


def make_demo_case(source_npz, out_path, slice_index, slim=True, slab=12):
    """Write ``source_npz`` to ``out_path`` with a ``forced_slice`` key added.

    ``inference._localize_lesion`` scores only ``forced_slice`` when present instead
    of scanning the whole volume for the highest-confidence slice. This exists
    because that scan is currently unreliable on full-frame DCE-MRI (confidence
    saturates near 1.0 on almost every slice -- verified 0/186 on held-out patients,
    see DOCUMENTATION.md §4.1/§4.2): the model segments a lesion well *once shown the right
    slice*, it just cannot reliably find that slice on its own yet. Demo cases are
    curated by hand (pick a real, verified-good slice) so the app has something
    trustworthy to show while that ranking problem is being worked on separately --
    this is a known, documented limitation, not a hidden shortcut.

    ``slim`` (the default) keeps a small **slab** centred on that slice instead of the
    whole volume: ``slab`` slices either side, so ``2 * slab + 1`` in total. Since
    ``forced_slice`` means only the centre slice is ever scored, carrying all ~176
    costs ~30 MB per case for nothing; a slab of 25 is ~4 MB, small enough to live in
    git so the demo works straight out of a clone. The neighbours are kept (rather
    than the centre alone) so the UI can offer slice-by-slice navigation and a MIP,
    which is what makes a lesion legible -- it should appear and disappear as you
    scroll, not just sit there in a single frame.

    The dropped depth is preserved in ``source_n_slices`` and the slab's first index
    is folded into ``crop_offset[0]``, so the app still reports "slice 52 of 176" --
    the same numbers a full-volume case produces. ``slab=0`` keeps the centre slice
    alone; ``slim=False`` keeps the whole volume (e.g. to re-derive another slice).
    """
    with np.load(source_npz) as data:
        kwargs = {k: data[k] for k in data.files}

    slice_index = int(slice_index)
    if not slim:
        kwargs["forced_slice"] = np.asarray(slice_index)
        np.savez_compressed(out_path, **kwargs)
        return out_path

    depth = int(kwargs["volume"].shape[0])
    if not 0 <= slice_index < depth:
        raise IndexError(f"slice_index {slice_index} out of range for depth {depth}.")

    offset = kwargs.get("crop_offset")
    offset = np.zeros(3, dtype=np.int32) if offset is None else np.asarray(offset, dtype=np.int32)

    lo = max(0, slice_index - int(slab))
    hi = min(depth, slice_index + int(slab) + 1)
    for key in ("volume", "mask"):
        if key in kwargs:
            kwargs[key] = kwargs[key][lo:hi]

    # The slab's first slice becomes index 0, so its original position moves into the
    # z offset -- exactly the mechanism `save_preprocessed`'s crop already uses, which
    # is what makes `best_slice` and `render_overlay_png` come out unchanged.
    kwargs["crop_offset"] = np.asarray([offset[0] + lo, offset[1], offset[2]], dtype=np.int32)
    kwargs["forced_slice"] = np.asarray(slice_index - lo)
    kwargs["source_n_slices"] = np.asarray(offset[0] + depth, dtype=np.int32)
    np.savez_compressed(out_path, **kwargs)
    return out_path
