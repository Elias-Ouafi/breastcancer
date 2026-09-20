"""Bronze DICOM -> native NIfTI -> the nnU-Net MRI corpus, one case at a time.

Two stages, kept apart because they change at different speeds:

* **ingest**  DICOM -> NIfTI in native geometry, reoriented to RAS, every dynamic phase,
  plus the box as physical-space ellipsoids and a registry row. Slow, and the only stage that
  reads bronze. Its output (``native/``) is lossless, so once it exists bronze is no longer
  needed to re-tune anything that follows.
* **process** native NIfTI -> the case the network trains on, in the order that is not
  negotiable: N4 -> body mask -> rigid registration -> resampling -> percentile clip -> z-score
  -> crop. The pseudo-mask is rasterised on the final grid from the ellipsoid, never
  interpolated.

Every case is idempotent (the hash of the parameters it depends on is stored beside it, and
an unchanged case is skipped), every case leaves a line in ``log/cases.jsonl``, and a case
that cannot be built leaves a line in ``exclusions.csv`` instead of failing silently or
taking the run down.
"""
from __future__ import annotations

import json
import logging
import os
import traceback

import numpy as np

import lineage

from . import registry, steps
from . import settings as settings_module
from . import spacing as spacing_module

log = logging.getLogger(__name__)

# Tolerances for "these two phases are the same grid": one hundredth of a millimetre on the
# origin, a thousandth on the spacing and direction. Same-session series agree far tighter.
GEOMETRY_ATOL = 1e-2


# ------------------------------------------------------------------------- inputs

def discover(raw_dir):
    """``{patient_id: {phase_rank: series_folder}}`` for the dynamic series under ``raw_dir``.

    The grouping and the phase convention are ``TransformData``'s (two ``SeriesDescription``
    conventions coexist in the collection), reused rather than restated. Empty when the
    folder is gone, which is the normal state after the bronze purge.
    """
    if not raw_dir or not os.path.isdir(raw_dir):
        return {}
    from TransformData import group_dce_series_by_patient

    return group_dce_series_by_patient(raw_dir)


def read_boxes(boxes_path):
    """``{patient_id: [{"cols", "rows", "slices"}, ...]}`` -- 1-based inclusive, as published."""
    from TransformData import _read_mri_boxes

    table = _read_mri_boxes(boxes_path)
    out = {}
    for pid, rows in table.groupby("Patient ID"):
        out[str(pid)] = [{"cols": (int(r["Start Column"]), int(r["End Column"])),
                          "rows": (int(r["Start Row"]), int(r["End Row"])),
                          "slices": (int(r["Start Slice"]), int(r["End Slice"]))}
                         for _, r in rows.iterrows()]
    return out


def _channel(settings, name):
    return next(c for c in settings["channels"] if c["name"] == name)


def _paths(out_dir, pid):
    native = os.path.join(out_dir, "native", pid)
    cases = os.path.join(out_dir, "cases", pid)
    return native, cases


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


# ------------------------------------------------------------------------- ingest

def _same_grid(a, b):
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-3)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-3)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=GEOMETRY_ATOL))


def ingest_case(pid, folders, boxes, settings, out_dir, force=False):
    """DICOM -> native RAS NIfTI for one patient. Returns ``"ok"``, ``"skipped"`` or ``"excluded"``."""
    from . import sitk_io

    native, _ = _paths(out_dir, pid)
    marker = os.path.join(native, "ingest.json")
    params_hash = settings_module.section_hash(settings, settings_module.INGEST_SECTIONS)
    if not force and os.path.exists(marker) and _load_json(marker).get("params_hash") == params_hash:
        return "skipped"

    case_log = registry.CaseLog(out_dir, pid, "ingest")

    def excluded(reason, detail=""):
        registry.exclude(out_dir, pid, "ingest", reason, detail)
        case_log.close("excluded", reason=reason, detail=str(detail)[:300])
        log.warning("[nnunet] %s excluded at ingestion: %s %s", pid, reason, detail)
        return "excluded"

    needed = sorted({c["phase"] for c in settings["channels"]})
    wanted = sorted(set(settings["ingest"]["phases"]) | set(needed))
    absent = [p for p in needed if p not in folders]
    if absent:
        return excluded("missing_phase", f"channel phase(s) {absent} not downloaded")
    if not boxes.get(pid):
        return excluded("no_box", "no row in the annotation table")

    series = {}
    for phase in wanted:
        if phase not in folders:
            case_log.anomaly(f"phase {phase} not downloaded")
            continue
        try:
            with case_log.timed("read_dicom", phase=phase) as details:
                image, files = sitk_io.read_series(folders[phase])
                listed = [f for f in os.listdir(folders[phase]) if f.lower().endswith(".dcm")]
                if len(listed) != len(files):
                    # GDCM drops a file it cannot parse without saying so; a series is only as
                    # complete as the files that were actually read.
                    raise ValueError(f"{len(listed)} .dcm files in the folder, {len(files)} readable")
                instance, position = sitk_io.slice_positions(files)
                deviation = sitk_io.slice_spacing_deviation(position)
                details.update(size=list(image.GetSize()), spacing=[round(s, 4) for s in image.GetSpacing()],
                               slice_gap_deviation=round(deviation, 4))
        except Exception as exc:  # a corrupt series is a fact about the case, not a crash of the run
            if phase in needed:
                return excluded("unreadable_series", f"phase {phase}: {type(exc).__name__}: {exc}")
            case_log.anomaly(f"phase {phase} unreadable: {exc}")
            continue
        if deviation > settings["ingest"]["max_slice_spacing_deviation"]:
            if phase in needed:
                return excluded("incomplete_series",
                                f"phase {phase}: slice gaps deviate {deviation:.1%} from their median")
            case_log.anomaly(f"phase {phase}: slice gaps deviate {deviation:.1%}; not used as a channel")
            continue
        series[phase] = (image, files, instance)

    reference_phase = _channel(settings, settings["reference_channel"])["phase"]
    reference_image = series[reference_phase][0]
    for phase in needed:
        if not _same_grid(reference_image, series[phase][0]):
            return excluded("geometry_mismatch", f"phase {phase} is not on the grid of phase {reference_phase}")

    verified = []
    for phase, (image, _, _) in series.items():
        path = os.path.join(native, f"{pid}_ph{phase}.nii.gz")
        with case_log.timed("write_nifti_ras", phase=phase) as details:
            oriented = sitk_io.to_ras(image)
            sitk_io.write_nifti(oriented, path)
            details["lossless"] = bool(sitk_io.same_pixels(oriented, sitk_io.read_nifti(path)))
        if not details["lossless"]:
            os.remove(path)
            return excluded("nifti_mismatch", f"phase {phase}: the NIfTI read back differs from the DICOM")
        verified.append(phase)

    # The box is drawn on the reference series' own grid, so it is converted there, once, into
    # physical-space ellipsoids that survive every later reorientation, crop and resampling.
    _, files, instance = series[reference_phase]
    if settings["pseudo_mask"]["slice_basis"] == "instance_number":
        z_of_rank = np.argsort(instance, kind="stable")  # rank r by InstanceNumber -> z index
    else:
        z_of_rank = np.arange(len(files))
    origin, spacing_xyz, direction = sitk_io.geometry(reference_image)
    entries = []
    try:
        for box in boxes[pid]:
            z_lo, z_hi = box["slices"]
            if z_lo < 1 or z_hi > len(files):
                raise ValueError(f"box slices {box['slices']} outside the {len(files)}-slice series")
            z_indices = z_of_rank[z_lo - 1:z_hi]
            ellipsoid = steps.box_to_ellipsoid(box["cols"], box["rows"], z_indices, origin, spacing_xyz, direction)
            entries.append({"published": box, "z_indices": [int(z) for z in z_indices],
                            "ellipsoid": {k: np.asarray(v).tolist() for k, v in ellipsoid.items()}})
    except ValueError as exc:
        return excluded("bad_box", exc)
    _dump_json(os.path.join(native, "box.json"), {
        "reference_phase": reference_phase, "slice_basis": settings["pseudo_mask"]["slice_basis"],
        "native_spacing": list(spacing_xyz), "native_size": list(reference_image.GetSize()),
        "boxes": entries})

    tags = sitk_io.series_tags(files[0])
    registry.upsert(os.path.join(out_dir, "registry.csv"), registry.REGISTRY_COLUMNS, "patient_id", {
        "patient_id": pid, "scanner": tags["scanner"], "field_strength_t": tags["field_strength_t"],
        "site": tags["site"], "study_date": tags["study_date"],
        "phases_available": ",".join(str(p) for p in sorted(series)),
        "series_descriptions": "; ".join(
            f"{p}:{sitk_io.series_tags(series[p][1][0])['series_description']}" for p in sorted(series)),
        "n_slices": len(files), "native_spacing_mm": "x".join(f"{s:.3f}" for s in spacing_xyz),
        "status": "ingested"})
    registry.clear_exclusion(out_dir, pid)
    _dump_json(marker, {"params_hash": params_hash, "phases": sorted(series),
                        "verified_phases": sorted(verified)})
    case_log.close("ok", phases=sorted(series))
    return "ok"


def verified_phases(out_dir, pid):
    """The phases of ``pid`` whose native NIfTI was written **and read back identical** to the DICOM.

    This is what the bronze purge asks before deleting a DICOM series: the lossless copy exists,
    it is on disk now, and it matched at the time it was written.
    """
    native, _ = _paths(out_dir, pid)
    try:
        phases = _load_json(os.path.join(native, "ingest.json")).get("verified_phases", [])
    except (OSError, ValueError):
        return set()
    return {int(p) for p in phases
            if os.path.exists(os.path.join(native, f"{pid}_ph{p}.nii.gz"))
            and os.path.getsize(os.path.join(native, f"{pid}_ph{p}.nii.gz")) > 0}


# ------------------------------------------------------------------------ process

def _outputs(settings, out_dir, pid):
    _, cases = _paths(out_dir, pid)
    files = [os.path.join(cases, f"{pid}_{i:04d}.nii.gz") for i in range(len(settings["channels"]))]
    files += [os.path.join(cases, f"{pid}_label.nii.gz"), os.path.join(cases, f"{pid}_bodymask.nii.gz")]
    return files


def process_case(pid, settings, out_dir, spacing_mm, force=False):
    """Native NIfTI -> the case nnU-Net trains on. Returns ``"ok"``, ``"skipped"`` or ``"excluded"``."""
    from . import sitk_io

    native, cases = _paths(out_dir, pid)
    marker = os.path.join(cases, "case.json")
    params_hash = settings_module.section_hash(settings, settings_module.PROCESS_SECTIONS,
                                               extra={"spacing_mm": spacing_mm})
    outputs = _outputs(settings, out_dir, pid)
    if (not force and os.path.exists(marker) and _load_json(marker).get("params_hash") == params_hash
            and all(os.path.exists(f) for f in outputs)):
        return "skipped"

    case_log = registry.CaseLog(out_dir, pid, "process")

    def excluded(reason, detail=""):
        registry.exclude(out_dir, pid, "process", reason, detail)
        case_log.close("excluded", reason=reason, detail=str(detail)[:300])
        log.warning("[nnunet] %s excluded at processing: %s %s", pid, reason, detail)
        return "excluded"

    box = _load_json(os.path.join(native, "box.json"))
    channels = settings["channels"]
    reference = _channel(settings, settings["reference_channel"])
    images = {c["name"]: sitk_io.read_nifti(os.path.join(native, f"{pid}_ph{c['phase']}.nii.gz"))
              for c in channels}

    # 1. N4, before anything else looks at the intensities.
    corrected = {}
    for c in channels:
        with case_log.timed("n4", channel=c["name"]) as details:
            if settings["n4"]["enabled"]:
                corrected[c["name"]], info = sitk_io.n4_correct(images[c["name"]], settings["n4"])
                details.update(info)
            else:
                corrected[c["name"]] = sitk_io.sitk.Cast(images[c["name"]], sitk_io.sitk.sitkFloat32)
                details["skipped"] = True
    ref_image = corrected[reference["name"]]

    # 2. Organ mask, on the corrected reference.
    with case_log.timed("body_mask") as details:
        mask_native, info = sitk_io.body_mask(ref_image, settings["body_mask"])
        details.update(info)
    if info["voxels"] == 0:
        return excluded("empty_body_mask", "Otsu + morphology left nothing")

    # 3. Rigid co-registration of every other channel to the reference.
    transforms = {}
    for c in channels:
        if c["name"] == reference["name"] or not settings["registration"]["enabled"]:
            continue
        with case_log.timed("register", channel=c["name"]) as details:
            transform, info = sitk_io.register_rigid(ref_image, corrected[c["name"]], mask_native,
                                                     settings["registration"], settings["seed"])
            details.update(info)
        if not info["accepted"]:
            case_log.anomaly(f"{c['name']}: {info['reason']}")
        transforms[c["name"]] = transform

    # 4. Resampling: images by B-spline (registration folded into the same interpolation),
    #    the organ mask by nearest neighbour. Anisotropic axes keep their native spacing.
    with case_log.timed("resample") as details:
        spacing = steps.choose_spacing(ref_image.GetSpacing(), spacing_mm,
                                       settings["resample"]["anisotropy_ratio"])
        grid = sitk_io.target_grid(ref_image, spacing)
        resampled = {c["name"]: sitk_io.resample(corrected[c["name"]], grid, transforms.get(c["name"]),
                                                 settings["resample"]["image_interpolation_order"])
                     for c in channels}
        organ = sitk_io.to_array(sitk_io.resample_labels(mask_native, grid))
        details.update(spacing_mm=[round(s, 4) for s in spacing], size=list(grid.GetSize()))

    # 5-6. Clip on the percentiles of the mask, then z-score with its mean/std; 0 outside.
    volumes, stats = {}, {}
    with case_log.timed("normalize") as details:
        for c in channels:
            array = sitk_io.to_array(resampled[c["name"]])
            clipped, low, high = steps.clip_percentiles(array, organ, *settings["normalize"]["clip_percentiles"])
            volumes[c["name"]], (mean, std) = steps.zscore_in_mask(clipped, organ)
            stats[c["name"]] = {"clip": [round(low, 3), round(high, 3)], "mean": round(mean, 3), "std": round(std, 3)}
        details["channels"] = stats

    # The pseudo-mask, rasterised on the final grid from the physical ellipsoid.
    origin, spacing_xyz, direction = sitk_io.geometry(grid)
    shape = organ.shape
    ellipsoids = [{k: np.asarray(v) for k, v in entry["ellipsoid"].items()} for entry in box["boxes"]]
    pseudo = np.zeros(shape, dtype=np.uint8)
    box_mask = np.zeros(shape, dtype=np.uint8)
    for e in ellipsoids:
        pseudo |= steps.ellipsoid_mask(shape, origin, spacing_xyz, direction, e)
        box_mask |= steps.ellipsoid_mask(shape, origin, spacing_xyz, direction, e, box=True)
    full_pseudo = int(pseudo.sum())
    if full_pseudo == 0:
        return excluded("empty_pseudo_mask", "the box did not land on the resampled grid")

    # 7. Crop on the organ mask's bounding box.
    with case_log.timed("crop") as details:
        window = steps.bbox_slices(organ, settings["crop"]["margin_voxels"])
        organ_c, pseudo_c, box_c = organ[window], pseudo[window], box_mask[window]
        volumes_c = {name: v[window] for name, v in volumes.items()}
        cropped_grid = sitk_io.crop_image(grid, window)
        retained = float(pseudo_c.sum()) / full_pseudo
        details.update(shape=list(organ_c.shape), lesion_retained=round(retained, 4))
    if retained < 0.95:
        return excluded("lesion_outside_organ_mask",
                        f"only {retained:.0%} of the pseudo-mask lies inside the cropped organ box")

    # Report on the boxes: the ratio a pseudo-mask must have, and whether the corners look loose.
    ratio = steps.box_volume_ratio(box_c.sum(), pseudo_c.sum())
    expected = 1.0 / steps.INSCRIBED_ELLIPSOID_RATIO
    contrast = float("nan")
    if len(channels) >= 2:
        first, second = (volumes_c[c["name"]] for c in channels[:2])
        contrast = steps.enhancement_contrast(second - first, pseudo_c, box_c, organ_c)
    if abs(ratio / expected - 1.0) > settings["pseudo_mask"]["ratio_tolerance"]:
        case_log.anomaly(f"box/pseudo-mask volume ratio {ratio:.2f} is far from 6/pi = {expected:.2f} "
                         "(very small box, or clipped by the volume)")
    if not np.isnan(contrast) and contrast < settings["pseudo_mask"]["contrast_warn_below"]:
        case_log.anomaly(f"loose or misplaced box? the pseudo-mask stands out by only {contrast:.2f} sd "
                         f"from the organ (< {settings['pseudo_mask']['contrast_warn_below']})")

    # Outputs.
    for i, c in enumerate(channels):
        sitk_io.write_nifti(sitk_io.image_like(volumes_c[c["name"]], cropped_grid), outputs[i])
    sitk_io.write_nifti(sitk_io.image_like(pseudo_c, cropped_grid, np.uint8), outputs[-2])
    sitk_io.write_nifti(sitk_io.image_like(organ_c, cropped_grid, np.uint8), outputs[-1])

    voxel_mm3 = float(np.prod(spacing_xyz))
    summary = {
        "shape": list(organ_c.shape), "spacing_mm": [round(s, 4) for s in spacing_xyz],
        "lesion_voxels": int(pseudo_c.sum()), "lesion_volume_mm3": round(int(pseudo_c.sum()) * voxel_mm3, 1),
        "box_to_pseudo_ratio": round(ratio, 3), "contrast": None if np.isnan(contrast) else round(contrast, 3),
        "smallest_lesion_axis_voxels": round(float((np.asarray(ellipsoids[0]["semi_axes"]) * 2
                                                    / np.array(spacing_xyz)).min()), 2),
        "n_anomalies": len(case_log.record["anomalies"]),
        "registration": {c["name"]: next((s for s in case_log.record["steps"]
                                          if s["step"] == "register" and s["channel"] == c["name"]), {})
                         for c in channels if c["name"] != reference["name"]},
        "channels": stats,
    }
    _dump_json(marker, {"params_hash": params_hash, "spacing_mm": spacing_mm, "summary": summary})
    registry.clear_exclusion(out_dir, pid)
    case_log.close("ok", summary=summary)
    return "ok"


# ------------------------------------------------------------------------ the run

def held_cases(out_dir, settings=None):
    """Patients fully built: native NIfTI for every ingested phase **and** every case output.

    This is what the bronze purge asks before it deletes a patient's DICOM: not "a manifest
    says so" but "the files are here". ``settings`` defaults to the shipped ``config.yaml``.
    """
    settings = settings or settings_module.load()
    held = set()
    cases_dir = os.path.join(out_dir, "cases")
    if not os.path.isdir(cases_dir):
        return held
    for pid in sorted(os.listdir(cases_dir)):
        native, _ = _paths(out_dir, pid)
        try:
            phases = _load_json(os.path.join(native, "ingest.json"))["phases"]
            _load_json(os.path.join(cases_dir, pid, "case.json"))
        except (OSError, KeyError, ValueError):
            continue
        native_files = [os.path.join(native, f"{pid}_ph{p}.nii.gz") for p in phases]
        if all(os.path.getsize(f) > 0 for f in native_files if os.path.exists(f)) \
                and all(os.path.exists(f) for f in native_files + _outputs(settings, out_dir, pid)):
            held.add(pid)
    return held


def build(raw_dir, boxes_path, out_dir, settings=None, patients=None, limit=None, force=False,
          ingest_only=False):
    """Ingest then process every patient; returns ``{"ok", "skipped", "excluded"}`` counts per stage.

    ``ingest_only`` stops after the lossless native copy: what has to exist before bronze can
    be purged, and all that needs bronze.

    ``patients`` restricts the run, ``limit`` caps it (both for trying the pipeline on a few
    real cases before committing hours to all of them). A patient already ingested whose bronze
    folders are gone is processed from ``native/`` alone.
    """
    settings = settings or settings_module.load()
    os.makedirs(out_dir, exist_ok=True)
    groups = discover(raw_dir)
    boxes = read_boxes(boxes_path) if os.path.exists(boxes_path) else {}
    native_root = os.path.join(out_dir, "native")
    on_disk = set(os.listdir(native_root)) if os.path.isdir(native_root) else set()
    everyone = sorted(set(groups) | on_disk)
    if patients:
        everyone = [p for p in everyone if p in set(patients)]
    if limit:
        everyone = everyone[:limit]

    counts = {"ingest": {"ok": 0, "skipped": 0, "excluded": 0},
              "process": {"ok": 0, "skipped": 0, "excluded": 0}}
    for pid in everyone:
        if pid in groups:
            try:
                counts["ingest"][ingest_case(pid, groups[pid], boxes, settings, out_dir, force)] += 1
            except Exception as exc:
                registry.exclude(out_dir, pid, "ingest", "ingest_error", f"{type(exc).__name__}: {exc}")
                counts["ingest"]["excluded"] += 1
                log.error("[nnunet] %s ingest failed:\n%s", pid, traceback.format_exc())
    if ingest_only:
        return counts
    spacing_mm = spacing_module.resolve(out_dir, settings)
    log.info("[nnunet] target spacing: %.3f mm", spacing_mm)

    for pid in everyone:
        if not os.path.exists(os.path.join(native_root, pid, "ingest.json")):
            continue
        try:
            counts["process"][process_case(pid, settings, out_dir, spacing_mm, force)] += 1
        except Exception as exc:
            registry.exclude(out_dir, pid, "process", "processing_error", f"{type(exc).__name__}: {exc}")
            counts["process"]["excluded"] += 1
            log.error("[nnunet] %s processing failed:\n%s", pid, traceback.format_exc())

    write_manifest(out_dir, raw_dir, settings, spacing_mm)
    return counts


def write_manifest(out_dir, raw_dir, settings, spacing_mm):
    """The lineage manifest of the corpus: parameters, and one summary per built case."""
    cases = {}
    cases_dir = os.path.join(out_dir, "cases")
    for pid in sorted(held_cases(out_dir, settings)):
        record = _load_json(os.path.join(cases_dir, pid, "case.json"))
        cases[pid] = {**record["summary"], "case_id": pid}
    warnings = []
    for line in _read_jsonl(os.path.join(out_dir, "log", "cases.jsonl")):
        if line.get("stage") == "process" and line.get("patient_id") in cases:
            warnings += [f"{line['patient_id']}: {a}" for a in line.get("anomalies", [])]
    return lineage.write_manifest(
        out_dir, source=raw_dir,
        parameters={"pipeline": "mri_nnunet.pipeline.build", "spacing_mm": spacing_mm,
                    "settings": settings,
                    "excluded": len(registry.read_csv(os.path.join(out_dir, "exclusions.csv")))},
        cases=cases, warnings=sorted(set(warnings)))


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
