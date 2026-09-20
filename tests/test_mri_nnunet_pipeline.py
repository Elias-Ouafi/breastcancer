"""The SimpleITK side of the nnU-Net MRI corpus: ingestion, geometry, N4, registration, a whole case.

Skipped when SimpleITK (the ``nnunet`` extra) is absent, which is the case in CI. The synthetic
DICOM series are real DICOM files, written slice by slice with ``InstanceNumber`` *decreasing*
along z -- the layout measured on the Duke series (144 -> 1 along increasing z) that makes the
published slice numbers run against the spatial order.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK", reason="nnunet extra not installed")
pytest.importorskip("yaml", reason="nnunet extra not installed")

from mri_nnunet import export, pipeline, registry, sitk_io, steps  # noqa: E402
from mri_nnunet import settings as settings_module  # noqa: E402

RAS = sitk_io.RAS_DIRECTION


@pytest.fixture
def settings():
    s = settings_module.load()
    s = copy.deepcopy(s)
    s["ingest"]["phases"] = [0, 2]
    s["n4"].update(shrink_factor=2, iterations=[8])
    s["registration"].update(shrink_factors=[2, 1], smoothing_sigmas=[1, 0], iterations=30)
    s["resample"]["spacing_mm"] = 2.0
    s["crop"]["margin_voxels"] = 2
    return s


# ------------------------------------------------------------------------ synthetic data

def _blob_volume(shape=(40, 56, 56), lesion=True, bias=True, seed=0):
    """A tissue ellipsoid in a dark background, a bias field over it, and one bright lesion."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape).astype(np.float32)
    cz, cy, cx = (s / 2 for s in shape)
    tissue = (((z - cz) / 17) ** 2 + ((y - cy) / 22) ** 2 + ((x - cx) / 22) ** 2) <= 1
    field = (1.0 + 0.35 * (x / shape[2] - 0.5)) if bias else 1.0
    base = np.where(tissue, 300.0 * field, 8.0) + rng.normal(0, 6, shape)
    lesion_center = (cz, cy + 4, cx - 5)
    lesion_mask = ((z - lesion_center[0]) ** 2 + (y - lesion_center[1]) ** 2 + (x - lesion_center[2]) ** 2) <= 4.5 ** 2
    post = base + np.where(lesion_mask, 220.0, 0.0) + np.where(tissue, 15.0, 0.0)
    return base.astype(np.float32), post.astype(np.float32), lesion_center


def _image(array, spacing=(1.5, 1.5, 1.5), direction=RAS, origin=(0.0, 0.0, 0.0)):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def _write_dicom_series(folder, array, spacing, instance_of_z, description, patient="Breast_MRI_900"):
    """One real DICOM file per slice; ``instance_of_z[z]`` is the InstanceNumber at height z."""
    os.makedirs(folder, exist_ok=True)
    image = sitk.GetImageFromArray(np.clip(array, 0, 30000).astype(np.int16))
    image.SetSpacing(spacing)
    series_uid = "1.2.826.0.1.3680043.8.498." + str(abs(hash((folder, description))) % 10 ** 12)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()   # otherwise ITK gives every slice its own series UID
    for z in range(image.GetSize()[2]):
        piece = image[:, :, z]
        position = image.TransformIndexToPhysicalPoint((0, 0, z))
        tags = {"0008|0060": "MR", "0020|000e": series_uid, "0020|000d": "1.2.3.4.5", "0008|103e": description,
                "0010|0020": patient, "0008|0070": "SIEMENS", "0008|1090": "Avanto", "0018|0087": "1.5",
                "0008|0020": "19900101", "0020|0013": str(int(instance_of_z[z])),
                "0008|0016": "1.2.840.10008.5.1.4.1.1.4", "0008|0018": f"{series_uid}.{z}",
                "0020|0032": "\\".join(f"{p:.6f}" for p in position), "0020|0037": "1\\0\\0\\0\\1\\0",
                "0018|0050": f"{spacing[2]}", "0028|0030": f"{spacing[1]}\\{spacing[0]}"}
        for key, value in tags.items():
            piece.SetMetaData(key, value)
        writer.SetFileName(os.path.join(folder, f"{z:03d}.dcm"))
        writer.Execute(piece)


def _dicom_case(tmp_path, n_slices=14, patient="Breast_MRI_900"):
    pre, post, _ = _blob_volume(shape=(n_slices, 40, 40))
    order = np.arange(n_slices, 0, -1)                       # InstanceNumber n..1 along z, as in Duke
    folders = {}
    for phase, array, desc in ((0, pre, "ax dyn pre"), (2, post, "ax dyn 2nd pass")):
        folder = str(tmp_path / f"ph{phase}")
        _write_dicom_series(folder, array, (1.0, 1.0, 1.2), order, desc, patient)
        folders[phase] = folder
    return folders


BOX = {"Breast_MRI_900": [{"cols": (10, 20), "rows": (10, 20), "slices": (1, 3)}]}


# --------------------------------------------------------------------------- ingestion

def test_the_published_slice_numbers_are_read_as_instance_numbers_not_as_heights(tmp_path, settings):
    """Slices 1..3 by InstanceNumber are the *top* of the stack when InstanceNumber runs down."""
    folders = _dicom_case(tmp_path)
    assert pipeline.ingest_case("Breast_MRI_900", folders, BOX, settings, str(tmp_path / "silver")) == "ok"

    box = json.load(open(tmp_path / "silver" / "native" / "Breast_MRI_900" / "box.json"))
    assert box["boxes"][0]["z_indices"] == [13, 12, 11], box["boxes"][0]["z_indices"]


def test_the_position_basis_reads_the_same_numbers_as_heights(tmp_path, settings):
    settings["pseudo_mask"]["slice_basis"] = "position"
    folders = _dicom_case(tmp_path)
    pipeline.ingest_case("Breast_MRI_900", folders, BOX, settings, str(tmp_path / "silver"))
    box = json.load(open(tmp_path / "silver" / "native" / "Breast_MRI_900" / "box.json"))
    assert box["boxes"][0]["z_indices"] == [0, 1, 2]


def test_the_native_nifti_is_canonical_ras_and_the_registry_holds_the_scanner(tmp_path, settings):
    nib = pytest.importorskip("nibabel")
    folders = _dicom_case(tmp_path)
    out = str(tmp_path / "silver")
    pipeline.ingest_case("Breast_MRI_900", folders, BOX, settings, out)

    for phase in (0, 2):
        img = nib.load(os.path.join(out, "native", "Breast_MRI_900", f"Breast_MRI_900_ph{phase}.nii.gz"))
        assert nib.aff2axcodes(img.affine) == ("R", "A", "S")
    (row,) = registry.read_csv(os.path.join(out, "registry.csv"))
    assert row["patient_id"] == "Breast_MRI_900"
    assert row["scanner"] == "SIEMENS Avanto" and row["field_strength_t"] == "1.5"
    assert row["site"] == "" and row["study_date"] == "19900101"
    assert row["phases_available"] == "0,2" and row["n_slices"] == "14"


def test_a_native_volume_is_lossless(tmp_path, settings):
    """The NIfTI copy is what replaces bronze: same pixels, same physical geometry."""
    folders = _dicom_case(tmp_path)
    out = str(tmp_path / "silver")
    pipeline.ingest_case("Breast_MRI_900", folders, BOX, settings, out)

    original, _ = sitk_io.read_series(folders[0])
    saved = sitk_io.read_nifti(os.path.join(out, "native", "Breast_MRI_900", "Breast_MRI_900_ph0.nii.gz"))
    assert saved.GetPixelID() == original.GetPixelID()
    assert sorted(sitk.GetArrayFromImage(saved).ravel().tolist()) == sorted(sitk.GetArrayFromImage(original).ravel().tolist())
    # the same physical point is the same voxel value
    point = original.TransformIndexToPhysicalPoint((20, 20, 7))
    assert saved.GetPixel(saved.TransformPhysicalPointToIndex(point)) == original.GetPixel((20, 20, 7))


def test_a_missing_channel_phase_is_an_exclusion_with_a_reason_not_a_crash(tmp_path, settings):
    folders = _dicom_case(tmp_path)
    out = str(tmp_path / "silver")
    assert pipeline.ingest_case("Breast_MRI_900", {0: folders[0]}, BOX, settings, out) == "excluded"
    (row,) = registry.read_csv(os.path.join(out, "exclusions.csv"))
    assert (row["patient_id"], row["stage"], row["reason"]) == ("Breast_MRI_900", "ingest", "missing_phase")


def test_a_series_with_a_missing_slice_is_excluded_as_incomplete(tmp_path, settings):
    folders = _dicom_case(tmp_path)
    os.remove(os.path.join(folders[2], "006.dcm"))
    out = str(tmp_path / "silver")
    assert pipeline.ingest_case("Breast_MRI_900", folders, BOX, settings, out) == "excluded"
    assert registry.read_csv(os.path.join(out, "exclusions.csv"))[0]["reason"] == "incomplete_series"


def test_a_corrupt_file_excludes_the_case_and_does_not_stop_the_run(tmp_path, settings):
    folders = _dicom_case(tmp_path)
    with open(os.path.join(folders[0], "005.dcm"), "wb") as handle:
        handle.write(b"this is not a DICOM file")
    out = str(tmp_path / "silver")
    assert pipeline.ingest_case("Breast_MRI_900", folders, BOX, settings, out) == "excluded"
    assert registry.read_csv(os.path.join(out, "exclusions.csv"))[0]["reason"] == "unreadable_series"


def test_a_patient_without_a_box_is_excluded(tmp_path, settings):
    folders = _dicom_case(tmp_path)
    out = str(tmp_path / "silver")
    assert pipeline.ingest_case("Breast_MRI_900", folders, {}, settings, out) == "excluded"
    assert registry.read_csv(os.path.join(out, "exclusions.csv"))[0]["reason"] == "no_box"


def test_ingestion_is_idempotent_until_its_parameters_change(tmp_path, settings):
    folders = _dicom_case(tmp_path)
    out = str(tmp_path / "silver")
    assert pipeline.ingest_case("Breast_MRI_900", folders, BOX, settings, out) == "ok"
    assert pipeline.ingest_case("Breast_MRI_900", folders, BOX, settings, out) == "skipped"
    changed = copy.deepcopy(settings)
    changed["ingest"]["max_slice_spacing_deviation"] = 0.5
    assert pipeline.ingest_case("Breast_MRI_900", folders, BOX, changed, out) == "ok"


# ---------------------------------------------------------------------------- steps

def test_slice_spacing_deviation_flags_a_missing_slice():
    complete = np.arange(0, 20, 1.2)
    assert sitk_io.slice_spacing_deviation(complete) == pytest.approx(0.0, abs=1e-6)
    assert sitk_io.slice_spacing_deviation(np.delete(complete, 5)) == pytest.approx(1.0, abs=1e-6)


def test_n4_flattens_a_bias_field():
    pre, _, _ = _blob_volume()
    biased = _image(pre)
    corrected, _ = sitk_io.n4_correct(biased, {"shrink_factor": 2, "iterations": [20], "convergence_threshold": 1e-3,
                                               "use_foreground_mask": True})
    z, y, x = np.indices(pre.shape)
    core = ((((z - 20) / 12) ** 2 + ((y - 28) / 16) ** 2 + ((x - 28) / 16) ** 2) <= 1)

    def cv(array):
        return float(array[core].std() / array[core].mean())

    assert cv(sitk.GetArrayFromImage(corrected)) < 0.75 * cv(pre)


def test_the_body_mask_keeps_the_organ_and_drops_a_stray_blob():
    pre, _, _ = _blob_volume(bias=False)
    pre[2:5, 2:5, 2:5] = 300.0                                     # an island of signal far from the organ
    mask, info = sitk_io.body_mask(_image(pre), settings_module.load()["body_mask"])
    array = sitk.GetArrayFromImage(mask) > 0
    z, y, x = np.indices(pre.shape)
    organ = ((((z - 20) / 15) ** 2 + ((y - 28) / 20) ** 2 + ((x - 28) / 20) ** 2) <= 1)
    assert (array & organ).sum() / organ.sum() > 0.95
    assert not array[2:5, 2:5, 2:5].any(), "the stray island survived the largest-component step"
    assert info["voxels"] == int(array.sum())


def test_rigid_registration_recovers_a_known_shift():
    cfg = copy.deepcopy(settings_module.load()["registration"])
    cfg.update(shrink_factors=[2, 1], smoothing_sigmas=[1, 0], iterations=80)
    pre, _, _ = _blob_volume(bias=False)
    fixed = _image(pre)
    moving = sitk.Resample(fixed, fixed, sitk.TranslationTransform(3, (-4.5, 0.0, 0.0)), sitk.sitkLinear, 0.0)
    mask = sitk.Cast(fixed > 100, sitk.sitkUInt8)

    transform, info = sitk_io.register_rigid(fixed, moving, mask, cfg, seed=1)

    assert info["accepted"] and info["translation_mm"] == pytest.approx(4.5, abs=1.5), info


def test_a_registration_out_of_bounds_is_refused_and_leaves_the_identity():
    cfg = copy.deepcopy(settings_module.load()["registration"])
    cfg.update(shrink_factors=[2, 1], smoothing_sigmas=[1, 0], iterations=80, max_translation_mm=0.5)
    pre, _, _ = _blob_volume(bias=False)
    fixed = _image(pre)
    moving = sitk.Resample(fixed, fixed, sitk.TranslationTransform(3, (-4.5, 0.0, 0.0)), sitk.sitkLinear, 0.0)

    transform, info = sitk_io.register_rigid(fixed, moving, sitk.Cast(fixed > 100, sitk.sitkUInt8), cfg, seed=1)

    assert info["accepted"] is False and "out of bounds" in info["reason"]
    assert transform.GetName() == "IdentityTransform" or not any(transform.GetParameters())


def test_the_target_grid_keeps_an_oblique_flipped_acquisition_in_place():
    """A tilted, flipped image resampled onto the RAS grid: the bright cube stays at the same place."""
    array = np.zeros((30, 40, 40), np.float32)
    array[12:18, 15:25, 15:25] = 100.0
    tilt = np.radians(3.0)
    c, s = np.cos(tilt), np.sin(tilt)
    image = _image(array, spacing=(1.2, 1.2, 1.5), direction=(-c, 0, -s, 0, -1, 0, -s, 0, c), origin=(30.0, 40.0, -20.0))
    grid = sitk_io.target_grid(image, (1.0, 1.0, 1.0))
    out = sitk_io.resample(image, grid)

    def centroid(img):
        arr = sitk.GetArrayFromImage(img)
        idx = np.argwhere(arr > 50).mean(axis=0)[::-1]
        return np.array(img.TransformContinuousIndexToPhysicalPoint(tuple(float(i) for i in idx)))

    assert np.allclose(centroid(out), centroid(image), atol=1.5)
    assert out.GetDirection() == RAS


def test_labels_resampled_by_nearest_neighbour_stay_binary():
    array = np.zeros((20, 30, 30), np.uint8)
    array[5:15, 8:22, 8:22] = 1
    label = _image(array.astype(np.uint8))
    label = sitk.Cast(label, sitk.sitkUInt8)
    grid = sitk_io.target_grid(label, (1.0, 1.0, 1.0))
    values = np.unique(sitk.GetArrayFromImage(sitk_io.resample_labels(label, grid)))
    assert values.tolist() == [0, 1]


# --------------------------------------------------------------------------- a whole case

def _native_case(tmp_path, settings, patient="P1"):
    """A native corpus with one synthetic case: two phases as NIfTI, and its box.json."""
    out = str(tmp_path / "silver")
    native = os.path.join(out, "native", patient)
    pre, post, lesion_center = _blob_volume()
    spacing = (1.5, 1.5, 1.5)
    for phase, array in ((0, pre), (2, post)):
        sitk_io.write_nifti(_image(array, spacing), os.path.join(native, f"{patient}_ph{phase}.nii.gz"))
    reference = _image(pre, spacing)
    cz, cy, cx = lesion_center
    ellipsoid = steps.box_to_ellipsoid((int(cx) - 5, int(cx) + 5), (int(cy) - 5, int(cy) + 5),
                                       list(range(int(cz) - 5, int(cz) + 5)), *sitk_io.geometry(reference))
    with open(os.path.join(native, "box.json"), "w") as handle:
        json.dump({"reference_phase": 0, "slice_basis": "instance_number", "native_spacing": list(spacing),
                   "native_size": list(reference.GetSize()),
                   "boxes": [{"published": {}, "z_indices": [],
                              "ellipsoid": {k: np.asarray(v).tolist() for k, v in ellipsoid.items()}}]}, handle)
    with open(os.path.join(native, "ingest.json"), "w") as handle:
        json.dump({"params_hash": "x", "phases": [0, 2]}, handle)
    return out, lesion_center


def test_a_whole_case_is_normalised_inside_its_mask_and_the_label_lands_on_the_lesion(tmp_path, settings):
    out, (cz, cy, cx) = _native_case(tmp_path, settings)

    assert pipeline.process_case("P1", settings, out, spacing_mm=2.0) == "ok"

    outputs = pipeline._outputs(settings, out, "P1")
    organ = sitk.GetArrayFromImage(sitk.ReadImage(outputs[-1])) > 0
    label = sitk.GetArrayFromImage(sitk.ReadImage(outputs[-2]))
    for path in outputs[:2]:
        volume = sitk.GetArrayFromImage(sitk.ReadImage(path))
        assert volume[organ].mean() == pytest.approx(0.0, abs=1e-3)
        assert volume[organ].std() == pytest.approx(1.0, abs=1e-2)
        assert (volume[~organ] == 0).all()
    assert set(np.unique(label)) <= {0, 1} and label.sum() > 0
    assert (label[~organ] == 0).all(), "a pseudo-mask voxel lies outside the organ mask"

    # the label sits on the lesion: the post-contrast channel is brighter than the pre there
    pre, post = (sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in outputs[:2])
    assert (post - pre)[label > 0].mean() > (post - pre)[organ & (label == 0)].mean() + 0.5


def test_the_processing_order_is_the_documented_one(tmp_path, settings):
    out, _ = _native_case(tmp_path, settings)
    pipeline.process_case("P1", settings, out, spacing_mm=2.0)
    line = [json.loads(x) for x in open(os.path.join(out, "log", "cases.jsonl"))][-1]
    names = [s["step"] for s in line["steps"]]
    assert names == ["n4", "n4", "body_mask", "register", "resample", "normalize", "crop"]
    assert line["status"] == "ok" and all("seconds" in s for s in line["steps"])


def test_a_case_is_skipped_until_a_parameter_it_depends_on_changes(tmp_path, settings):
    out, _ = _native_case(tmp_path, settings)
    assert pipeline.process_case("P1", settings, out, spacing_mm=2.0) == "ok"
    assert pipeline.process_case("P1", settings, out, spacing_mm=2.0) == "skipped"
    # the spacing is part of what a case depends on...
    assert pipeline.process_case("P1", settings, out, spacing_mm=2.5) == "ok"
    # ...and so is any section it reads
    changed = copy.deepcopy(settings)
    changed["normalize"]["clip_percentiles"] = [1.0, 99.0]
    assert pipeline.process_case("P1", changed, out, spacing_mm=2.5) == "ok"


def test_a_lesion_the_organ_crop_would_cut_off_excludes_the_case(tmp_path, settings):
    out, _ = _native_case(tmp_path, settings)
    with open(os.path.join(out, "native", "P1", "box.json")) as handle:
        box = json.load(handle)
    box["boxes"][0]["ellipsoid"]["center"] = [400.0, 400.0, 400.0]      # nowhere near the organ
    with open(os.path.join(out, "native", "P1", "box.json"), "w") as handle:
        json.dump(box, handle)

    assert pipeline.process_case("P1", settings, out, spacing_mm=2.0) == "excluded"
    assert registry.read_csv(os.path.join(out, "exclusions.csv"))[0]["reason"] == "empty_pseudo_mask"


def test_held_cases_are_the_ones_whose_files_are_all_there(tmp_path, settings):
    out, _ = _native_case(tmp_path, settings)
    assert pipeline.held_cases(out, settings) == set()
    pipeline.process_case("P1", settings, out, spacing_mm=2.0)
    assert pipeline.held_cases(out, settings) == {"P1"}
    os.remove(pipeline._outputs(settings, out, "P1")[0])                    # a missing output: no longer held
    assert pipeline.held_cases(out, settings) == set()


def test_the_export_is_a_valid_nnunet_v2_dataset(tmp_path, settings):
    out, _ = _native_case(tmp_path, settings)
    pipeline.process_case("P1", settings, out, spacing_mm=2.0)

    folder, cases = export.export(out, str(tmp_path / "nnUNet_raw"), settings)

    assert cases == ["P1"] and os.path.basename(folder) == "Dataset501_DukeDCEBreast"
    assert sorted(os.listdir(os.path.join(folder, "imagesTr"))) == ["P1_0000.nii.gz", "P1_0001.nii.gz"]
    assert os.listdir(os.path.join(folder, "labelsTr")) == ["P1.nii.gz"]
    meta = json.load(open(os.path.join(folder, "dataset.json")))
    assert meta["channel_names"] == {"0": "noNorm", "1": "noNorm"}
    assert meta["labels"] == {"background": 0, "lesion": 1}
    assert meta["numTraining"] == 1 and meta["file_ending"] == ".nii.gz"
    # channels and label share one grid, which nnU-Net checks
    grids = [sitk.ReadImage(os.path.join(folder, "imagesTr", f"P1_000{i}.nii.gz")) for i in (0, 1)]
    grids.append(sitk.ReadImage(os.path.join(folder, "labelsTr", "P1.nii.gz")))
    assert len({(g.GetSize(), g.GetSpacing(), g.GetOrigin()) for g in grids}) == 1
