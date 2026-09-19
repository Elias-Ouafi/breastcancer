"""Preparing a DCE-MRI exam nobody has annotated yet.

Everything the served model reads was built by ``preprocess_dce_mri_with_boxes``,
which skips any patient absent from the annotation table. A new exam is exactly such
a patient, so until ``preprocess_dce_mri_exams`` existed there was no way to take one
from DICOM to a volume the app accepts.

The load-bearing test here is
``test_an_unannotated_exam_gets_the_same_volume_as_the_annotated_path``: a new exam is
only meaningful to the checkpoint if it lands on the same intensity scale and geometry
as the corpus that trained it. The rest guard the ways an exam can be incomplete.

Fixtures are synthetic single-slice DICOM series -- no dataset, no network. The one
exception is ``test_real_dicom_reproduces_the_corpus_volume_bit_for_bit``, which runs
only on a machine that has both the raw DCE-MRI layer and the built corpus, and is the
strongest check available: real DICOM in, the corpus's own volume out.
"""
import os

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import TransformData as T

# 128 px is validation.py's floor for a full-frame study: below it, a volume is taken
# to be a lesion crop and warned about. A fixture under that floor would test the
# pipeline on a geometry the real one rejects.
DEPTH, ROWS, COLS = 6, 128, 128
# The enhancing region, and it has to be this big on purpose: `normalize_intensity`
# clips at the 99th percentile, so anything occupying less than 1% of the volume is
# flattened to the ceiling and stops being the brightest voxel. 2*48*48 of 6*128*128
# is 4.7%. See test_a_lesion_below_the_99th_percentile_is_clipped_to_the_ceiling.
LESION = (slice(2, 4), slice(40, 88), slice(40, 88))


def write_series(folder, patient_id, description, depth=DEPTH, base=100, lesion=0,
                 rows=ROWS, cols=COLS, seed=0):
    """Write one DCE series folder: `depth` single-slice .dcm files.

    `lesion` is added inside LESION, which is how a post-contrast pass is made to
    enhance relative to its pre-contrast baseline.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    volume = rng.integers(base, base + 20, size=(depth, rows, cols)).astype(np.uint16)
    if lesion:
        volume[LESION] += lesion

    for i in range(depth):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.ImplementationClassUID = generate_uid()

        path = str(folder / f"{i:03d}.dcm")
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.PatientID = patient_id
        ds.SeriesDescription = description
        # Deliberately written out of order on disk: the loader must sort on
        # InstanceNumber, not on filename.
        ds.InstanceNumber = i + 1
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = volume[i].tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(path, write_like_original=False)
    return folder


def write_exam(root, patient_id="Breast_MRI_900", post_rank=2, **kwargs):
    """Write a complete exam: pre-contrast plus the `post_rank` post-contrast pass.

    The two phases get different seeds. Sharing one would make ``post - pre`` exactly
    zero outside the lesion, which no real pair of acquisitions is.
    """
    write_series(root / f"{patient_id}_pre", patient_id, "ax dyn pre", seed=1, **kwargs)
    write_series(root / f"{patient_id}_post", patient_id,
                 f"ax dyn {['', '1st', '2nd', '3rd', '4th'][post_rank]} pass",
                 lesion=400, seed=2, **kwargs)
    return root


def write_boxes(path, patient_id="Breast_MRI_900"):
    """The TCIA annotation table for `patient_id`, 1-indexed inclusive like the real one."""
    pd.DataFrame([{
        "Patient ID": patient_id,
        "Start Row": LESION[1].start + 1, "End Row": LESION[1].stop,
        "Start Column": LESION[2].start + 1, "End Column": LESION[2].stop,
        "Start Slice": LESION[0].start + 1, "End Slice": LESION[0].stop,
    }]).to_csv(path, index=False)
    return str(path)


def volume_of(npz_path):
    with np.load(npz_path) as data:
        return data["volume"]


# --- the invariant that makes a new exam usable at all ----------------------


def test_an_unannotated_exam_gets_the_same_volume_as_the_annotated_path(tmp_path):
    """A new exam must reach the model on the scale the corpus trained it on.

    Both paths are given the same two series. The annotated one additionally paints a
    mask, which must not change a single voxel of the volume -- otherwise a new exam
    is scored on a distribution the checkpoint has never seen, and no amount of
    threshold tuning would fix it.
    """
    raw = write_exam(tmp_path / "raw")
    boxes = write_boxes(tmp_path / "boxes.csv")

    annotated_dir = tmp_path / "annotated"
    unannotated_dir = tmp_path / "unannotated"
    T.preprocess_dce_mri_with_boxes(str(raw), boxes, output_dir=str(annotated_dir),
                                    post_phase_rank=2, crop=False)
    T.preprocess_dce_mri_exams(str(raw), output_dir=str(unannotated_dir),
                               post_phase_rank=2)

    annotated = volume_of(annotated_dir / "Breast_MRI_900.npz")
    unannotated = volume_of(unannotated_dir / "Breast_MRI_900.npz")
    np.testing.assert_array_equal(annotated, unannotated)


def test_the_annotated_path_skips_an_exam_the_new_path_prepares(tmp_path):
    """The gap this module closes, pinned so it cannot silently come back."""
    raw = write_exam(tmp_path / "raw")
    # An annotation table that knows every patient except this one.
    boxes = write_boxes(tmp_path / "boxes.csv", patient_id="Breast_MRI_001")

    saved_annotated, skipped_annotated = T.preprocess_dce_mri_with_boxes(
        str(raw), boxes, output_dir=str(tmp_path / "annotated"), crop=False)
    saved_new, skipped_new = T.preprocess_dce_mri_exams(
        str(raw), output_dir=str(tmp_path / "new"))

    assert (saved_annotated, skipped_annotated) == (0, 1)
    assert (saved_new, skipped_new) == (1, 0)


# --- the contract the app and inference read --------------------------------


def test_the_written_npz_satisfies_the_inference_contract(tmp_path):
    """``inference._load_volume_and_offset`` must accept the file unchanged."""
    inference = pytest.importorskip("inference", reason="inference needs torch")

    raw = write_exam(tmp_path / "raw")
    T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    path = str(tmp_path / "out" / "Breast_MRI_900.npz")
    vol, crop_offset, forced_slice, source_n_slices = inference._load_volume_and_offset(path)

    assert vol.shape == (DEPTH, ROWS, COLS)
    # No annotation means no region of interest, so nothing may be cropped away.
    assert crop_offset == (0, 0, 0)
    # A new exam pins no slice: the app has to choose one, and must say that it did.
    assert forced_slice is None
    assert source_n_slices is None


def test_the_frame_is_kept_whole(tmp_path):
    """Full frame is the geometry the served checkpoint was trained on."""
    raw = write_exam(tmp_path / "raw")
    T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    with np.load(tmp_path / "out" / "Breast_MRI_900.npz") as data:
        assert data["volume"].shape == (DEPTH, ROWS, COLS)
        assert tuple(data["crop_offset"]) == (0, 0, 0)
        assert data["mask"].shape == data["volume"].shape
        assert data["mask"].sum() == 0
        assert str(data["case_id"]) == "Breast_MRI_900"


def test_no_label_is_invented_for_an_unread_exam(tmp_path):
    """Nobody has read this exam, so the file must not claim a status or a label."""
    raw = write_exam(tmp_path / "raw")
    T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    with np.load(tmp_path / "out" / "Breast_MRI_900.npz") as data:
        assert "label" not in data.files
        assert "exam_status" not in data.files
        assert "lesion_class" not in data.files


# --- the subtraction itself -------------------------------------------------


def test_subtraction_keeps_only_enhancement():
    """Washout (post below pre) carries no lesion signal, so it is clipped at 0."""
    pre = np.full((2, 4, 4), 100.0)
    post = pre.copy()
    post[0, 0, 0] = 300.0   # enhancement
    post[1, 0, 0] = 10.0    # washout

    raw = np.clip(post - pre, a_min=0, a_max=None)
    assert raw[1, 0, 0] == 0

    out = T.dce_subtraction(pre, post)
    assert out.shape == pre.shape
    assert np.isfinite(out).all()
    # The enhancing voxel must stay the brightest one.
    assert out[0, 0, 0] == out.max()


def test_a_lesion_below_the_99th_percentile_is_clipped_to_the_ceiling():
    """Percentile clipping is a window, and a small lesion sits outside it.

    `normalize_intensity` clips at the 99th percentile before standardising, so a
    lesion occupying less than 1% of the volume is pulled down to the same ceiling as
    the brightest 1% of everything else. It stays saturated rather than disappearing,
    but it stops being uniquely the maximum -- which is why a synthetic fixture needs
    a lesion above 1% for `argmax` to mean anything, and why the corpus statistics in
    DOCUMENTATION.md section 4.9 are computed per exam rather than pooled.
    """
    rng = np.random.default_rng(0)
    pre = rng.uniform(100, 120, size=(6, 128, 128))
    post = pre + rng.uniform(0, 20, size=(6, 128, 128))
    tiny = (slice(2, 4), slice(10, 18), slice(10, 18))   # 128 voxels, 0.13%
    post[tiny] += 400

    out = T.dce_subtraction(pre, post)

    # Saturated, not preserved: the lesion shares the maximum with ordinary voxels.
    assert out[tiny].max() == pytest.approx(out.max())
    assert (out == out.max()).sum() > 2 * out[tiny].size


def test_subtraction_is_z_normalised():
    """The corpus convention: z-normalised after clipping, which is what the model reads."""
    rng = np.random.default_rng(0)
    pre = rng.uniform(50, 150, size=(4, 16, 16))
    post = pre + rng.uniform(0, 50, size=(4, 16, 16))

    out = T.dce_subtraction(pre, post)
    assert out.mean() == pytest.approx(0.0, abs=1e-6)
    assert out.std() == pytest.approx(1.0, abs=1e-3)


# --- incomplete or inconsistent exams ---------------------------------------


def test_an_exam_missing_its_post_contrast_pass_is_skipped_not_crashed(tmp_path):
    raw = tmp_path / "raw"
    write_series(raw / "pre", "Breast_MRI_900", "ax dyn pre")

    saved, skipped = T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    assert (saved, skipped) == (0, 1)


def test_an_exam_missing_its_pre_contrast_series_is_skipped(tmp_path):
    raw = tmp_path / "raw"
    write_series(raw / "post", "Breast_MRI_900", "ax dyn 2nd pass", lesion=400)

    saved, skipped = T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    assert (saved, skipped) == (0, 1)


def test_phases_of_different_shapes_are_skipped(tmp_path):
    """A subtraction between mismatched acquisitions would be meaningless, not merely wrong."""
    raw = tmp_path / "raw"
    write_series(raw / "pre", "Breast_MRI_900", "ax dyn pre", rows=128, cols=128)
    write_series(raw / "post", "Breast_MRI_900", "ax dyn 2nd pass", rows=160, cols=160)

    saved, skipped = T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    assert (saved, skipped) == (0, 1)


def test_a_requested_pass_that_the_exam_does_not_have_is_skipped(tmp_path):
    """Asking for the 4th pass of an exam that stops at the 2nd must not fall back."""
    raw = write_exam(tmp_path / "raw", post_rank=2)

    saved, skipped = T.preprocess_dce_mri_exams(
        str(raw), output_dir=str(tmp_path / "out"), post_phase_rank=4)

    assert (saved, skipped) == (0, 1)


def test_one_broken_exam_does_not_stop_the_others(tmp_path):
    """A batch of new exams is only useful if a single bad one cannot sink it."""
    raw = tmp_path / "raw"
    write_exam(raw, patient_id="Breast_MRI_900")
    write_series(raw / "901_pre", "Breast_MRI_901", "ax dyn pre")  # no post pass

    saved, skipped = T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    assert (saved, skipped) == (1, 1)
    assert (tmp_path / "out" / "Breast_MRI_900.npz").exists()
    assert not (tmp_path / "out" / "Breast_MRI_901.npz").exists()


# --- the two SeriesDescription conventions of the collection ----------------


@pytest.mark.parametrize("pre_desc, post_desc", [
    ("ax dyn pre", "ax dyn 2nd pass"),   # convention A: spelled-out pass
    ("ax 3d dyn", "Ph2/ax 3d dyn"),      # convention B: bare pre, Ph-prefixed passes
])
def test_both_series_naming_conventions_are_understood(tmp_path, pre_desc, post_desc):
    """Half the collection uses each convention, so missing one loses half the cohort."""
    raw = tmp_path / "raw"
    write_series(raw / "pre", "Breast_MRI_900", pre_desc)
    write_series(raw / "post", "Breast_MRI_900", post_desc, lesion=400)

    saved, skipped = T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    assert (saved, skipped) == (1, 0)


def test_series_outside_the_dynamic_protocol_are_ignored(tmp_path):
    """A T1 TSE sitting next to the dynamic series must not be mistaken for a phase."""
    raw = write_exam(tmp_path / "raw")
    write_series(raw / "t1", "Breast_MRI_900", "ax t1 tse +c")

    saved, skipped = T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    assert (saved, skipped) == (1, 0)


def test_slices_are_ordered_by_instance_number_not_filename(tmp_path):
    """Filename order and acquisition order differ; only InstanceNumber is authoritative."""
    raw = write_exam(tmp_path / "raw")
    T.preprocess_dce_mri_exams(str(raw), output_dir=str(tmp_path / "out"))

    volume = volume_of(tmp_path / "out" / "Breast_MRI_900.npz")
    # The enhancing blob was added to slices 2..3 of the post-contrast pass, so those
    # are the slices the subtraction must light up.
    per_slice = volume.astype(np.float32).sum(axis=(1, 2))
    assert per_slice.argmax() in (LESION[0].start, LESION[0].stop - 1)


# --- the app, end to end ----------------------------------------------------


def test_the_app_accepts_a_new_exam_and_says_it_chose_the_slice(tmp_path, monkeypatch):
    """The whole point: upload an exam nobody annotated and get an answer back.

    Run against the real checkpoint (versioned, so a fresh clone has it) but on a
    deliberately small volume, to keep the suite free of the dataset. What is pinned
    is not the verdict -- it is that the request succeeds and that the result admits
    the slice was picked by the classifier rather than handed over, since on a real
    upload that choice is right about 43% of the time (DOCUMENTATION.md section 4.3).
    """
    pytest.importorskip("torch", reason="the dce_mri backend needs torch")
    import config
    if not os.path.exists(config.DCE_MRI_UNET_CKPT):
        pytest.skip("U-Net checkpoint not present")

    from app import predictor as predictor_module
    from app.server import app

    monkeypatch.setenv("MRI_APP_BACKEND", "dce_mri")
    monkeypatch.setattr(predictor_module, "_PREDICTORS", {})

    path = tmp_path / "new_exam.npz"
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        volume=rng.normal(0, 1, size=(4, ROWS, COLS)).astype(np.float16),
        mask=np.zeros((4, ROWS, COLS), dtype=np.uint8),
        crop_offset=np.asarray((0, 0, 0), dtype=np.int32),
        case_id=np.asarray("NEW_PATIENT"),
    )

    with open(path, "rb") as fh:
        response = app.test_client().post(
            "/api/predict", data={"mri": (fh, "new_exam.npz")},
            content_type="multipart/form-data")

    assert response.status_code == 200
    body = response.get_json()
    assert body["backend"] == "dce_mri"
    assert body["n_slices"] == 4
    # No pinned slice, and the result has to say so rather than let a reader assume
    # a human chose it -- that conflation was the on-screen claim removed on 2026-09-12.
    assert body["slice_preselected"] is False
    assert body["slice_selector"] == "classifier"


# --- against the real collection, when it is on the machine -----------------


def test_real_dicom_reproduces_the_corpus_volume_bit_for_bit():
    """Real DICOM in, the corpus's own volume out -- no fixture can prove this.

    The corpus under ``dce_mri_p2/`` was written by the *annotated* path. Feeding the
    same patient's raw series to the *unannotated* path has to land on exactly the same
    array, or a new exam reaches the checkpoint on a distribution it never saw.

    Skipped unless both layers are present, so CI and a fresh clone stay green. Slow:
    it reads one DICOM header per series folder to find the patient.
    """
    import config

    raw_root = os.path.join(config.TCIA_DIR, "duke_mri")
    corpus = config.DCE_MRI_PREPROCESSED_DIR
    if not os.path.isdir(raw_root) or not os.path.isdir(corpus):
        pytest.skip("raw DCE-MRI layer or built corpus not on this machine")

    import pydicom

    # Pick a patient the corpus already holds, then find its series on disk. Reading
    # one header per folder is what `group_dce_series_by_patient` does anyway.
    built = {f[:-4] for f in os.listdir(corpus) if f.endswith(".npz")}
    if not built:
        pytest.skip("corpus is empty")

    phases = {}
    patient = None
    for name in sorted(os.listdir(raw_root)):
        folder = os.path.join(raw_root, name)
        if not os.path.isdir(folder):
            continue
        dcm = next((f for f in os.listdir(folder) if f.lower().endswith(".dcm")), None)
        if dcm is None:
            continue
        header = pydicom.dcmread(os.path.join(folder, dcm), stop_before_pixels=True)
        pid = str(getattr(header, "PatientID", ""))
        if pid not in built or (patient is not None and pid != patient):
            continue
        rank = T._dce_phase_rank(getattr(header, "SeriesDescription", ""))
        if rank in (0, 2):
            patient, phases[rank] = pid, folder
        if len(phases) == 2:
            break

    if len(phases) != 2:
        pytest.skip("no patient with both phases on disk and a volume in the corpus")

    pre = T._load_series_volume(phases[0])
    post = T._load_series_volume(phases[2])
    rebuilt = T.dce_subtraction(pre, post).astype(np.float16)

    with np.load(os.path.join(corpus, f"{patient}.npz")) as data:
        np.testing.assert_array_equal(rebuilt, data["volume"])
        # The corpus is full frame; a crop here would mean the two disagree on geometry.
        assert tuple(data["crop_offset"]) == (0, 0, 0)


# --- lineage ----------------------------------------------------------------


def test_a_manifest_records_how_the_exam_was_prepared(tmp_path):
    """A folder with no manifest is a folder whose run was interrupted."""
    import json

    raw = write_exam(tmp_path / "raw")
    out = tmp_path / "out"
    T.preprocess_dce_mri_exams(str(raw), output_dir=str(out), post_phase_rank=2)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["parameters"]
    assert params["pipeline"] == "preprocess_dce_mri_exams"
    assert params["post_phase_rank"] == 2
    assert params["crop"] is False
    assert params["annotated"] is False
    assert "Breast_MRI_900" in manifest["cases"]


def test_an_empty_mask_is_not_reported_as_a_warning(tmp_path):
    """An unread exam has no lesion by definition, so an empty mask is the expected
    result rather than the smell it would be in the annotated corpus."""
    import json

    raw = write_exam(tmp_path / "raw")
    out = tmp_path / "out"
    T.preprocess_dce_mri_exams(str(raw), output_dir=str(out))

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_warnings"] == []
