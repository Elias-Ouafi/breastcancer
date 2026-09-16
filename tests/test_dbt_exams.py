"""The exam-level corpus: what makes a cancer / no-cancer target measurable at all.

Everything the project had preprocessed came from the boxes CSVs, so every volume on
disk carried a lesion: prevalence 100 %, and a specificity that cannot be measured on a
corpus with no negatives (DOCUMENTATION.md, "Cible chiffrée"). ``preprocess_dbt_exams`` labels a
series from ``BCS-DBT-labels-*.csv`` instead, where a series with no box is a *negative*
rather than a skip.

Two properties matter more than the bookkeeping, and both are the kind that produce a
brilliant meaningless number when they break:

- **One geometry for both classes.** If positives were lesion crops and negatives full
  frames, array shape alone would separate them. So these tests check that a normal and
  a cancer exam come out of the same call with the same shape, and that a box follows
  the pixels through the resampling instead of staying at full-resolution coordinates.
- **One laterality convention, applied to negatives too.** The flip rule exists for
  annotated studies stored rotated relative to their boxes; a negative has no box to be
  rotated against. Applying it only where boxes exist would make "was flipped" a proxy
  for "has a label", which is a proxy for the answer.

Synthetic DICOMs again, at a small plane size, so the whole file runs in seconds.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import lineage
import validation
from TransformData import preprocess_dbt_exams, resize_in_plane

LABEL_COLUMNS = ["PatientID", "StudyUID", "View", "Normal", "Actionable", "Benign", "Cancer"]
BOX_COLUMNS = ["PatientID", "StudyUID", "View", "Subject", "Slice",
               "X", "Y", "Width", "Height", "Class", "AD"]
FILE_PATH_COLUMNS = ["PatientID", "StudyUID", "View", "descriptive_path", "classic_path"]

# The synthetic frame: wider than tall, like a real DBT frame, so padding is exercised.
ROWS, COLS, DEPTH = 128, 96, 6
BLOB = (slice(2, 5), slice(40, 60), slice(20, 40))   # (z, y, x) of the bright lesion
PLANE = 32


def study_of(patient):
    return f"S-{patient}"


def write_series(root, series_dir, patient_id, view="rmlo", laterality=None, blob=False):
    """One synthetic multi-frame series; ``blob`` paints a bright square at ``BLOB``."""
    folder = os.path.join(root, series_dir)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "1-1.dcm")

    rng = np.random.default_rng(len(patient_id) + len(series_dir))
    pixels = rng.integers(0, 60, size=(DEPTH, ROWS, COLS), dtype=np.uint16)
    side = (laterality or view[0]).upper()
    breast = slice(COLS // 2, COLS) if side == "R" else slice(0, COLS // 2)
    pixels[:, :, breast] = rng.integers(1500, 2500, size=(DEPTH, ROWS, COLS // 2),
                                        dtype=np.uint16)
    if blob:
        pixels[BLOB] = 4000

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.13.1.3"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.ImageLaterality = "L"
    ds.ViewPosition = view[1:].rstrip("0123456789").upper()
    ds.Rows, ds.Columns, ds.NumberOfFrames = ROWS, COLS, DEPTH
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(path, write_like_original=False)
    return folder


def write_labels(path, rows):
    """``rows`` are (patient, view, status); ``None`` writes a row with no flag set."""
    pd.DataFrame([
        {"PatientID": patient, "StudyUID": study_of(patient), "View": view,
         "Normal": int(status == "normal"), "Actionable": int(status == "actionable"),
         "Benign": int(status == "benign"), "Cancer": int(status == "cancer")}
        for patient, view, status in rows
    ], columns=LABEL_COLUMNS).to_csv(path, index=False)
    return str(path)


def write_file_paths(path, rows):
    """``rows`` are (patient, view, series_dir)."""
    pd.DataFrame([
        {"PatientID": patient, "StudyUID": study_of(patient), "View": view,
         "descriptive_path": f"BCS/{patient}/x/y/1-1.dcm",
         "classic_path": f"BCS/{patient}/{study_of(patient)}/{series_dir}/1-1.dcm"}
        for patient, view, series_dir in rows
    ], columns=FILE_PATH_COLUMNS).to_csv(path, index=False)
    return str(path)


def write_boxes(path, rows):
    """``rows`` are (patient, view, slice, class); the box covers ``BLOB``."""
    z_of, y_of, x_of = BLOB
    pd.DataFrame([
        {"PatientID": patient, "StudyUID": study_of(patient), "View": view, "Subject": 0,
         "Slice": z, "X": x_of.start, "Y": y_of.start,
         "Width": x_of.stop - x_of.start, "Height": y_of.stop - y_of.start,
         "Class": cls, "AD": 0}
        for patient, view, z, cls in rows
    ], columns=BOX_COLUMNS).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def corpus(tmp_path):
    """A normal exam and a cancer exam, one series each, both listed and labelled."""
    root = tmp_path / "tcia"
    write_series(str(root), "series-normal", "DBT-P00100", view="rmlo")
    write_series(str(root), "series-cancer", "DBT-P00200", view="rmlo", blob=True)
    labels = write_labels(tmp_path / "labels.csv", [
        ("DBT-P00100", "rmlo", "normal"),
        ("DBT-P00200", "rmlo", "cancer"),
    ])
    paths = write_file_paths(tmp_path / "paths.csv", [
        ("DBT-P00100", "rmlo", "series-normal"),
        ("DBT-P00200", "rmlo", "series-cancer"),
    ])
    boxes = write_boxes(tmp_path / "boxes.csv", [("DBT-P00200", "rmlo", 3, "cancer")])
    return str(root), labels, paths, boxes, str(tmp_path / "out")


def run(corpus, **kwargs):
    root, labels, paths, boxes, out = corpus
    kwargs.setdefault("boxes_csv", boxes)
    return preprocess_dbt_exams(root_dir=root, labels_csv=labels, file_paths_csv=paths,
                                output_dir=out, plane=PLANE, **kwargs), out


def load(out_dir, series_dir):
    with np.load(os.path.join(out_dir, f"{series_dir}.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# The geometry, which is where a meaningless AUC would come from
# --------------------------------------------------------------------------- #

def test_a_frame_keeps_its_aspect_ratio_and_is_padded_with_the_mean():
    volume = np.zeros((2, 100, 50), dtype=np.float32)
    resized, scale, (row_pad, col_pad) = resize_in_plane(volume, size=40)

    assert resized.shape == (2, 40, 40)
    assert scale == pytest.approx(0.4)          # bound by the long axis
    assert (row_pad, col_pad) == (0, 10)        # 50 * 0.4 = 20, centred in 40
    assert (resized[:, :, :10] == 0).all()      # zero is the mean after normalisation


def test_downsampling_averages_instead_of_sampling():
    """A bright line 1 px wide must survive as a dimmer band, not disappear.

    Sampling one row in four is how a small mass is lost. Here 4 source rows land on
    one output row, so an isolated row of 16 has to come back as exactly 4: sampling
    would give 16 (hit) or 0 (miss), and averaging cannot give anything else.
    """
    volume = np.zeros((1, 64, 64), dtype=np.float32)
    volume[0, 30, :] = 16.0
    resized, _scale, _offsets = resize_in_plane(volume, size=16)

    assert resized.max() == pytest.approx(4.0)
    assert resized.mean() == pytest.approx(volume.mean(), rel=1e-6)


def test_both_classes_come_out_with_the_same_shape(corpus):
    """The whole point: a shape that betrays the class would be read as signal."""
    (saved, skipped), out = run(corpus)
    assert (saved, skipped) == (2, 0)

    normal, cancer = load(out, "series-normal"), load(out, "series-cancer")
    assert normal["volume"].shape == cancer["volume"].shape
    assert normal["volume"].shape[1:] == (PLANE, PLANE)


def test_the_box_follows_the_pixels_through_the_resampling(corpus):
    """A box left at full-resolution coordinates would mark background."""
    (_saved, _skipped), out = run(corpus)
    cancer = load(out, "series-cancer")
    volume, mask = cancer["volume"].astype(np.float32), cancer["mask"]

    assert mask.sum() > 0
    assert set(np.unique(mask)) <= {0, 1}
    # The masked region is brighter than the rest of the frame: the blob is inside it.
    assert volume[mask > 0].mean() > volume[mask == 0].mean()


def test_a_small_box_does_not_round_away_to_nothing(tmp_path):
    """At 2457 -> 384 a 20 px box is 3 px. Rounding it to 0 would lose the lesion."""
    root = tmp_path / "tcia"
    write_series(str(root), "s", "DBT-P1", view="rmlo", blob=True)
    labels = write_labels(tmp_path / "l.csv", [("DBT-P1", "rmlo", "cancer")])
    paths = write_file_paths(tmp_path / "p.csv", [("DBT-P1", "rmlo", "s")])
    tiny = pd.read_csv(write_boxes(tmp_path / "b.csv", [("DBT-P1", "rmlo", 3, "cancer")]))
    tiny.loc[0, ["Width", "Height"]] = 1
    tiny.to_csv(tmp_path / "tiny.csv", index=False)

    preprocess_dbt_exams(root_dir=str(root), labels_csv=labels, file_paths_csv=paths,
                         boxes_csv=str(tmp_path / "tiny.csv"),
                         output_dir=str(tmp_path / "out"), plane=8, slice_margin=0)
    assert load(str(tmp_path / "out"), "s")["mask"].sum() >= 1


# --------------------------------------------------------------------------- #
# The label, which now comes from the labels table
# --------------------------------------------------------------------------- #

def test_a_normal_exam_is_a_negative_rather_than_a_skip(corpus):
    (_saved, _skipped), out = run(corpus)
    normal = load(out, "series-normal")

    assert str(normal["exam_status"]) == "normal"
    assert int(normal["label"]) == 0
    assert normal["mask"].sum() == 0
    assert "lesion_class" not in normal


def test_an_empty_mask_on_a_negative_is_not_reported_as_a_smell(corpus, caplog):
    """One warning per negative would bury the warnings that mean something."""
    with caplog.at_level(logging.WARNING):
        (_saved, _skipped), out = run(corpus)

    assert "mask is empty" not in caplog.text
    warnings = lineage.read_manifest(out)["validation_warnings"]
    assert not [w for w in warnings if "mask is empty" in w], warnings


@pytest.mark.parametrize("status,label", [("normal", 0), ("actionable", 0),
                                          ("benign", 0), ("cancer", 1)])
def test_only_a_cancer_is_a_positive(tmp_path, status, label):
    """A recall and a benign biopsy are not cancers, and the word is kept either way."""
    root = tmp_path / "tcia"
    write_series(str(root), "s", "DBT-P1", view="lcc")
    labels = write_labels(tmp_path / "l.csv", [("DBT-P1", "lcc", status)])
    paths = write_file_paths(tmp_path / "p.csv", [("DBT-P1", "lcc", "s")])

    preprocess_dbt_exams(root_dir=str(root), labels_csv=labels, file_paths_csv=paths,
                         output_dir=str(tmp_path / "out"), plane=PLANE)
    stored = load(str(tmp_path / "out"), "s")
    assert str(stored["exam_status"]) == status
    assert int(stored["label"]) == label
    assert validation.EXAM_STATUSES[status] == label


def test_a_series_without_a_labels_row_is_skipped_not_assumed_normal(tmp_path, caplog):
    """Guessing "normal" is how a corpus acquires negatives nobody ever called normal."""
    root = tmp_path / "tcia"
    write_series(str(root), "s", "DBT-P1", view="lcc")
    labels = write_labels(tmp_path / "l.csv", [("DBT-P2", "lcc", "normal")])
    paths = write_file_paths(tmp_path / "p.csv", [("DBT-P1", "lcc", "s")])

    with caplog.at_level(logging.WARNING):
        saved, skipped = preprocess_dbt_exams(root_dir=str(root), labels_csv=labels,
                                              file_paths_csv=paths,
                                              output_dir=str(tmp_path / "out"),
                                              plane=PLANE)
    assert (saved, skipped) == (0, 1)
    assert "no labels row" in caplog.text
    parameters = lineage.read_manifest(str(tmp_path / "out"))["parameters"]
    assert parameters["skipped_unlabelled"] == 1


# --------------------------------------------------------------------------- #
# Laterality and the manifest
# --------------------------------------------------------------------------- #

def test_a_negative_is_flipped_by_the_same_rule_as_a_positive(tmp_path):
    """Otherwise "was flipped" would correlate with "has a box", hence with the label."""
    root = tmp_path / "tcia"
    # Listed as a left view, pixels on the right: a study stored rotated, and no box.
    write_series(str(root), "series-mirror", "DBT-P1", view="lmlo", laterality="R")
    labels = write_labels(tmp_path / "l.csv", [("DBT-P1", "lmlo", "normal")])
    paths = write_file_paths(tmp_path / "p.csv", [("DBT-P1", "lmlo", "series-mirror")])
    out = str(tmp_path / "out")

    preprocess_dbt_exams(root_dir=str(root), labels_csv=labels, file_paths_csv=paths,
                         output_dir=out, plane=PLANE)
    manifest = lineage.read_manifest(out)
    assert manifest["parameters"]["mirrored_series"] == 1
    assert manifest["cases"]["series-mirror"]["mirrored"] is True


def test_the_manifest_counts_each_status_and_names_its_tables(corpus):
    (_saved, _skipped), out = run(corpus)
    manifest = lineage.read_manifest(out)

    assert manifest["parameters"]["saved_by_status"] == {"normal": 1, "actionable": 0,
                                                        "benign": 0, "cancer": 1}
    assert manifest["parameters"]["plane"] == PLANE
    assert [os.path.basename(p) for p in manifest["parameters"]["labels"]] == ["labels.csv"]
    assert manifest["cases"]["series-cancer"]["exam_status"] == "cancer"
    assert manifest["cases"]["series-cancer"]["in_plane_scale"] > 0


def test_a_second_pass_skips_what_is_already_written(corpus):
    """A pass over this collection is hours of decoding; it has to be resumable."""
    (first, _), out = run(corpus)
    (second, skipped), _ = run(corpus)

    assert first == 2
    assert (second, skipped) == (0, 2)
    assert lineage.read_manifest(out)["parameters"]["skipped_existing"] == 2


def test_a_resumed_pass_keeps_every_case_in_the_manifest(corpus):
    """The manifest describes the corpus, not the last run.

    A resumed pass used to write a manifest holding only the series it had just
    decoded, so adding one series to an 870-case corpus left a manifest of one case --
    and the catalogue, which reads the manifest, lost the other 869.
    """
    root, labels, paths, boxes, out = corpus
    run(corpus)
    write_series(root, "series-late", "DBT-P00300", view="rmlo")
    pd.concat([pd.read_csv(labels), pd.read_csv(write_labels(
        os.path.join(os.path.dirname(labels), "late.csv"), [("DBT-P00300", "rmlo", "normal")]))]
              ).to_csv(labels, index=False)
    pd.concat([pd.read_csv(paths), pd.read_csv(write_file_paths(
        os.path.join(os.path.dirname(paths), "late_paths.csv"),
        [("DBT-P00300", "rmlo", "series-late")]))]).to_csv(paths, index=False)

    (saved, skipped), _ = run(corpus)

    assert (saved, skipped) == (1, 2)
    cases = lineage.read_manifest(out)["cases"]
    assert set(cases) == {"series-normal", "series-cancer", "series-late"}
    assert cases["series-cancer"]["exam_status"] == "cancer"  # carried over, not re-guessed


def test_a_volume_without_a_manifest_entry_is_processed_again(corpus):
    """A .npz with no manifest entry comes from an interrupted pass: its provenance is
    unknown, so it is rebuilt rather than trusted."""
    (_first, _), out = run(corpus)
    os.remove(os.path.join(out, lineage.MANIFEST_NAME))

    (saved, skipped), _ = run(corpus)

    assert (saved, skipped) == (2, 0)
    assert len(lineage.read_manifest(out)["cases"]) == 2


def test_a_series_absent_from_the_inventory_is_skipped(tmp_path, caplog):
    root = tmp_path / "tcia"
    write_series(str(root), "s", "DBT-P1", view="lcc")
    labels = write_labels(tmp_path / "l.csv", [("DBT-P1", "lcc", "normal")])
    paths = write_file_paths(tmp_path / "p.csv", [("DBT-P1", "lcc", "elsewhere")])

    with caplog.at_level(logging.WARNING):
        saved, skipped = preprocess_dbt_exams(root_dir=str(root), labels_csv=labels,
                                              file_paths_csv=paths,
                                              output_dir=str(tmp_path / "out"),
                                              plane=PLANE)
    assert (saved, skipped) == (0, 1)
    assert "not in the file-paths inventory" in caplog.text
