"""What ``preprocess_dbt_with_boxes`` has to get right: which box, and whose.

Two bugs lived here, and both were silent. ``Class`` -- benign or cancer -- shipped
with the annotations and nothing read it, so both kinds of box went into one mask and
"positive" meant "some lesion": of the 72 patients preprocessed that way, 48 were
benign. And the series was matched to its box by the DICOM laterality tag, which
reads ``L`` on all 262 downloaded series: 147 of 253 annotated series were found, and
23 masks were painted on background instead of tissue.

So these tests pin, on the class: the column is required, it reaches the ``.npz``, and
choosing which classes to paint does not change what the label says. And on the
match: laterality is taken from the pixels, the tag is ignored even when it lies, a
study stored rotated relative to its annotation frame is flipped rather than
mismatched, and a series whose patient has nothing annotated at its view position is
dropped without paying to decode it.

Synthetic multi-frame DICOMs stand in for the real series: 22 GB of tomosynthesis is
not needed to check bookkeeping, and a fake series makes the awkward cases (a mixed
series, an unknown class, a mirrored study, a lying tag) constructible, which real
data does not.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import validation
from TransformData import (
    dbt_series_view,
    image_laterality,
    preprocess_dbt_with_boxes,
    save_preprocessed,
)

BOX_COLUMNS = ["PatientID", "StudyUID", "View", "Subject", "Slice",
               "X", "Y", "Width", "Height", "Class", "AD"]


def write_series(root, series_dir, patient_id, view="rmlo", depth=8, size=64,
                 stored_laterality=None):
    """One synthetic multi-frame DBT series, readable by ``pydicom.dcmread``.

    The laterality is carried by the *pixels*, since that is what the matching reads:
    the breast is painted on the side ``stored_laterality`` names, defaulting to the
    side ``view`` implies. The DICOM tag is set to ``L`` whatever the pixels say --
    which is exactly what the real collection does on all 262 series, and what these
    tests need it to do.
    """
    folder = os.path.join(root, series_dir)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "1-1.dcm")

    rng = np.random.default_rng(len(patient_id) + depth)
    pixels = rng.integers(0, 60, size=(depth, size, size), dtype=np.uint16)
    laterality = (stored_laterality or view[0]).upper()
    breast = slice(size // 2, size) if laterality == "R" else slice(0, size // 2)
    pixels[:, :, breast] = rng.integers(2000, 4000, size=(depth, size, size // 2),
                                        dtype=np.uint16)

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    # Breast Tomosynthesis Image Storage.
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.13.1.3"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = patient_id
    ds.ImageLaterality = "L"                  # wrong on purpose: so does the real data
    ds.ViewPosition = view[1:].upper()        # 'mlo' -> 'MLO'
    ds.Rows, ds.Columns, ds.NumberOfFrames = size, size, depth
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


def write_boxes(path, rows):
    """A boxes CSV with the real schema; ``rows`` are (patient, view, slice, class)."""
    pd.DataFrame([
        {"PatientID": patient, "StudyUID": f"S-{patient}", "View": view, "Subject": 0,
         "Slice": z, "X": 12, "Y": 14, "Width": 20, "Height": 18, "Class": cls, "AD": 0}
        for patient, view, z, cls in rows
    ], columns=BOX_COLUMNS).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def corpus(tmp_path):
    """One benign patient and one cancer patient, one series each."""
    root = tmp_path / "tcia"
    write_series(str(root), "series-benign", "DBT-P00001")
    write_series(str(root), "series-cancer", "DBT-P00002")
    boxes = write_boxes(tmp_path / "boxes.csv", [
        ("DBT-P00001", "rmlo", 3, "benign"),
        ("DBT-P00002", "rmlo", 4, "cancer"),
    ])
    return str(root), boxes, str(tmp_path / "out")


def load(out_dir, series_dir):
    with np.load(os.path.join(out_dir, f"{series_dir}.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# The column itself
# --------------------------------------------------------------------------- #

def test_a_boxes_csv_without_the_class_column_is_refused(tmp_path, corpus):
    """Silently treating every box as one lesion is the bug being fixed."""
    root, boxes, out = corpus
    df = pd.read_csv(boxes).drop(columns=["Class"])
    df.to_csv(tmp_path / "no_class.csv", index=False)

    with pytest.raises(ValueError, match="Class"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=str(tmp_path / "no_class.csv"),
                                  output_dir=out)


def test_an_unrecognised_class_is_refused(tmp_path, corpus):
    """A third value would be dropped by the class filter without a word."""
    root, _boxes, out = corpus
    path = write_boxes(tmp_path / "odd.csv", [("DBT-P00001", "rmlo", 3, "malignant")])

    with pytest.raises(ValueError, match="malignant"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=path, output_dir=out)


@pytest.mark.parametrize("spelling,expected", [
    ("benign", 0), ("cancer", 1), ("Cancer", 1), ("  BENIGN ", 0),
])
def test_the_class_encoding_lives_in_one_place(spelling, expected):
    canonical, label = validation.lesion_class_label(spelling)
    assert label == expected
    assert canonical == spelling.strip().lower()


def test_a_class_typo_never_reaches_the_disk(tmp_path):
    """save_preprocessed is the single writer, so it is where the check belongs."""
    volume = np.zeros((4, 32, 32), dtype=np.float32)
    mask = np.zeros((4, 32, 32), dtype=np.uint8)
    mask[1, 4:8, 4:8] = 1

    with pytest.raises(validation.VolumeValidationError, match="lesion_class"):
        save_preprocessed("case", volume, mask, str(tmp_path), lesion_class="maligant")
    assert not os.path.exists(os.path.join(str(tmp_path), "case.npz"))


# --------------------------------------------------------------------------- #
# What ends up in the files
# --------------------------------------------------------------------------- #

def test_each_file_carries_its_own_class(corpus):
    root, boxes, out = corpus
    saved, skipped = preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes,
                                               output_dir=out)
    assert (saved, skipped) == (2, 0)

    benign, cancer = load(out, "series-benign"), load(out, "series-cancer")
    assert str(benign["lesion_class"]) == "benign" and int(benign["label"]) == 0
    assert str(cancer["lesion_class"]) == "cancer" and int(cancer["label"]) == 1
    # Both are lesions, so both keep a mask: the label is what separates them.
    assert benign["mask"].sum() > 0 and cancer["mask"].sum() > 0


def test_the_mask_stays_binary(corpus):
    """The class is a label, not a second value painted into the mask."""
    root, boxes, out = corpus
    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, output_dir=out)
    for series in ("series-benign", "series-cancer"):
        assert set(np.unique(load(out, series)["mask"])) <= {0, 1}


def test_an_unlabelled_volume_stores_no_label(tmp_path):
    """The DCE-MRI collection has no class column; absent must mean unknown."""
    volume = np.zeros((4, 32, 32), dtype=np.float32)
    mask = np.zeros((4, 32, 32), dtype=np.uint8)
    mask[1, 4:8, 4:8] = 1
    save_preprocessed("case", volume, mask, str(tmp_path))

    stored = load(str(tmp_path), "case")
    assert "lesion_class" not in stored and "label" not in stored


# --------------------------------------------------------------------------- #
# Choosing what to paint, without changing what the label says
# --------------------------------------------------------------------------- #

def test_painting_cancer_only_leaves_the_benign_series_out(corpus):
    root, boxes, out = corpus
    saved, skipped = preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes,
                                               output_dir=out, mask_classes=("cancer",))
    assert (saved, skipped) == (1, 1)
    assert not os.path.exists(os.path.join(out, "series-benign.npz"))
    assert int(load(out, "series-cancer")["label"]) == 1


def test_a_mixed_series_is_labelled_cancer_and_says_so(tmp_path, caplog):
    """One missed cancer is not offset by a correctly called benign."""
    root = tmp_path / "tcia"
    write_series(str(root), "series-mixed", "DBT-P00003")
    boxes = write_boxes(tmp_path / "mixed.csv", [
        ("DBT-P00003", "rmlo", 2, "benign"),
        ("DBT-P00003", "rmlo", 5, "cancer"),
    ])
    out = str(tmp_path / "out")

    with caplog.at_level(logging.WARNING):
        preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes, output_dir=out)

    assert int(load(out, "series-mixed")["label"]) == 1
    assert "several classes" in caplog.text


def test_an_unknown_mask_class_is_refused(corpus):
    root, boxes, out = corpus
    with pytest.raises(ValueError, match="mask_classes"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, output_dir=out,
                                  mask_classes=("cancer", "suspicious"))


# --------------------------------------------------------------------------- #
# The manifest, which is what a reader checks before trusting a corpus
# --------------------------------------------------------------------------- #

def test_the_manifest_counts_each_class(corpus):
    import lineage

    root, boxes, out = corpus
    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, output_dir=out)

    manifest = lineage.read_manifest(out)
    assert manifest["parameters"]["saved_by_class"] == {"benign": 1, "cancer": 1}
    assert manifest["parameters"]["mask_classes"] == ["benign", "cancer"]
    assert manifest["n_cases"] == 2
    assert {case["lesion_class"] for case in manifest["cases"].values()} == {"benign", "cancer"}


def test_the_manifest_keeps_every_series_of_a_patient(tmp_path):
    """A DBT patient has up to four views. Keyed by patient, three of them vanish."""
    import lineage

    root = tmp_path / "tcia"
    write_series(str(root), "series-rmlo", "DBT-P00007", view="rmlo")
    write_series(str(root), "series-rcc", "DBT-P00007", view="rcc")
    boxes = write_boxes(tmp_path / "two_views.csv", [
        ("DBT-P00007", "rmlo", 3, "cancer"),
        ("DBT-P00007", "rcc", 4, "cancer"),
    ])
    out = str(tmp_path / "out")

    saved, _skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                                output_dir=out)
    manifest = lineage.read_manifest(out)
    assert saved == 2
    assert manifest["n_cases"] == 2
    assert set(manifest["cases"]) == {"series-rmlo", "series-rcc"}
    # ... and each one still says which patient it belongs to, for the split.
    assert {c["case_id"] for c in manifest["cases"].values()} == {"DBT-P00007"}


def test_the_manifest_separates_the_two_reasons_for_skipping(tmp_path, corpus):
    """"No box at all" and "no box of the painted class" are different facts."""
    import lineage

    root, boxes, out = corpus
    write_series(root, "series-unannotated", "DBT-P09999")
    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, output_dir=out,
                              mask_classes=("cancer",))

    parameters = lineage.read_manifest(out)["parameters"]
    assert parameters["skipped_unannotated"] == 1   # DBT-P09999, not in the CSV
    assert parameters["skipped_unpainted"] == 1     # the benign patient


# --------------------------------------------------------------------------- #
# Which series a box belongs to
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("laterality", ["R", "L"])
def test_laterality_comes_from_the_side_that_carries_signal(laterality):
    frame = np.zeros((16, 16), dtype=np.uint16)
    breast = slice(8, 16) if laterality == "R" else slice(0, 8)
    frame[:, breast] = 3000
    assert image_laterality(frame) == laterality


def test_the_view_ignores_a_lying_laterality_tag(tmp_path):
    """The real collection tags all 262 series 'L'; 134 of them are right breasts."""
    import pydicom

    folder = write_series(str(tmp_path), "s", "DBT-P1", view="rmlo")
    ds = pydicom.dcmread(os.path.join(folder, "1-1.dcm"))
    assert ds.ImageLaterality == "L"                     # the tag says left
    assert dbt_series_view(ds, ds.pixel_array[0]) == "rmlo"   # the pixels say right


def test_a_right_breast_series_gets_its_own_box_not_the_left_one(tmp_path):
    """Nine series were matched to the other breast's box: same patient, both views."""
    root = tmp_path / "tcia"
    write_series(str(root), "series-right", "DBT-P00500", view="rmlo")
    write_series(str(root), "series-left", "DBT-P00500", view="lmlo")
    boxes = write_boxes(tmp_path / "both.csv", [
        ("DBT-P00500", "rmlo", 3, "cancer"),
        ("DBT-P00500", "lmlo", 4, "benign"),
    ])
    out = str(tmp_path / "out")

    saved, skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                               output_dir=out)
    assert (saved, skipped) == (2, 0)
    # Each series took the box of its own side, so each carries its own class.
    assert int(load(out, "series-right")["label"]) == 1
    assert int(load(out, "series-left")["label"]) == 0


def test_a_study_stored_rotated_is_flipped_rather_than_mismatched(tmp_path):
    """Seven patients, left-only boxes, both series reading right by pixels."""
    import lineage

    root = tmp_path / "tcia"
    # Pixels on the right, annotation on the left, and no right-side box anywhere.
    write_series(str(root), "series-mirrored", "DBT-P02471", view="lmlo",
                 stored_laterality="R")
    boxes = write_boxes(tmp_path / "left_only.csv",
                        [("DBT-P02471", "lmlo", 3, "cancer")])
    out = str(tmp_path / "out")

    saved, skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                               output_dir=out)
    assert (saved, skipped) == (1, 0)
    assert load(out, "series-mirrored")["mask"].sum() > 0

    manifest = lineage.read_manifest(out)
    assert manifest["parameters"]["mirrored_series"] == 1
    assert manifest["cases"]["series-mirrored"]["mirrored"] is True


def test_a_matched_series_is_not_recorded_as_mirrored(corpus):
    import lineage

    root, boxes, out = corpus
    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, output_dir=out)

    manifest = lineage.read_manifest(out)
    assert manifest["parameters"]["mirrored_series"] == 0
    assert all(case["mirrored"] is False for case in manifest["cases"].values())


def test_an_unannotated_view_position_is_dropped_without_decoding(tmp_path, monkeypatch):
    """Pixels cost 7-17 s and ~100 MB per series; the header settles this one."""
    import pydicom

    root = tmp_path / "tcia"
    write_series(str(root), "series-cc", "DBT-P00700", view="rcc")
    write_series(str(root), "series-mlo", "DBT-P00700", view="rmlo")
    # Annotated at the MLO position only.
    boxes = write_boxes(tmp_path / "mlo_only.csv",
                        [("DBT-P00700", "rmlo", 3, "cancer")])

    real_dcmread = pydicom.dcmread
    with_pixels = []

    def counting_dcmread(path, *args, **kwargs):
        if not kwargs.get("stop_before_pixels"):
            with_pixels.append(str(path))
        return real_dcmread(path, *args, **kwargs)

    monkeypatch.setattr("TransformData.pydicom.dcmread", counting_dcmread)
    saved, skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                               output_dir=str(tmp_path / "out"))

    assert (saved, skipped) == (1, 1)
    assert len(with_pixels) == 1 and "series-mlo" in with_pixels[0]
