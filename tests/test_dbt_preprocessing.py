"""What ``preprocess_dbt_with_boxes`` has to get right: which box, and whose.

Three bugs lived here, and all three were silent. ``Class`` -- benign or cancer --
shipped with the annotations and nothing read it, so both kinds of box went into one
mask and "positive" meant "some lesion": of the 72 patients preprocessed that way, 48
were benign. Then the series was matched to its box by the DICOM laterality tag, which
reads ``L`` on all 262 downloaded series: 147 of 253 annotated series were found, and 23
masks were painted on background instead of tissue. Deriving the laterality from the
pixels instead fixed those 23, and was still an inference: right 237 times out of 262,
253 of the 260 annotated series found, and 4 masks taken from the *other* acquisition of
a repeated view.

The collection publishes what all three were guessing. ``BCS-DBT-file-paths-*.csv``
gives ``(PatientID, StudyUID, View)`` for every series folder, so the match is a join.
These tests pin that -- the view comes from the inventory, a repeated acquisition keeps
its own box, a folder the inventory does not list is skipped and counted rather than
matched to something plausible -- and keep the pixels at the one job the dataset's own
reader gives them: deciding whether a study is stored rotated relative to the frame its
boxes live in. On the class, they pin that the column is required, that it reaches the
``.npz``, and that choosing what to paint does not change what the label says.

Synthetic multi-frame DICOMs stand in for the real series: 22 GB of tomosynthesis is
not needed to check bookkeeping, and a fake series makes the awkward cases (a mixed
series, an unknown class, a mirrored study, a repeated view, a lying tag) constructible,
which real data does not.
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
    dbt_patient_status,
    image_laterality,
    preprocess_dbt_with_boxes,
    save_preprocessed,
    series_uid_from_classic_path,
    view_position_of,
)

BOX_COLUMNS = ["PatientID", "StudyUID", "View", "Subject", "Slice",
               "X", "Y", "Width", "Height", "Class", "AD"]
FILE_PATH_COLUMNS = ["PatientID", "StudyUID", "View", "descriptive_path", "classic_path"]


def study_of(patient):
    """The StudyUID the helpers below agree on, so boxes and inventory line up."""
    return f"S-{patient}"


def write_series(root, series_dir, patient_id, view="rmlo", depth=8, size=64,
                 stored_laterality=None):
    """One synthetic multi-frame DBT series, readable by ``pydicom.dcmread``.

    ``stored_laterality`` says which side the *pixels* carry the breast on, defaulting
    to the side ``view`` implies; the two differ on a study stored rotated relative to
    its annotation frame. The DICOM laterality tag is set to ``L`` whatever the pixels
    say -- which is what the real collection does on all 262 series, and what these
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
    ds.ViewPosition = view_position_of(view).upper()   # 'lmlo1' -> 'MLO'
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
        {"PatientID": patient, "StudyUID": study_of(patient), "View": view, "Subject": 0,
         "Slice": z, "X": 12, "Y": 14, "Width": 20, "Height": 18, "Class": cls, "AD": 0}
        for patient, view, z, cls in rows
    ], columns=BOX_COLUMNS).to_csv(path, index=False)
    return str(path)


def write_file_paths(path, rows):
    """The series inventory; ``rows`` are (patient, view, series_dir).

    ``classic_path`` is shaped as the collection ships it --
    ``collection/patient/study/series/1-1.dcm`` -- because the series folder name is
    read back out of it.
    """
    pd.DataFrame([
        {"PatientID": patient,
         "StudyUID": study_of(patient),
         "View": view,
         "descriptive_path": f"Breast-Cancer-Screening-DBT/{patient}/01-01-2000/x/1-1.dcm",
         "classic_path": (f"Breast-Cancer-Screening-DBT/{patient}/"
                          f"{study_of(patient)}/{series_dir}/1-1.dcm")}
        for patient, view, series_dir in rows
    ], columns=FILE_PATH_COLUMNS).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def corpus(tmp_path):
    """One benign patient and one cancer patient, one series each, both listed."""
    root = tmp_path / "tcia"
    write_series(str(root), "series-benign", "DBT-P00001")
    write_series(str(root), "series-cancer", "DBT-P00002")
    boxes = write_boxes(tmp_path / "boxes.csv", [
        ("DBT-P00001", "rmlo", 3, "benign"),
        ("DBT-P00002", "rmlo", 4, "cancer"),
    ])
    paths = write_file_paths(tmp_path / "file_paths.csv", [
        ("DBT-P00001", "rmlo", "series-benign"),
        ("DBT-P00002", "rmlo", "series-cancer"),
    ])
    return str(root), boxes, paths, str(tmp_path / "out")


def load(out_dir, series_dir):
    with np.load(os.path.join(out_dir, f"{series_dir}.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# The column itself
# --------------------------------------------------------------------------- #

def test_a_boxes_csv_without_the_class_column_is_refused(tmp_path, corpus):
    """Silently treating every box as one lesion is the bug being fixed."""
    root, boxes, paths, out = corpus
    df = pd.read_csv(boxes).drop(columns=["Class"])
    df.to_csv(tmp_path / "no_class.csv", index=False)

    with pytest.raises(ValueError, match="Class"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=str(tmp_path / "no_class.csv"),
                                  file_paths_csv=paths, output_dir=out)


def test_a_boxes_csv_without_a_join_column_is_refused(tmp_path, corpus):
    """Without StudyUID there is no key to match a series on, only a guess again."""
    root, boxes, paths, out = corpus
    pd.read_csv(boxes).drop(columns=["StudyUID"]).to_csv(tmp_path / "no_study.csv",
                                                         index=False)

    with pytest.raises(ValueError, match="StudyUID"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=str(tmp_path / "no_study.csv"),
                                  file_paths_csv=paths, output_dir=out)


def test_an_unrecognised_class_is_refused(tmp_path, corpus):
    """An unknown value would be dropped by the class filter, mask and all."""
    root, boxes, paths, out = corpus
    df = pd.read_csv(boxes)
    df.loc[0, "Class"] = "suspicious"
    path = str(tmp_path / "odd_class.csv")
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="suspicious"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=path, file_paths_csv=paths,
                                  output_dir=out)


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
    root, boxes, paths, out = corpus
    saved, skipped = preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes,
                                               file_paths_csv=paths, output_dir=out)
    assert (saved, skipped) == (2, 0)

    benign, cancer = load(out, "series-benign"), load(out, "series-cancer")
    assert str(benign["lesion_class"]) == "benign" and int(benign["label"]) == 0
    assert str(cancer["lesion_class"]) == "cancer" and int(cancer["label"]) == 1
    # Both are lesions, so both keep a mask: the label is what separates them.
    assert benign["mask"].sum() > 0 and cancer["mask"].sum() > 0


def test_the_mask_stays_binary(corpus):
    """The class is a label, not a second value painted into the mask."""
    root, boxes, paths, out = corpus
    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, file_paths_csv=paths,
                              output_dir=out)
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
    root, boxes, paths, out = corpus
    saved, skipped = preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes,
                                               file_paths_csv=paths, output_dir=out,
                                               mask_classes=("cancer",))
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
    paths = write_file_paths(tmp_path / "mixed_paths.csv",
                            [("DBT-P00003", "rmlo", "series-mixed")])
    out = str(tmp_path / "out")

    with caplog.at_level(logging.WARNING):
        preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                  file_paths_csv=paths, output_dir=out)

    assert int(load(out, "series-mixed")["label"]) == 1
    assert "several classes" in caplog.text


def test_an_unknown_mask_class_is_refused(corpus):
    root, boxes, paths, out = corpus
    with pytest.raises(ValueError, match="mask_classes"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, file_paths_csv=paths,
                                  output_dir=out, mask_classes=("cancer", "suspicious"))


# --------------------------------------------------------------------------- #
# The manifest, which is what a reader checks before trusting a corpus
# --------------------------------------------------------------------------- #

def test_the_manifest_counts_each_class(corpus):
    import lineage

    root, boxes, paths, out = corpus
    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, file_paths_csv=paths,
                              output_dir=out)

    manifest = lineage.read_manifest(out)
    assert manifest["parameters"]["saved_by_class"] == {"benign": 1, "cancer": 1}
    assert manifest["parameters"]["mask_classes"] == ["benign", "cancer"]
    assert manifest["n_cases"] == 2
    assert {case["lesion_class"] for case in manifest["cases"].values()} == {"benign", "cancer"}


def test_the_manifest_records_the_inventory_and_the_view(corpus):
    """Which table matched the boxes, and to which view: that is the provenance."""
    import lineage

    root, boxes, paths, out = corpus
    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, file_paths_csv=paths,
                              output_dir=out)

    manifest = lineage.read_manifest(out)
    assert [os.path.basename(p) for p in manifest["parameters"]["file_paths"]] \
        == ["file_paths.csv"]
    assert manifest["cases"]["series-benign"]["view"] == "rmlo"
    assert manifest["cases"]["series-benign"]["study_uid"] == "S-DBT-P00001"


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
    paths = write_file_paths(tmp_path / "two_views_paths.csv", [
        ("DBT-P00007", "rmlo", "series-rmlo"),
        ("DBT-P00007", "rcc", "series-rcc"),
    ])
    out = str(tmp_path / "out")

    saved, _skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                                file_paths_csv=paths, output_dir=out)
    manifest = lineage.read_manifest(out)
    assert saved == 2
    assert manifest["n_cases"] == 2
    assert set(manifest["cases"]) == {"series-rmlo", "series-rcc"}
    # ... and each one still says which patient it belongs to, for the split.
    assert {c["case_id"] for c in manifest["cases"].values()} == {"DBT-P00007"}


def test_the_manifest_separates_the_reasons_for_skipping(tmp_path, corpus):
    """"No box", "no box of the painted class" and "not listed" are different facts."""
    import lineage

    root, boxes, paths, out = corpus
    write_series(root, "series-unannotated", "DBT-P09999")
    write_series(root, "series-unlisted", "DBT-P08888")
    listed = pd.read_csv(paths)
    listed = pd.concat([listed, pd.read_csv(write_file_paths(
        tmp_path / "extra.csv", [("DBT-P09999", "rmlo", "series-unannotated")]))])
    listed.to_csv(tmp_path / "with_unannotated.csv", index=False)

    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes,
                              file_paths_csv=str(tmp_path / "with_unannotated.csv"),
                              output_dir=out, mask_classes=("cancer",))

    parameters = lineage.read_manifest(out)["parameters"]
    assert parameters["skipped_unannotated"] == 1   # DBT-P09999: listed, no box
    assert parameters["skipped_unpainted"] == 1     # the benign patient
    assert parameters["skipped_unlisted"] == 1      # DBT-P08888: no inventory row


# --------------------------------------------------------------------------- #
# Which series a box belongs to
# --------------------------------------------------------------------------- #

def test_the_inventory_names_the_series_folder(tmp_path):
    """The folder name is the series UID, second-to-last in ``classic_path``."""
    assert series_uid_from_classic_path(
        "Breast-Cancer-Screening-DBT/DBT-P00013/1.2.826.0/1.2.826.0.99/1-1.dcm"
    ) == "1.2.826.0.99"
    # Windows separators reach this from a path pasted out of a local listing.
    assert series_uid_from_classic_path(r"collection\patient\study\series\1-1.dcm") \
        == "series"
    with pytest.raises(ValueError, match="series folder"):
        series_uid_from_classic_path("1-1.dcm")


@pytest.mark.parametrize("view,position", [("rmlo", "mlo"), ("lcc", "cc"),
                                           ("lmlo1", "mlo"), ("rcc2", "cc")])
def test_a_repeated_view_still_names_its_incidence(view, position):
    """``ViewPosition`` carries the incidence only; the repeat index is not in it."""
    assert view_position_of(view) == position


def test_a_file_paths_csv_missing_its_columns_is_refused(tmp_path, corpus):
    root, boxes, paths, out = corpus
    pd.read_csv(paths).drop(columns=["View"]).to_csv(tmp_path / "no_view.csv", index=False)

    with pytest.raises(ValueError, match="View"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, output_dir=out,
                                  file_paths_csv=str(tmp_path / "no_view.csv"))


def test_a_series_listed_twice_is_refused(tmp_path, corpus):
    """Two rows for one folder means the lookup would resolve to whichever came first."""
    root, boxes, paths, out = corpus
    doubled = pd.concat([pd.read_csv(paths)] * 2)
    doubled.to_csv(tmp_path / "doubled.csv", index=False)

    with pytest.raises(ValueError, match="twice"):
        preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, output_dir=out,
                                  file_paths_csv=str(tmp_path / "doubled.csv"))


def test_a_series_absent_from_the_inventory_is_skipped_not_guessed(tmp_path, caplog):
    """Nothing establishes which box is its own, so no box is given to it."""
    root = tmp_path / "tcia"
    write_series(str(root), "series-unlisted", "DBT-P00600", view="rmlo")
    boxes = write_boxes(tmp_path / "boxes.csv", [("DBT-P00600", "rmlo", 3, "cancer")])
    paths = write_file_paths(tmp_path / "elsewhere.csv",
                             [("DBT-P00600", "rmlo", "some-other-folder")])
    out = str(tmp_path / "out")

    with caplog.at_level(logging.WARNING):
        saved, skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                                   file_paths_csv=paths, output_dir=out)

    assert (saved, skipped) == (0, 1)
    assert "not in the file-paths inventory" in caplog.text


def test_a_right_breast_series_gets_its_own_box_not_the_left_one(tmp_path):
    """Nine series were matched to the other breast's box: same patient, both views."""
    root = tmp_path / "tcia"
    write_series(str(root), "series-right", "DBT-P00500", view="rmlo")
    write_series(str(root), "series-left", "DBT-P00500", view="lmlo")
    boxes = write_boxes(tmp_path / "both.csv", [
        ("DBT-P00500", "rmlo", 3, "cancer"),
        ("DBT-P00500", "lmlo", 4, "benign"),
    ])
    paths = write_file_paths(tmp_path / "both_paths.csv", [
        ("DBT-P00500", "rmlo", "series-right"),
        ("DBT-P00500", "lmlo", "series-left"),
    ])
    out = str(tmp_path / "out")

    saved, skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                               file_paths_csv=paths, output_dir=out)
    assert (saved, skipped) == (2, 0)
    # Each series took the box of its own side, so each carries its own class.
    assert int(load(out, "series-right")["label"]) == 1
    assert int(load(out, "series-left")["label"]) == 0


def test_a_repeated_acquisition_keeps_its_own_box(tmp_path):
    """The case the pixels cannot decide: two ``lmlo`` series, one patient, one side.

    Four masks came from the other acquisition of the same view, because nothing in the
    image says which of the two it is. ``View`` in the inventory does -- ``lmlo`` against
    ``lmlo1`` -- so each series takes its own box, and both are kept instead of one.
    """
    root = tmp_path / "tcia"
    write_series(str(root), "series-first", "DBT-P00800", view="lmlo")
    write_series(str(root), "series-repeat", "DBT-P00800", view="lmlo1")
    boxes = write_boxes(tmp_path / "repeat.csv", [
        ("DBT-P00800", "lmlo", 2, "benign"),
        ("DBT-P00800", "lmlo1", 5, "cancer"),
    ])
    paths = write_file_paths(tmp_path / "repeat_paths.csv", [
        ("DBT-P00800", "lmlo", "series-first"),
        ("DBT-P00800", "lmlo1", "series-repeat"),
    ])
    out = str(tmp_path / "out")

    saved, skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                               file_paths_csv=paths, output_dir=out,
                                               slice_margin=0, crop=False)
    assert (saved, skipped) == (2, 0)
    assert int(load(out, "series-first")["label"]) == 0
    assert int(load(out, "series-repeat")["label"]) == 1
    # And each mask sits on the slice its own row names, not the other row's.
    assert load(out, "series-first")["mask"].sum(axis=(1, 2)).argmax() == 2
    assert load(out, "series-repeat")["mask"].sum(axis=(1, 2)).argmax() == 5


def test_a_study_stored_rotated_is_flipped_rather_than_mismatched(tmp_path):
    """Seven patients, left views, both series reading right by pixels."""
    import lineage

    root = tmp_path / "tcia"
    # The inventory says left; the pixels carry the breast on the right.
    write_series(str(root), "series-mirrored", "DBT-P02471", view="lmlo",
                 stored_laterality="R")
    boxes = write_boxes(tmp_path / "left_only.csv",
                        [("DBT-P02471", "lmlo", 3, "cancer")])
    paths = write_file_paths(tmp_path / "left_paths.csv",
                             [("DBT-P02471", "lmlo", "series-mirrored")])
    out = str(tmp_path / "out")

    saved, skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                               file_paths_csv=paths, output_dir=out)
    assert (saved, skipped) == (1, 0)
    assert load(out, "series-mirrored")["mask"].sum() > 0

    manifest = lineage.read_manifest(out)
    assert manifest["parameters"]["mirrored_series"] == 1
    assert manifest["cases"]["series-mirrored"]["mirrored"] is True


def test_a_matched_series_is_not_recorded_as_mirrored(corpus):
    import lineage

    root, boxes, paths, out = corpus
    preprocess_dbt_with_boxes(root_dir=root, boxes_csv=boxes, file_paths_csv=paths,
                              output_dir=out)

    manifest = lineage.read_manifest(out)
    assert manifest["parameters"]["mirrored_series"] == 0
    assert all(case["mirrored"] is False for case in manifest["cases"].values())


@pytest.mark.parametrize("laterality", ["R", "L"])
def test_laterality_comes_from_the_side_that_carries_signal(laterality):
    """The pixels keep one job: deciding the flip, as the reference reader does."""
    frame = np.zeros((16, 16), dtype=np.uint16)
    breast = slice(8, 16) if laterality == "R" else slice(0, 8)
    frame[:, breast] = 3000
    assert image_laterality(frame) == laterality


def test_a_patient_id_disagreeing_with_the_inventory_is_reported(tmp_path, caplog):
    """The inventory is trusted, but a mismatch means the mapping is off somewhere."""
    root = tmp_path / "tcia"
    write_series(str(root), "series-odd", "DBT-P00999", view="rmlo")
    boxes = write_boxes(tmp_path / "boxes.csv", [("DBT-P00111", "rmlo", 3, "cancer")])
    paths = write_file_paths(tmp_path / "paths.csv",
                             [("DBT-P00111", "rmlo", "series-odd")])
    out = str(tmp_path / "out")

    with caplog.at_level(logging.WARNING):
        saved, _ = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                             file_paths_csv=paths, output_dir=out)

    assert saved == 1
    assert "the DICOM says DBT-P00999" in caplog.text
    # The case is filed under the patient the inventory names, which the split reads.
    assert str(load(out, "series-odd")["case_id"]) == "DBT-P00111"


def test_an_unannotated_series_is_dropped_without_reading_it(tmp_path, monkeypatch):
    """Pixels cost 7-17 s and ~100 MB per series; the inventory settles this first.

    The join needs no header at all, so an unannotated series is now skipped without
    even the header read the previous version paid for.
    """
    import pydicom

    root = tmp_path / "tcia"
    write_series(str(root), "series-cc", "DBT-P00700", view="rcc")
    write_series(str(root), "series-mlo", "DBT-P00700", view="rmlo")
    # Annotated at the MLO view only; both views are in the inventory.
    boxes = write_boxes(tmp_path / "mlo_only.csv",
                        [("DBT-P00700", "rmlo", 3, "cancer")])
    paths = write_file_paths(tmp_path / "paths.csv", [
        ("DBT-P00700", "rcc", "series-cc"),
        ("DBT-P00700", "rmlo", "series-mlo"),
    ])

    real_dcmread = pydicom.dcmread
    read = []

    def counting_dcmread(path, *args, **kwargs):
        read.append(str(path))
        return real_dcmread(path, *args, **kwargs)

    monkeypatch.setattr("TransformData.pydicom.dcmread", counting_dcmread)
    saved, skipped = preprocess_dbt_with_boxes(root_dir=str(root), boxes_csv=boxes,
                                               file_paths_csv=paths,
                                               output_dir=str(tmp_path / "out"))

    assert (saved, skipped) == (1, 1)
    assert all("series-mlo" in path for path in read), read


# --------------------------------------------------------------------------- #
# The per-view labels, which is the only place an exam is called normal
# --------------------------------------------------------------------------- #

def write_labels(path, rows):
    """A labels CSV with the real schema; ``rows`` are (patient, view, status).

    ``status`` may be ``None`` for a row with no flag set at all, which the real
    tables do not contain and which must not be read as a normal exam.
    """
    pd.DataFrame([
        {"PatientID": patient, "StudyUID": study_of(patient), "View": view,
         "Normal": int(status == "normal"), "Actionable": int(status == "actionable"),
         "Benign": int(status == "benign"), "Cancer": int(status == "cancer")}
        for patient, view, status in rows
    ], columns=["PatientID", "StudyUID", "View",
                "Normal", "Actionable", "Benign", "Cancer"]).to_csv(path, index=False)
    return str(path)


def test_a_patient_is_read_at_its_worst_view(tmp_path):
    """One cancer view makes a cancer exam: the rule the box path already applies."""
    path = write_labels(tmp_path / "labels.csv", [
        ("DBT-P1", "lcc", "normal"), ("DBT-P1", "lmlo", "cancer"),
        ("DBT-P2", "lcc", "normal"), ("DBT-P2", "lmlo", "benign"),
        ("DBT-P3", "lcc", "normal"), ("DBT-P3", "lmlo", "actionable"),
        ("DBT-P4", "lcc", "normal"), ("DBT-P4", "lmlo", "normal"),
    ])
    assert dbt_patient_status(path) == {"DBT-P1": "cancer", "DBT-P2": "benign",
                                       "DBT-P3": "actionable", "DBT-P4": "normal"}


def test_a_row_with_no_flag_is_not_counted_as_normal(tmp_path):
    """An absent status is not a negative; treating it as one inflates specificity."""
    path = write_labels(tmp_path / "blank.csv", [("DBT-P9", "lcc", None)])
    assert dbt_patient_status(path) == {}


def test_several_labels_csvs_pool_into_one_view(tmp_path):
    """The collection publishes one table per split; the statuses are one population."""
    train = write_labels(tmp_path / "train.csv", [("DBT-P1", "lcc", "normal")])
    validation_csv = write_labels(tmp_path / "validation.csv",
                                  [("DBT-P2", "lcc", "cancer")])
    assert dbt_patient_status([train, validation_csv]) == {"DBT-P1": "normal",
                                                          "DBT-P2": "cancer"}


def test_a_labels_csv_missing_a_flag_column_is_refused(tmp_path):
    """Missing "Cancer" would read as "no patient has a cancer" -- an absence that
    looks exactly like a measurement."""
    path = write_labels(tmp_path / "labels.csv", [("DBT-P1", "lcc", "cancer")])
    pd.read_csv(path).drop(columns=["Cancer"]).to_csv(tmp_path / "no_cancer.csv",
                                                      index=False)
    with pytest.raises(ValueError, match="Cancer"):
        dbt_patient_status(str(tmp_path / "no_cancer.csv"))
