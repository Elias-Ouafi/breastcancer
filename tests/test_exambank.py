"""The exam bank: paying decompression once, without losing which slice is whose.

870 exam volumes at 384x384 in float16 come to ~17.6 GB decompressed -- too much to
hold in memory across an epoch, and re-inflating every ``.npz`` on every access is
what already made the DCE-MRI slice bank necessary (``imaging.slicebank``). The exam
bank pays that cost once and resizes further to a size five stride-2 pooling stages
divide evenly. What has to survive that: a slice must still say which patient and
which exam (file) it came from, so a MIL bag is drawn from one exam's own slices and a
patient-level split never crosses a boundary it should not.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from imaging.exambank import ExamBank, build_exam_bank, exam_id_for_path


def write_exam(directory, patient, label, exam="s0", depth=6, size=16, with_label=True):
    """One preprocessed-exam-looking ``.npz``: full frame, a label, no cropping."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(abs(hash((patient, exam))) % 2**31)
    mask = np.zeros((depth, size, size), dtype=np.uint8)
    if label:
        mask[2, 4:8, 4:8] = 1  # a cancer exam here always carries a painted box
    volume = rng.standard_normal((depth, size, size)).astype(np.float16)
    arrays = {
        "volume": volume, "mask": mask, "crop_offset": np.zeros(3, dtype=np.int32),
        "case_id": np.asarray(patient),
    }
    if with_label:
        arrays["label"] = np.asarray(label, dtype=np.uint8)
    path = os.path.join(directory, f"{patient}-{exam}.npz")
    np.savez_compressed(path, **arrays)
    return path


def test_exam_id_is_the_filename_stem():
    assert exam_id_for_path("/a/b/DBT-P1-rmlo.npz") == "DBT-P1-rmlo"


@pytest.mark.skipif(os.name != "nt", reason="a backslash is a legal filename character "
                                            "on POSIX, so there is no directory to strip")
def test_exam_id_strips_a_windows_directory():
    """Only meaningful where the backslash separates directories.

    This assertion used to run everywhere and failed the whole suite on CI's Linux
    runner: there, ``C:\\a\\b\\DBT-P1-rmlo.npz`` is one filename rather than a path,
    so returning it whole is correct rather than a bug. The function is fed paths from
    ``glob`` on the running platform, which never mixes the two conventions.
    """
    assert exam_id_for_path(r"C:\a\b\DBT-P1-rmlo.npz") == "DBT-P1-rmlo"


def test_a_file_without_a_label_is_refused(tmp_path):
    write_exam(str(tmp_path), "DBT-P1", label=0, with_label=False)
    paths = [os.path.join(str(tmp_path), f) for f in os.listdir(str(tmp_path))]

    with pytest.raises(KeyError, match="label"):
        build_exam_bank(paths, str(tmp_path / "bank"), image_size=8)


def test_the_bank_keeps_each_slice_with_its_patient_exam_and_label(tmp_path):
    root = str(tmp_path / "src")
    p1 = write_exam(root, "DBT-P1", label=0, exam="rmlo", depth=4)
    p2 = write_exam(root, "DBT-P2", label=1, exam="lcc", depth=5)

    bank_dir = build_exam_bank([p1, p2], str(tmp_path / "bank"), image_size=8)
    bank = ExamBank(bank_dir)

    assert bank.volumes.shape == (9, 8, 8)
    assert set(bank.unique_exams()) == {"DBT-P1-rmlo", "DBT-P2-lcc"}
    assert bank.exam_label["DBT-P1-rmlo"] == 0
    assert bank.exam_label["DBT-P2-lcc"] == 1
    assert bank.exam_case["DBT-P1-rmlo"] == "DBT-P1"
    assert bank.exam_case["DBT-P2-lcc"] == "DBT-P2"
    assert len(bank.rows_for("DBT-P1-rmlo")) == 4
    assert len(bank.rows_for("DBT-P2-lcc")) == 5
    # No row is shared between the two exams.
    assert not set(bank.rows_for("DBT-P1-rmlo")) & set(bank.rows_for("DBT-P2-lcc"))


def test_has_lesion_reflects_the_mask_not_the_label(tmp_path):
    """A benign exam still paints a box; a normal one paints nothing at all."""
    root = str(tmp_path / "src")
    cancer = write_exam(root, "DBT-P1", label=1, exam="rmlo", depth=4)
    normal = write_exam(root, "DBT-P2", label=0, exam="lcc", depth=4)

    bank = ExamBank(build_exam_bank([cancer, normal], str(tmp_path / "bank"), image_size=8))
    cancer_rows, normal_rows = bank.rows_for("DBT-P1-rmlo"), bank.rows_for("DBT-P2-lcc")

    assert bank.has_lesion[cancer_rows].any()
    assert not bank.has_lesion[normal_rows].any()


def test_a_second_call_with_the_same_request_reuses_the_bank(tmp_path):
    root = str(tmp_path / "src")
    path = write_exam(root, "DBT-P1", label=0)
    bank_dir = str(tmp_path / "bank")

    build_exam_bank([path], bank_dir, image_size=8)
    written_at = os.path.getmtime(os.path.join(bank_dir, "volumes.npy"))

    build_exam_bank([path], bank_dir, image_size=8)  # same request: should not rewrite
    assert os.path.getmtime(os.path.join(bank_dir, "volumes.npy")) == written_at


def test_a_different_image_size_forces_a_rebuild(tmp_path):
    root = str(tmp_path / "src")
    path = write_exam(root, "DBT-P1", label=0)
    bank_dir = str(tmp_path / "bank")

    build_exam_bank([path], bank_dir, image_size=8)
    build_exam_bank([path], bank_dir, image_size=16)
    bank = ExamBank(bank_dir)
    assert bank.volumes.shape[1:] == (16, 16)


def test_force_rebuilds_even_for_the_same_request(tmp_path):
    """A ``force=True`` rebuild is not skipped by the cache check, and stays correct."""
    root = str(tmp_path / "src")
    path = write_exam(root, "DBT-P1", label=1, depth=4)
    bank_dir = str(tmp_path / "bank")

    build_exam_bank([path], bank_dir, image_size=8)
    build_exam_bank([path], bank_dir, image_size=8, force=True)
    bank = ExamBank(bank_dir)
    assert bank.volumes.shape == (4, 8, 8)
    assert bank.exam_label["DBT-P1-s0"] == 1
