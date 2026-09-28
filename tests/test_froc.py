"""The FROC evaluation: 3D components, the hit criterion, the curve and its interval.

Synthetic volumes only -- no corpus, no SimpleITK, no GPU -- so this runs in CI.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imaging import froc  # noqa: E402


def cube(volume, centre, half, value=1.0):
    z, y, x = centre
    volume[z - half:z + half + 1, y - half:y + half + 1, x - half:x + half + 1] = value
    return volume


# --------------------------------------------------------------------------- components

def test_two_separated_blobs_are_two_components():
    mask = np.zeros((20, 20, 20), dtype=bool)
    cube(mask, (5, 5, 5), 1, True)
    cube(mask, (15, 15, 15), 1, True)
    labels, count = froc.label_components(mask)
    assert count == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_a_blob_joined_only_in_3d_is_one_component():
    """Two squares on neighbouring slices touch through z; a 2D pass would call them two."""
    mask = np.zeros((6, 10, 10), dtype=bool)
    mask[2, 3:6, 3:6] = True
    mask[3, 3:6, 3:6] = True
    _, count = froc.label_components(mask)
    assert count == 1


def test_diagonal_neighbours_are_not_connected():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1, 1, 1] = True
    mask[2, 2, 2] = True
    _, count = froc.label_components(mask)
    assert count == 2


def test_a_long_snake_is_one_component():
    """The label propagation must survive a component longer than a couple of rounds."""
    mask = np.zeros((4, 4, 60), dtype=bool)
    mask[1, 1, :] = True
    _, count = froc.label_components(mask)
    assert count == 1


def test_empty_volume_has_no_component():
    labels, count = froc.label_components(np.zeros((5, 5, 5), dtype=bool))
    assert count == 0 and labels.sum() == 0


def test_a_full_volume_is_one_component():
    _, count = froc.label_components(np.ones((7, 7, 7), dtype=bool))
    assert count == 1


def test_a_2d_array_is_refused():
    with pytest.raises(ValueError, match="3D"):
        froc.label_components(np.ones((5, 5), dtype=bool))


# --------------------------------------------------------------------------- candidates

def test_candidate_score_is_the_peak_probability():
    probability = np.zeros((20, 20, 20))
    cube(probability, (10, 10, 10), 2, 0.4)
    probability[10, 10, 10] = 0.92
    found = froc.candidates(probability, detection_threshold=0.1, min_voxels=1)
    assert len(found) == 1
    assert found[0]["score"] == pytest.approx(0.92)


def test_specks_below_min_voxels_are_dropped():
    probability = np.zeros((20, 20, 20))
    probability[3, 3, 3] = 0.99           # one voxel
    cube(probability, (12, 12, 12), 2, 0.8)   # 125 voxels
    assert len(froc.candidates(probability, min_voxels=10)) == 1


def test_lesion_volume_uses_the_spacing():
    mask = np.zeros((20, 20, 20), dtype=bool)
    cube(mask, (10, 10, 10), 1, True)     # 27 voxels
    found = froc.lesions(mask, spacing=(2.0, 1.0, 1.0))
    assert found[0]["volume_mm3"] == pytest.approx(54.0)


# --------------------------------------------------------------------------- matching

def make_case(lesion_centres, candidate_centres, scores, shape=(40, 40, 40), half=3):
    label = np.zeros(shape, dtype=np.uint8)
    for centre in lesion_centres:
        cube(label, centre, half, 1)
    probability = np.zeros(shape)
    for centre, score in zip(candidate_centres, scores):
        cube(probability, centre, 2, score)
    return probability, label


def test_a_candidate_inside_the_box_is_a_hit_and_none_is_a_false_positive():
    probability, label = make_case([(10, 10, 10)], [(10, 10, 10)], [0.9])
    record = froc.case_record(probability, label, min_voxels=1)
    assert record["n_lesions"] == 1
    assert record["lesion_scores"] == [0.9]
    assert record["fp_scores"] == []


def test_a_candidate_elsewhere_is_a_false_positive_and_the_lesion_is_missed():
    probability, label = make_case([(10, 10, 10)], [(30, 30, 30)], [0.8])
    record = froc.case_record(probability, label, min_voxels=1)
    assert record["lesion_scores"] == [None]
    assert record["fp_scores"] == [0.8]


def test_a_second_candidate_on_the_same_lesion_is_not_a_false_positive():
    """Two findings on one lesion is the same finding twice, not an error (FROC convention)."""
    probability, label = make_case([(10, 10, 10)], [(10, 10, 10), (10, 10, 12)], [0.9, 0.7])
    record = froc.case_record(probability, label, min_voxels=1)
    assert record["fp_scores"] == []
    assert record["lesion_scores"] == [0.9]   # the best candidate that hit it


def test_the_lesion_keeps_the_best_score_among_its_candidates():
    probability, label = make_case([(10, 10, 10)], [(10, 10, 10)], [0.3])
    probability[10, 10, 10] = 0.95
    record = froc.case_record(probability, label, min_voxels=1)
    assert record["lesion_scores"] == [pytest.approx(0.95)]


def test_overlap_criterion_accepts_what_the_centroid_criterion_refuses():
    """A candidate straddling the lesion edge: its centre is outside, its voxels are not."""
    label = np.zeros((40, 40, 40), dtype=np.uint8)
    cube(label, (20, 20, 20), 3, 1)                 # box spans 17..23
    probability = np.zeros((40, 40, 40))
    cube(probability, (20, 20, 27), 4, 0.8)         # spans 23..31 on x, centre at 27
    strict = froc.case_record(probability, label, min_voxels=1, criterion="centroid_in_box")
    loose = froc.case_record(probability, label, min_voxels=1, criterion="overlap")
    assert strict["lesion_scores"] == [None] and strict["fp_scores"] == [0.8]
    assert loose["lesion_scores"] == [0.8] and loose["fp_scores"] == []


def test_unknown_criterion_is_refused():
    with pytest.raises(ValueError, match="criterion"):
        froc.match([], [], criterion="vibes")


def test_two_lesions_are_scored_separately():
    probability, label = make_case([(10, 10, 10), (30, 30, 30)], [(10, 10, 10)], [0.9])
    record = froc.case_record(probability, label, min_voxels=1)
    assert record["n_lesions"] == 2
    assert record["lesion_scores"] == [0.9, None]


# --------------------------------------------------------------------------- the curve

def synthetic_records(n_exams=20, detected=0.5, fp_per_exam=2, seed=0):
    """``n_exams`` one-lesion records; a share ``detected`` are found, each with some FPs."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_exams):
        found = i < round(detected * n_exams)
        records.append({
            "case_id": f"case_{i:03d}",
            "n_lesions": 1,
            "lesion_scores": [float(rng.uniform(0.6, 1.0)) if found else None],
            "fp_scores": sorted(rng.uniform(0.1, 0.55, size=fp_per_exam).tolist(), reverse=True),
        })
    return records


def test_curve_is_monotone_in_both_axes():
    points = froc.curve(synthetic_records())
    sens = np.asarray(points["sensitivity"])
    fps = np.asarray(points["fp_per_exam"])
    assert np.all(np.diff(sens) >= -1e-12)
    assert np.all(np.diff(fps) >= -1e-12)
    assert points["n_exams"] == 20 and points["n_lesions"] == 20


def test_a_perfect_model_reaches_full_sensitivity_with_no_false_positive():
    records = [{"case_id": f"c{i}", "n_lesions": 1, "lesion_scores": [0.99], "fp_scores": []}
               for i in range(10)]
    assert froc.sensitivity_at(records, (0.0, 1.0)) == {"0.0": 1.0, "1.0": 1.0}


def test_a_model_that_finds_nothing_scores_zero():
    records = [{"case_id": f"c{i}", "n_lesions": 1, "lesion_scores": [None], "fp_scores": [0.4]}
               for i in range(10)]
    assert froc.sensitivity_at(records, (1.0,))["1.0"] == 0.0


def test_sensitivity_is_read_at_the_allowed_false_positive_rate():
    """Every lesion is found, but only below 0.5: at 0 FP/exam the model finds nothing."""
    records = [{"case_id": f"c{i}", "n_lesions": 1, "lesion_scores": [0.4],
                "fp_scores": [0.9]} for i in range(10)]
    got = froc.sensitivity_at(records, (0.0, 1.0))
    assert got["0.0"] == 0.0
    assert got["1.0"] == 1.0


def test_more_allowed_false_positives_never_lowers_sensitivity():
    records = synthetic_records(n_exams=30, detected=0.7, fp_per_exam=3)
    got = froc.sensitivity_at(records, (0.5, 1.0, 2.0, 4.0))
    values = [got[k] for k in ("0.5", "1.0", "2.0", "4.0")]
    assert values == sorted(values)


def test_curve_refuses_an_empty_set():
    with pytest.raises(ValueError, match="no case"):
        froc.curve([])


# --------------------------------------------------------------------------- intervals

def test_bootstrap_brackets_the_point_estimate_and_resamples_patients():
    records = synthetic_records(n_exams=40, detected=0.6, fp_per_exam=2)
    got = froc.bootstrap_sensitivity(records, (2.0,), n_resamples=300, seed=1)["2.0"]
    assert got["lo"] <= got["sensitivity"] <= got["hi"]
    assert got["lo"] < got["hi"]      # a degenerate interval would mean nothing was resampled
    assert got["n_usable"] > 0


def test_bootstrap_interval_narrows_with_more_patients():
    """More patients, same behaviour: the interval must shrink, or it is not measuring one."""
    small = froc.bootstrap_sensitivity(synthetic_records(20, 0.6, 2, seed=2), (2.0,),
                                       n_resamples=300, seed=1)["2.0"]
    large = froc.bootstrap_sensitivity(synthetic_records(200, 0.6, 2, seed=2), (2.0,),
                                       n_resamples=300, seed=1)["2.0"]
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])


def test_bootstrap_is_reproducible_with_the_same_seed():
    records = synthetic_records(n_exams=25)
    first = froc.bootstrap_sensitivity(records, (1.0,), n_resamples=200, seed=3)
    again = froc.bootstrap_sensitivity(records, (1.0,), n_resamples=200, seed=3)
    assert first == again


# --------------------------------------------------------------------------- by lesion size

def test_sensitivity_is_split_by_lesion_size():
    records = [
        {"case_id": "a", "n_lesions": 2, "lesion_scores": [0.9, None], "fp_scores": []},
        {"case_id": "b", "n_lesions": 2, "lesion_scores": [0.8, None], "fp_scores": []},
    ]
    sizes = [[20000.0, 500.0], [20000.0, 500.0]]
    got = froc.sensitivity_by_lesion_size(records, sizes, fp_rate=1.0, edges=(1000.0,))
    assert got[">= 1000 mm3"] == {"sensitivity": 1.0, "n_lesions": 2}
    assert got["0-1000 mm3"] == {"sensitivity": 0.0, "n_lesions": 2}


# --------------------------------------------------------------------------- end to end

def test_a_whole_exam_goes_from_volumes_to_an_interval():
    records = []
    for i in range(12):
        probability, label = make_case([(10, 10, 10)],
                                       [(10, 10, 10)] if i % 2 == 0 else [(32, 32, 32)],
                                       [0.85])
        records.append(froc.case_record(probability, label, min_voxels=1, case_id=f"case_{i}"))
    got = froc.bootstrap_sensitivity(records, (1.0,), n_resamples=200, seed=0)["1.0"]
    assert got["sensitivity"] == pytest.approx(0.5)
    assert got["lo"] < 0.5 < got["hi"]


# --------------------------------------------------------------- reading predictions

def test_the_lesion_channel_is_taken_from_a_saved_probability_array():
    """nnU-Net writes (class, z, y, x), background first.

    Reading channel 0 would score the FROC on the background and report a model that finds
    nothing -- the failure would look like a bad model, not like a bad reader.
    """
    background = np.zeros((4, 4, 4))
    lesion = np.ones((4, 4, 4)) * 0.9
    stacked = np.stack([background, lesion])
    assert froc.foreground_probability(stacked).max() == pytest.approx(0.9)
    assert froc.foreground_probability(lesion).max() == pytest.approx(0.9)   # already 3D


def test_a_probability_array_without_the_lesion_class_is_refused():
    with pytest.raises(ValueError, match="no class 1"):
        froc.foreground_probability(np.zeros((1, 4, 4, 4)))
    with pytest.raises(ValueError, match="expected a 3D volume"):
        froc.foreground_probability(np.zeros((4, 4)))


def test_nnunet_npz_predictions_are_read_with_their_own_key(tmp_path):
    path = os.path.join(tmp_path, "case.npz")
    lesion = np.zeros((4, 4, 4))
    lesion[1, 1, 1] = 0.77
    np.savez(path, probabilities=np.stack([1 - lesion, lesion]))
    volume, _ = froc._read_volume(path)
    assert volume.shape == (4, 4, 4)
    assert volume.max() == pytest.approx(0.77)


def test_an_nnunet_prediction_folder_gives_one_pair_per_case(tmp_path):
    """Two files per case (.nii.gz labels and .npz probabilities), plus a .pkl: one exam each.

    Counting both would score every exam twice, the second time on a map of 0s and 1s whose
    threshold sweep says nothing.
    """
    predictions = tmp_path / "pred"
    labels = tmp_path / "labelsTs"
    predictions.mkdir()
    labels.mkdir()
    for case in ("Breast_MRI_007", "Breast_MRI_042"):
        (predictions / f"{case}.nii.gz").write_bytes(b"x")
        (predictions / f"{case}.npz").write_bytes(b"x")
        (predictions / f"{case}.pkl").write_bytes(b"x")
        (labels / f"{case}.nii.gz").write_bytes(b"x")
    (predictions / "Breast_MRI_099.npz").write_bytes(b"x")      # prediction without a label

    pairs = list(froc._pairs(str(predictions), str(labels)))

    assert [case for case, _, _ in pairs] == ["Breast_MRI_007", "Breast_MRI_042"]
    assert all(prediction.endswith(".npz") for _, prediction, _ in pairs)   # scores, not labels
    assert all(label.endswith(".nii.gz") for _, _, label in pairs)
