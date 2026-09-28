"""Export the silver corpus as an nnU-Net v2 ``nnUNet_raw`` dataset (gold).

    nnUNet_raw/Dataset501_DukeDCEBreast/
        dataset.json
        splits_final.json                # the 5 folds, stratified by scanner (mri_nnunet/splits.py)
        imagesTr/<case>_0000.nii.gz      # channel 0 (pre), _0001 (post2), ...
        labelsTr/<case>.nii.gz           # the pseudo-mask: 0 background, 1 lesion
        imagesTs/<case>_0000.nii.gz      # the held-out test patients, same layout
        labelsTs/<case>.nii.gz

Files are hard-linked when the two folders share a volume (the corpus is tens of GB and the
export is meant to be redone), copied otherwise. ``dataset.json`` declares every channel as
``noNorm``: they are already normalised inside the organ mask, and nnU-Net's own z-score would
re-centre them on the whole zero-padded array.

The test patients go to ``imagesTs``, **not** ``imagesTr``: nnU-Net trains on everything in
``imagesTr`` and fits its post-processing on the cross-validation, so a test patient left there
would be seen by the very choices its number is meant to judge. ``numTraining`` counts the
training set only, which is what nnU-Net checks against ``imagesTr``.

Nothing here imports nnunetv2: the layout is the contract, and it is checked against the
library's own naming rules by the tests (``_XXXX`` channel suffix, ``file_ending``).
"""
from __future__ import annotations

import json
import os
import shutil

from . import pipeline
from . import splits as splits_module


def _link(source, destination):
    if os.path.exists(destination):
        os.remove(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def dataset_json(settings, n_cases):
    """The ``dataset.json`` nnU-Net v2 requires: channels, labels, count, file ending."""
    norm = settings["export"]["channel_normalization"]
    return {
        "channel_names": {str(i): norm for i, _ in enumerate(settings["channels"])},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": int(n_cases),
        "file_ending": ".nii.gz",
        "name": settings["export"]["dataset_name"],
        "description": ("Duke-Breast-Cancer-MRI, DCE. Pseudo-masks: ellipsoids inscribed in the "
                        "published bounding boxes (not manual segmentations)."),
        "channel_meaning": {str(i): c["name"] for i, c in enumerate(settings["channels"])},
        "licence": "CC BY-NC 4.0",
    }


def dataset_folder_name(settings):
    return f"Dataset{int(settings['export']['dataset_id']):03d}_{settings['export']['dataset_name']}"


def _write_cases(cases, images, labels, silver_dir, settings):
    """Link one group of cases into an ``images*``/``labels*`` pair, emptied of anything else.

    The export is redone after every rebuild, so a case that is no longer held -- or that
    moved from training to test -- must leave the folder it was in, or the dataset would hold
    more cases than ``dataset.json`` declares and nnU-Net would train on a patient the split
    calls held out.
    """
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    held = set(cases)
    suffix = ".nii.gz"
    for directory, case_of in ((images, lambda stem: stem.rsplit("_", 1)[0]),   # <case>_0000
                               (labels, lambda stem: stem)):                     # <case>
        for name in os.listdir(directory):
            if not name.endswith(suffix) or case_of(name[:-len(suffix)]) not in held:
                os.remove(os.path.join(directory, name))
    for pid in cases:
        outputs = pipeline._outputs(settings, silver_dir, pid)
        for i, path in enumerate(outputs[:len(settings["channels"])]):
            _link(path, os.path.join(images, f"{pid}_{i:04d}.nii.gz"))
        _link(outputs[-2], os.path.join(labels, f"{pid}.nii.gz"))


def export(silver_dir, raw_dir, settings):
    """Write the dataset for every fully built case. Returns ``(folder, sorted training ids)``.

    Uses the split written by ``mri_nnunet splits``: its test patients go to ``imagesTs`` and
    the rest to ``imagesTr``, and its folds are copied out as ``splits_final.json``. Without a
    split file every case is exported as training, which is the old behaviour and is reported
    as such by the caller.
    """
    cases = sorted(pipeline.held_cases(silver_dir, settings))
    assignment = splits_module.read(silver_dir)
    if assignment is not None:
        splits_module.check(assignment)
        known = set(assignment["test"]) | set(assignment["trainval"])
        missing = [c for c in cases if c not in known]
        if missing:
            raise ValueError(f"{len(missing)} built case(s) are in no split ({missing[:3]}...); "
                             "re-run `python -m mri_nnunet splits` after a rebuild")
        test = [c for c in cases if c in set(assignment["test"])]
    else:
        test = []
    training = [c for c in cases if c not in set(test)]

    folder = os.path.join(raw_dir, dataset_folder_name(settings))
    _write_cases(training, os.path.join(folder, "imagesTr"), os.path.join(folder, "labelsTr"),
                 silver_dir, settings)
    if test or assignment is not None:
        _write_cases(test, os.path.join(folder, "imagesTs"), os.path.join(folder, "labelsTs"),
                     silver_dir, settings)
    with open(os.path.join(folder, "dataset.json"), "w", encoding="utf-8") as handle:
        json.dump(dataset_json(settings, len(training)), handle, indent=2)
    if assignment is not None:
        # nnU-Net reads this from nnUNet_preprocessed/<dataset>/; it is written here, beside
        # the data it describes, and copied there by the training command.
        with open(os.path.join(folder, splits_module.NNUNET_SPLITS_FILENAME), "w",
                  encoding="utf-8") as handle:
            json.dump(splits_module.nnunet_splits(assignment), handle, indent=2)
    return folder, training
