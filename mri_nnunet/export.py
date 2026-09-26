"""Export the silver corpus as an nnU-Net v2 ``nnUNet_raw`` dataset (gold).

    nnUNet_raw/Dataset501_DukeDCEBreast/
        dataset.json
        imagesTr/<case>_0000.nii.gz      # channel 0 (pre), _0001 (post2), ...
        labelsTr/<case>.nii.gz           # the pseudo-mask: 0 background, 1 lesion

Files are hard-linked when the two folders share a volume (the corpus is tens of GB and the
export is meant to be redone), copied otherwise. ``dataset.json`` declares every channel as
``noNorm``: they are already normalised inside the organ mask, and nnU-Net's own z-score would
re-centre them on the whole zero-padded array.

Nothing here imports nnunetv2: the layout is the contract, and it is checked against the
library's own naming rules by the tests (``_XXXX`` channel suffix, ``file_ending``).
"""
from __future__ import annotations

import json
import os
import shutil

from . import pipeline


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


def export(silver_dir, raw_dir, settings):
    """Write the dataset for every fully built case. Returns ``(folder, sorted case ids)``."""
    cases = sorted(pipeline.held_cases(silver_dir, settings))
    folder = os.path.join(raw_dir, dataset_folder_name(settings))
    images, labels = os.path.join(folder, "imagesTr"), os.path.join(folder, "labelsTr")
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    # The export is redone after every rebuild: a case no longer held must leave it too, or
    # the folder holds more cases than dataset.json declares.
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
    with open(os.path.join(folder, "dataset.json"), "w", encoding="utf-8") as handle:
        json.dump(dataset_json(settings, len(cases)), handle, indent=2)
    return folder, cases
