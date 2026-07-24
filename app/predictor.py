"""Pluggable prediction backend for the MRI cancer-detection web app.

The web layer (``app/server.py``) never talks to a model directly. It talks to a
:class:`Predictor`. That indirection is the whole point of this file: right now the
app ships with :class:`MockPredictor` so the UI is fully usable *before* any model is
wired in, and swapping in the real AI later is a **one-line change** in
:func:`get_predictor` (or one environment variable).

The result contract is aligned with ``inference.predict_dbt`` so the future
:class:`DbtUNetPredictor` can forward its output almost verbatim:

    {
        "lesion_detected": bool,      # cancer / lesion present?
        "confidence": float,          # 0..1 detection confidence
        "best_slice": int | None,     # slice index of the finding (imaging)
        "box_xywh": [x, y, w, h] | None,
        "n_slices": int | None,
        "backend": str,               # which predictor answered
    }
"""
from __future__ import annotations

import os
import random
from typing import Optional


class Predictor:
    """Interface every prediction backend implements.

    A backend receives the path to an uploaded study on disk and returns the result
    dict documented in the module docstring.
    """

    name = "base"

    def predict(self, file_path: str) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class MockPredictor(Predictor):
    """Deterministic-ish stand-in used until the real model is connected.

    It does **not** look at the pixels — it fabricates a plausible result so the whole
    upload -> predict -> display flow can be exercised end to end. The verdict is seeded
    from the file name so re-uploading the same file gives a stable answer during demos.
    """

    name = "mock"

    def predict(self, file_path: str) -> dict:
        seed = os.path.basename(file_path)
        rng = random.Random(seed)
        confidence = round(rng.uniform(0.05, 0.95), 3)
        detected = confidence >= 0.5
        n_slices = rng.randint(40, 120)
        return {
            "lesion_detected": detected,
            "confidence": confidence,
            "best_slice": rng.randint(0, n_slices - 1) if detected else None,
            "box_xywh": [rng.randint(20, 120), rng.randint(20, 120), 48, 48]
            if detected
            else None,
            "n_slices": n_slices,
            "backend": self.name,
        }


class DbtUNetPredictor(Predictor):
    """Real backend: the trained 2D U-Net via ``inference.predict_dbt``.

    Left un-wired by default. To connect the AI, set ``MRI_APP_BACKEND=unet`` (the
    checkpoint at ``results/unet_best.pt`` must exist and the upload must be a
    preprocessed ``.npz`` volume). Everything else in the app stays the same.
    """

    name = "unet"

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint

    def predict(self, file_path: str) -> dict:
        from inference import DEFAULT_UNET_CKPT, predict_dbt

        result = predict_dbt(file_path, checkpoint=self.checkpoint or DEFAULT_UNET_CKPT)
        result.setdefault("backend", self.name)
        return result


class DceMriUNetPredictor(Predictor):
    """Real backend: the trained 2D U-Net via ``inference.predict_dce_mri``.

    Trained on Duke-Breast-Cancer-MRI subtraction volumes (post minus pre-contrast),
    distinct from the DBT U-Net above. Set ``MRI_APP_BACKEND=dce_mri`` to connect it
    (the checkpoint at ``results_mri/unet_best.pt`` must exist). The upload must be a
    preprocessed ``.npz`` (see ``TransformData.preprocess_dce_mri_with_boxes``) --
    unlike DBT there is no single-file raw-DICOM path, since DCE-MRI needs two whole
    series (pre + post-contrast) to build the subtraction.
    """

    name = "dce_mri"

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint

    def predict(self, file_path: str) -> dict:
        from inference import DEFAULT_MRI_UNET_CKPT, predict_dce_mri

        result = predict_dce_mri(file_path, checkpoint=self.checkpoint or DEFAULT_MRI_UNET_CKPT)
        result.setdefault("backend", self.name)
        return result


def get_predictor() -> Predictor:
    """Return the active backend.

    Selection is driven by the ``MRI_APP_BACKEND`` env var (``mock`` by default).
    THIS is the single place to change when connecting the real AI.
    """
    backend = os.environ.get("MRI_APP_BACKEND", "mock").lower()
    if backend == "unet":
        return DbtUNetPredictor()
    if backend == "dce_mri":
        return DceMriUNetPredictor()
    return MockPredictor()
