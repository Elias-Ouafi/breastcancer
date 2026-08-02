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
        "slice_preselected": bool,    # was the slice pinned by a human, not found?
        "confidence": float,          # 0..1 detection confidence
        "best_slice": int | None,     # slice index of the finding (imaging)
        "box_xywh": [x, y, w, h] | None,
        "n_slices": int | None,
        "inference_ms": float | None, # scoring time, model load excluded
        "backend": str,               # which predictor answered
    }
"""
from __future__ import annotations

import os
import random
import time
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
            "slice_preselected": False,  # nothing is selected: no pixel is ever read
            "confidence": confidence,
            "best_slice": rng.randint(0, n_slices - 1) if detected else None,
            "box_xywh": [rng.randint(20, 120), rng.randint(20, 120), 48, 48]
            if detected
            else None,
            "n_slices": n_slices,
            "inference_ms": None,  # nothing was computed, so there is nothing to time
            "backend": self.name,
        }


class _CachedUNetPredictor(Predictor):
    """Shared plumbing for the two U-Net backends: load the weights once, and time.

    Without the cache every request paid a fresh ``load_unet`` -- reading a 31 MB
    checkpoint off disk and pushing it to the GPU -- which dominated the response and
    made any timing shown to the user meaningless. The model is stateless at
    inference (``eval()``, no grad), so one instance can serve every request.

    ``inference_ms`` deliberately covers scoring only, not the load: it is the number
    that describes the product ("how long does an exam take"), whereas the one-off
    load is a startup cost the first request happens to pay.
    """

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint
        self._model = None
        self._device = None

    def _ensure_model(self, default_checkpoint: str):
        if self._model is None:
            from inference import load_unet

            self._model, self._device = load_unet(self.checkpoint or default_checkpoint)
        return self._model, self._device

    @staticmethod
    def _timed(fn):
        """Run ``fn`` and return its result with ``inference_ms`` filled in."""
        start = time.perf_counter()
        result = fn()
        result["inference_ms"] = round(1000 * (time.perf_counter() - start), 1)
        return result


class DbtUNetPredictor(_CachedUNetPredictor):
    """Real backend: the trained 2D U-Net via ``inference.predict_dbt``.

    Left un-wired by default. To connect the AI, set ``MRI_APP_BACKEND=unet`` (the
    checkpoint at ``results/unet_best.pt`` must exist and the upload must be a
    preprocessed ``.npz`` volume). Everything else in the app stays the same.
    """

    name = "unet"

    def predict(self, file_path: str) -> dict:
        from inference import DEFAULT_UNET_CKPT, predict_dbt

        model, device = self._ensure_model(DEFAULT_UNET_CKPT)
        result = self._timed(lambda: predict_dbt(file_path, model=model, device=device))
        result.setdefault("backend", self.name)
        return result


class DceMriUNetPredictor(_CachedUNetPredictor):
    """Real backend: the trained 2D U-Net via ``inference.predict_dce_mri``.

    Trained on Duke-Breast-Cancer-MRI subtraction volumes (post minus pre-contrast),
    distinct from the DBT U-Net above. Set ``MRI_APP_BACKEND=dce_mri`` to connect it
    (the checkpoint at ``results_mri_p2_negfix/unet_best.pt`` must exist -- second
    post-contrast pass, scratch GroupNorm U-Net, 186-patient full-frame sample; see
    plan.md §4.1). The upload must be a preprocessed ``.npz`` (see
    ``TransformData.preprocess_dce_mri_with_boxes``) -- unlike DBT there is no
    single-file raw-DICOM path, since DCE-MRI needs two whole series (pre +
    post-contrast) to build the subtraction.

    KNOWN LIMITATION (plan.md §4.2): automatic slice selection does not yet work on
    a raw full-volume upload -- verified 0/186 on held-out patients, the model's
    confidence saturates on essentially every slice. It segments well once shown the
    right slice, it just cannot find that slice on its own yet. Use
    ``TransformData.make_demo_case`` to pin a verified-good ``forced_slice`` for
    reliable demo cases (see ``demo_cases/``) until that ranking problem is fixed.
    """

    name = "dce_mri"

    def predict(self, file_path: str) -> dict:
        from inference import DEFAULT_MRI_UNET_CKPT, predict_dce_mri

        model, device = self._ensure_model(DEFAULT_MRI_UNET_CKPT)
        result = self._timed(lambda: predict_dce_mri(file_path, model=model, device=device))
        result.setdefault("backend", self.name)
        return result


_PREDICTORS: dict = {}


def get_predictor() -> Predictor:
    """Return the active backend.

    Selection is driven by the ``MRI_APP_BACKEND`` env var (``mock`` by default).
    THIS is the single place to change when connecting the real AI.

    Instances are memoised per backend name. They are stateless apart from the
    lazily-loaded weights, and the server calls this on every request (including
    twice per ``/predict``), so returning a fresh object each time would throw the
    model cache away and reload 31 MB of weights per page view.
    """
    backend = os.environ.get("MRI_APP_BACKEND", "mock").lower()
    if backend not in _PREDICTORS:
        if backend == "unet":
            _PREDICTORS[backend] = DbtUNetPredictor()
        elif backend == "dce_mri":
            _PREDICTORS[backend] = DceMriUNetPredictor()
        else:
            _PREDICTORS[backend] = MockPredictor()
    return _PREDICTORS[backend]
