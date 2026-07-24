"""Reusable inference layer for the breast-cancer project.

Two entry points, one per pipeline, so a demo/UI can get a prediction in one call
without re-running the batch training scripts:

* :func:`predict_tabular` — scores a single Wisconsin 30-feature record with the
  persisted Spark ``PipelineModel`` (see ``train_tabular_model.py``). Returns the
  predicted diagnosis and, for the logistic model, a malignancy probability.
* :func:`predict_dbt` — runs the trained 2D U-Net (``results/unet_best.pt``) over a
  preprocessed DBT ``.npz`` volume (or a raw volume array) and returns the localised
  lesion: best slice, bounding box, and a detection confidence.

Neither entry point retrains anything; both load saved artefacts. The tabular path
needs a JVM (PySpark); the imaging path needs only ``torch`` + ``numpy``.
"""
from __future__ import annotations

import io
import json
import os
from typing import Mapping, Sequence, Union

import numpy as np

# --------------------------------------------------------------------------- #
# Tabular inference (Wisconsin, Spark MLlib)
# --------------------------------------------------------------------------- #

DEFAULT_TABULAR_DIR = os.path.join("results", "tabular_model")


def _load_tabular_metadata(model_dir):
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No tabular model metadata at {meta_path!r}. "
            "Fit and persist the model first: python train_tabular_model.py"
        )
    with open(meta_path) as f:
        return json.load(f)


def _order_features(features, feature_order):
    """Return the feature values as a list in ``feature_order``.

    Accepts a ``{name: value}`` mapping (order-independent, keys validated) or a plain
    sequence already in ``feature_order``.
    """
    if isinstance(features, Mapping):
        missing = [c for c in feature_order if c not in features]
        extra = [c for c in features if c not in feature_order]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        if extra:
            raise ValueError(f"Unexpected features: {extra}")
        return [float(features[c]) for c in feature_order]

    values = list(features)
    if len(values) != len(feature_order):
        raise ValueError(
            f"Expected {len(feature_order)} feature values, got {len(values)}. "
            f"Order must be: {feature_order}"
        )
    return [float(v) for v in values]


def predict_tabular(features: Union[Mapping[str, float], Sequence[float]],
                    model_dir: str = DEFAULT_TABULAR_DIR):
    """Score one Wisconsin record with the persisted tabular pipeline.

    Parameters
    ----------
    features : mapping or sequence
        The 30 diagnostic features, either as a ``{feature_name: value}`` mapping or a
        sequence in the persisted feature order (see ``metadata.json``).
    model_dir : str
        Directory holding ``pipeline_model/`` and ``metadata.json``.

    Returns
    -------
    dict
        ``{"prediction": 0.0|1.0, "diagnosis": "Benign"|"Malignant",
        "malignant_probability": float|None}``. The probability is ``None`` when the
        served model is Linear SVM (no probability output).
    """
    from pyspark.ml import PipelineModel
    from pyspark.sql.types import DoubleType, StructField, StructType

    from TransformData import _get_spark

    meta = _load_tabular_metadata(model_dir)
    feature_order = meta["feature_order"]
    values = _order_features(features, feature_order)

    spark = _get_spark()
    model = PipelineModel.load(os.path.join(model_dir, "pipeline_model"))

    schema = StructType([StructField(c, DoubleType(), True) for c in feature_order])
    sdf = spark.createDataFrame([tuple(values)], schema=schema)
    row = model.transform(sdf).select("prediction", *(
        ["probability"] if meta.get("produces_probability") else []
    )).head()

    prediction = float(row["prediction"])
    malignant_probability = None
    if meta.get("produces_probability"):
        # probability is a DenseVector [P(benign), P(malignant)]; malignant is label 1.
        malignant_probability = float(row["probability"][1])

    return {
        "prediction": prediction,
        "diagnosis": meta["label_map"][str(prediction)],
        "malignant_probability": malignant_probability,
    }


# --------------------------------------------------------------------------- #
# Imaging inference (DBT lesion localisation, 2D U-Net / PyTorch)
# --------------------------------------------------------------------------- #

DEFAULT_UNET_CKPT = os.path.join("results", "unet_best.pt")


def load_unet(checkpoint: str = DEFAULT_UNET_CKPT, base: int = 32, device=None,
             architecture: str = "scratch", encoder_name: str = "resnet34"):
    """Load the trained U-Net in eval mode. Returns ``(model, device)``.

    ``architecture``/``base``/``encoder_name`` must match what the checkpoint was
    trained with (``imaging.train``'s ``--architecture``, ``--base-channels``,
    ``--encoder-name``) since they determine the state_dict's layer shapes/keys.
    ``encoder_weights`` is not needed here: the checkpoint overwrites whatever the
    encoder was initialised with.
    """
    import torch

    from imaging.unet import build_model

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No U-Net checkpoint at {checkpoint!r}. Train it first: "
            "python -m imaging.train --data-dir preprocessed_data"
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(architecture=architecture, base_channels=base,
                        encoder_name=encoder_name, encoder_weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model, device


def load_dbt_dicom(path):
    """Read a single (possibly multi-frame) DBT DICOM file into a normalised volume.

    BCS-DBT series are stored as one multi-frame DICOM file per view (see
    ``TransformData.preprocess_dbt_with_boxes``, which reads only the first ``.dcm``
    in a series folder), so a single uploaded file is enough to reconstruct the full
    ``(depth, H, W)`` stack. Intensities are normalised with
    ``TransformData.normalize_intensity`` — the same convention used to build the
    training data — so a raw upload is scored on the distribution the model saw at
    training time, not the raw pixel values.
    """
    import pydicom

    from TransformData import normalize_intensity

    ds = pydicom.dcmread(path)
    volume = ds.pixel_array.astype(np.float32)
    if volume.ndim == 2:
        volume = volume[None]
    return normalize_intensity(volume)


def _bounding_box(binary_mask):
    """Axis-aligned box ``(x, y, w, h)`` around the True pixels, or ``None`` if empty."""
    ys, xs = np.nonzero(binary_mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _localize_lesion(vol, model, device, image_size=256, threshold=0.5, crop_offset=(0, 0, 0)):
    """Shared slice-scan loop behind :func:`predict_dbt` and :func:`predict_dce_mri`.

    Scores every axial slice of ``vol`` (a ``(depth, H, W)`` z-normalised array) with
    ``model``, keeps the slice with the highest lesion probability, and returns its
    bounding box both in the given volume's own indexing and mapped back to the
    original (uncropped) frame via ``crop_offset`` (as stored by
    ``TransformData.save_preprocessed``).
    """
    import torch
    import torch.nn.functional as F

    depth, orig_h, orig_w = vol.shape
    best_conf, best_slice, best_prob_small = -1.0, 0, None

    with torch.no_grad():
        for z in range(depth):
            img = torch.from_numpy(vol[z])[None, None].to(device)  # (1,1,H,W)
            img = F.interpolate(img, size=(image_size, image_size),
                                mode="bilinear", align_corners=False)
            prob = torch.sigmoid(model(img))
            conf = float(prob.max())
            if conf > best_conf:
                best_conf, best_slice, best_prob_small = conf, z, prob

        # Only the winning slice needs the (potentially large) native-resolution
        # map, since only its box is returned -- upsampling every slice to native
        # resolution dominated runtime on full-frame (uncropped) uploads for no
        # benefit (the max is already known from the small-scale map).
        best_prob_map = F.interpolate(best_prob_small, size=(orig_h, orig_w),
                                      mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    lesion_detected = best_conf >= threshold
    box = _bounding_box(best_prob_map >= threshold) if lesion_detected else None

    box_full = None
    if box is not None:
        oz, oy, ox = crop_offset
        box_full = (box[0] + ox, box[1] + oy, box[2], box[3])

    return {
        "lesion_detected": lesion_detected,
        "confidence": best_conf,
        "best_slice": best_slice + crop_offset[0],
        "box_xywh": box,
        "box_full_frame_xywh": box_full,
        "n_slices": depth,
    }


def _load_volume_and_offset(volume, raw_loader, raw_extensions):
    """Resolve ``volume`` (path or array) to a ``(vol, crop_offset)`` pair.

    ``.npz`` paths use their ``volume``/``crop_offset`` keys (as written by
    ``TransformData.save_preprocessed``); paths ending in ``raw_extensions`` go
    through ``raw_loader``; anything else is treated as an already-loaded array.
    """
    crop_offset = (0, 0, 0)
    if isinstance(volume, str):
        lower = volume.lower()
        if lower.endswith(".npz"):
            with np.load(volume) as data:
                vol = data["volume"].astype(np.float32)
                if "crop_offset" in data.files:
                    crop_offset = tuple(int(v) for v in data["crop_offset"])
        elif lower.endswith(raw_extensions):
            vol = raw_loader(volume)
        else:
            raise ValueError(
                f"Unsupported volume file {volume!r}: expected .npz or {raw_extensions}."
            )
    else:
        vol = np.asarray(volume, dtype=np.float32)

    if vol.ndim == 2:
        vol = vol[None]
    if vol.ndim != 3:
        raise ValueError(f"Expected a (depth, H, W) volume, got shape {vol.shape}.")
    return vol, crop_offset


def predict_dbt(volume: Union[str, np.ndarray],
                checkpoint: str = DEFAULT_UNET_CKPT,
                image_size: int = 256,
                threshold: float = 0.5,
                model=None,
                device=None,
                architecture: str = "scratch",
                encoder_name: str = "resnet34"):
    """Localise a lesion in a preprocessed DBT volume with the trained U-Net.

    Parameters
    ----------
    volume : str or np.ndarray
        Path to a preprocessed ``.npz`` (uses its ``volume`` key, and ``crop_offset``
        if present to map the box back to full-frame coordinates), a raw DBT
        ``.dcm``/``.dicom`` file (single multi-frame series, normalised on the fly
        via :func:`load_dbt_dicom`), or a raw ``(depth, H, W)`` z-normalised volume
        array.
    checkpoint, image_size, threshold : see training defaults.
    model, device : optional preloaded ``load_unet(...)`` result, to score many
        volumes without reloading the weights.

    Returns
    -------
    dict
        ``{"lesion_detected": bool, "confidence": float, "best_slice": int,
        "box_xywh": (x, y, w, h) | None, "box_full_frame_xywh": ... | None,
        "n_slices": int}``. ``confidence`` is the max lesion-probability over the
        volume; ``best_slice`` is the slice achieving it (in the given volume's
        indexing). Boxes are on the best slice at the original slice resolution.
    """
    vol, crop_offset = _load_volume_and_offset(volume, load_dbt_dicom, (".dcm", ".dicom"))

    if model is None:
        model, device = load_unet(checkpoint, device=device,
                                  architecture=architecture, encoder_name=encoder_name)
    elif device is None:
        device = next(model.parameters()).device

    return _localize_lesion(vol, model, device, image_size, threshold, crop_offset)


DEFAULT_MRI_UNET_CKPT = os.path.join("results_mri", "unet_best.pt")


def predict_dce_mri(volume: Union[str, np.ndarray],
                    checkpoint: str = DEFAULT_MRI_UNET_CKPT,
                    image_size: int = 256,
                    threshold: float = 0.5,
                    model=None,
                    device=None,
                    architecture: str = "scratch",
                    encoder_name: str = "resnet34"):
    """Localise a lesion in a preprocessed DCE-MRI subtraction volume with the
    trained U-Net (see ``TransformData.preprocess_dce_mri_with_boxes``).

    Parameters
    ----------
    volume : str or np.ndarray
        Path to a preprocessed ``.npz`` (post-minus-pre subtraction volume, as
        written by ``preprocess_dce_mri_with_boxes``) or a raw ``(depth, H, W)``
        z-normalised subtraction array. Unlike DBT, there is no single-file raw
        DICOM path here: DCE-MRI needs two whole series (pre + post-contrast) to
        compute the subtraction, so a web upload must be the already-preprocessed
        ``.npz``.
    checkpoint, image_size, threshold : see training defaults. ``checkpoint``
        defaults to the DCE-MRI checkpoint (``results_mri/unet_best.pt``), distinct
        from the DBT one, since the two are trained on different modalities.
    model, device : optional preloaded ``load_unet(...)`` result, to score many
        volumes without reloading the weights.

    Returns
    -------
    dict
        Same contract as :func:`predict_dbt` (``lesion_detected``, ``confidence``,
        ``best_slice``, ``box_xywh``, ``box_full_frame_xywh``, ``n_slices``).
    """
    vol, crop_offset = _load_volume_and_offset(volume, raw_loader=None, raw_extensions=())

    if model is None:
        model, device = load_unet(checkpoint, device=device,
                                  architecture=architecture, encoder_name=encoder_name)
    elif device is None:
        device = next(model.parameters()).device

    return _localize_lesion(vol, model, device, image_size, threshold, crop_offset)


def render_overlay_png(volume: Union[str, np.ndarray], best_slice: int, box_xywh=None):
    """Render the scored slice with the detected lesion box drawn on top, as PNG bytes.

    Meant to be called right after :func:`predict_dbt`/:func:`predict_dce_mri` with
    their own ``best_slice``/``box_xywh`` (the *local*, cropped-volume-relative box
    -- not ``box_full_frame_xywh``), on the same ``.npz``/array that was scored.

    ``best_slice`` is the *full-frame* index those functions return (offset by the
    stored ``crop_offset``); this re-reads ``crop_offset`` from the ``.npz`` to map
    it back to an index into the (already cropped) ``volume`` array actually stored
    on disk, since that is what a demo upload has -- the original uncropped scan is
    typically not around to re-load.
    """
    from PIL import Image, ImageDraw

    crop_offset = (0, 0, 0)
    if isinstance(volume, str):
        with np.load(volume) as data:
            vol = data["volume"].astype(np.float32)
            if "crop_offset" in data.files:
                crop_offset = tuple(int(v) for v in data["crop_offset"])
    else:
        vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None]

    local_slice = max(0, min(best_slice - crop_offset[0], vol.shape[0] - 1))
    img = vol[local_slice]

    # Percentile stretch to 8-bit grayscale for display (the stored array is
    # z-normalised, i.e. roughly zero-mean float, not in a displayable range).
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    pil_img = Image.fromarray((img * 255).astype(np.uint8), mode="L").convert("RGB")

    if box_xywh is not None:
        x, y, w, h = box_xywh
        # Brand accent (--accent, #FF7A59 -- "rehaussement"/overlay lesion, per
        # plan.md Partie 3) rather than an arbitrary red.
        ImageDraw.Draw(pil_img).rectangle([x, y, x + w, y + h], outline=(255, 122, 89), width=2)

    # Upscale small crops so the box is legible in the UI (nearest-neighbour to
    # keep the pixelation honest rather than implying resolution that isn't there).
    scale = max(1, 320 // max(pil_img.size))
    if scale > 1:
        pil_img = pil_img.resize((pil_img.width * scale, pil_img.height * scale), Image.NEAREST)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    # Tiny smoke path for the imaging side: score the first preprocessed volume.
    from glob import glob

    npzs = sorted(glob(os.path.join("preprocessed_data", "*.npz")))
    if npzs:
        print(f"Scoring {npzs[0]} ...")
        print(predict_dbt(npzs[0]))
    else:
        print("No preprocessed .npz volumes found to demo predict_dbt.")
