"""Reusable inference layer for the breast-cancer project.

Two entry points, one per pipeline, so a demo/UI can get a prediction in one call
without re-running the batch training scripts:

* :func:`predict_tabular` — scores a single Wisconsin 30-feature record with the
  persisted Spark ``PipelineModel`` (see ``train_tabular_model.py``). Returns the
  predicted diagnosis and, for the logistic model, a malignancy probability.
* :func:`predict_dbt` — runs the trained 2D U-Net (``models/dbt/unet_best.pt``) over
  a preprocessed DBT ``.npz`` volume (or a raw volume array) and returns the localised
  lesion: best slice, bounding box, and a detection confidence.

Neither entry point retrains anything; both load saved artefacts. The tabular path
needs a JVM (PySpark); the imaging path needs only ``torch`` + ``numpy``.
"""
from __future__ import annotations

import io
import json
import logging
import os
from typing import Mapping, Sequence, Union

import numpy as np

import config
from logging_setup import setup_logging

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tabular inference (Wisconsin, Spark MLlib)
# --------------------------------------------------------------------------- #

DEFAULT_TABULAR_DIR = config.TABULAR_MODEL_DIR


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

DEFAULT_UNET_CKPT = config.DBT_UNET_CKPT


def load_unet(checkpoint: str = DEFAULT_UNET_CKPT, base: int = 32, device=None):
    """Load the trained U-Net in eval mode. Returns ``(model, device)``.

    ``base`` must match what the checkpoint was trained with
    (``imaging.train --base-channels``) since it determines the state_dict's layer
    shapes.
    """
    import torch

    from imaging.unet import build_model

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No U-Net checkpoint at {checkpoint!r}. Train it first: "
            "python -m imaging.train --data-dir data/preprocessed_data/dbt"
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(base_channels=base)
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


DEFAULT_SLICE_CLF_CKPT = config.SLICE_CLF_CKPT


def load_slice_classifier(checkpoint: str = DEFAULT_SLICE_CLF_CKPT, base: int = 16, device=None):
    """Load the slice-ranking classifier, or return ``(None, None)`` if absent.

    Missing weights are not an error: the classifier is an optional improvement over
    ranking by segmentation confidence, and every caller falls back to that.
    """
    if not os.path.exists(checkpoint):
        return None, None
    import torch

    from imaging.sliceclf import SliceClassifier

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SliceClassifier(base=base)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model, device


def _rank_slices(vol, classifier, device, image_size=256, batch_size=32):
    """Lesion probability per slice, from the slice classifier."""
    import torch
    import torch.nn.functional as F

    scores = np.zeros(vol.shape[0], dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, vol.shape[0], batch_size):
            chunk = torch.from_numpy(vol[lo:lo + batch_size]).to(device)[:, None]
            chunk = F.interpolate(chunk, size=(image_size, image_size),
                                  mode="bilinear", align_corners=False)
            scores[lo:lo + batch_size] = torch.sigmoid(classifier(chunk))[:, 0].cpu().numpy()
    return scores


def _localize_lesion(vol, model, device, image_size=256, threshold=0.5, crop_offset=(0, 0, 0),
                     forced_slice=None, source_n_slices=None, classifier=None):
    """Shared slice-scan loop behind :func:`predict_dbt` and :func:`predict_dce_mri`.

    Scores every axial slice of ``vol`` (a ``(depth, H, W)`` z-normalised array) with
    ``model``, keeps the slice with the highest lesion probability, and returns its
    bounding box both in the given volume's own indexing and mapped back to the
    original (uncropped) frame via ``crop_offset`` (as stored by
    ``TransformData.save_preprocessed``).

    ``forced_slice``, if given, skips the scan and scores only that one slice. This
    exists for curated demo cases (see ``TransformData.make_demo_case``): on
    full-frame DCE-MRI, per-pixel confidence saturates near 1.0 on effectively every
    slice (verified across 186 held-out patients -- the argmax landed on a
    lesion-bearing slice 0/186 times), so ranking slices by confidence does not
    reliably find the lesion even though the model segments it well *once shown the
    right slice* (~0.58 Dice on ground-truth-positive slices). Root-causing that
    confidence collapse is tracked separately; ``forced_slice`` sidesteps it for
    known-good cases without touching the scan path everyone else still uses.

    ``source_n_slices`` overrides the reported ``n_slices``. It exists for slim demo
    cases (``TransformData.make_demo_case``), which store only the forced slice: the
    array's own depth is then 1, but the study it was taken from had ~176 slices, and
    that is the number the UI should show.

    ``classifier``, when given and no slice is forced, picks the slice to segment
    instead of the segmentation confidence. It is a genuine improvement on a real
    upload -- 42.9 % top-1 [CI95 25.0-60.7] against 0.0 % -- but nowhere near
    reliable, which is why curated demo cases still pin their slice. The chosen
    mechanism is reported back as ``slice_selector`` so the UI never has to guess how
    the slice was obtained.
    """
    import torch
    import torch.nn.functional as F

    depth, orig_h, orig_w = vol.shape
    best_conf, best_slice, best_prob_small = -1.0, (forced_slice or 0), None

    if forced_slice is not None:
        selector, z_range = "pinned", [int(forced_slice)]
    elif classifier is not None:
        selector = "classifier"
        z_range = [int(np.argmax(_rank_slices(vol, classifier, device, image_size)))]
    else:
        selector, z_range = "segmentation_confidence", range(depth)

    with torch.no_grad():
        for z in z_range:
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
        # True when the slice was pinned rather than found by the model. The UI
        # states this outright instead of letting a curated case read as autonomous
        # detection -- the limitation is documented, so it should also be visible.
        "slice_preselected": forced_slice is not None,
        "slice_selector": selector,
        "confidence": best_conf,
        "best_slice": best_slice + crop_offset[0],
        "box_xywh": box,
        "box_full_frame_xywh": box_full,
        "n_slices": int(source_n_slices) if source_n_slices is not None else depth,
    }


def _load_volume_and_offset(volume, raw_loader, raw_extensions):
    """Resolve ``volume`` to a ``(vol, crop_offset, forced_slice, source_n_slices)`` tuple.

    ``.npz`` paths use their ``volume``/``crop_offset`` keys (as written by
    ``TransformData.save_preprocessed``) plus ``forced_slice`` and
    ``source_n_slices`` if present (as written by
    ``TransformData.make_demo_case``); paths ending in ``raw_extensions`` go
    through ``raw_loader``; anything else is treated as an already-loaded array.
    """
    crop_offset = (0, 0, 0)
    forced_slice = None
    source_n_slices = None
    if isinstance(volume, str):
        lower = volume.lower()
        if lower.endswith(".npz"):
            with np.load(volume) as data:
                vol = data["volume"].astype(np.float32)
                if "crop_offset" in data.files:
                    crop_offset = tuple(int(v) for v in data["crop_offset"])
                if "forced_slice" in data.files:
                    forced_slice = int(data["forced_slice"])
                if "source_n_slices" in data.files:
                    source_n_slices = int(data["source_n_slices"])
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
    return vol, crop_offset, forced_slice, source_n_slices


def predict_dbt(volume: Union[str, np.ndarray],
                checkpoint: str = DEFAULT_UNET_CKPT,
                image_size: int = 256,
                threshold: float = 0.5,
                model=None,
                device=None):
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
        ``{"lesion_detected": bool, "slice_preselected": bool, "confidence": float,
        "best_slice": int, "box_xywh": (x, y, w, h) | None,
        "box_full_frame_xywh": ... | None,
        "n_slices": int}``. ``confidence`` is the max lesion-probability over the
        volume; ``best_slice`` is the slice achieving it (in the given volume's
        indexing). Boxes are on the best slice at the original slice resolution.
    """
    vol, crop_offset, forced_slice, source_n_slices = _load_volume_and_offset(
        volume, load_dbt_dicom, (".dcm", ".dicom"))

    if model is None:
        model, device = load_unet(checkpoint, device=device)
    elif device is None:
        device = next(model.parameters()).device

    return _localize_lesion(vol, model, device, image_size, threshold, crop_offset,
                            forced_slice, source_n_slices)


DEFAULT_MRI_UNET_CKPT = config.DCE_MRI_UNET_CKPT


def predict_dce_mri(volume: Union[str, np.ndarray],
                    checkpoint: str = DEFAULT_MRI_UNET_CKPT,
                    image_size: int = 256,
                    threshold: float = 0.5,
                    model=None,
                    device=None,
                    classifier=None,
                    use_classifier: bool = True):
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
        defaults to the DCE-MRI checkpoint (``results_mri_p2/unet_best.pt`` --
        second post-contrast pass, scratch GroupNorm U-Net, 186-patient full-frame
        sample; see plan.md §4.1 for how this configuration was chosen), distinct
        from the DBT one, since the two are trained on different modalities.
    model, device : optional preloaded ``load_unet(...)`` result, to score many
        volumes without reloading the weights.
    classifier, use_classifier : the slice-ranking model (see
        :func:`load_slice_classifier`). When the volume does not pin a slice, it
        chooses which slice to segment -- 42.9 % top-1 against 0.0 % for the
        segmentation confidence it replaces, still far from reliable. Loaded on
        demand if not supplied; set ``use_classifier=False`` to force the old
        confidence scan (useful for reproducing earlier numbers).

    Returns
    -------
    dict
        Same contract as :func:`predict_dbt` (``lesion_detected``, ``confidence``,
        ``best_slice``, ``box_xywh``, ``box_full_frame_xywh``, ``n_slices``), plus
        ``slice_selector`` -- ``"pinned"``, ``"classifier"`` or
        ``"segmentation_confidence"``.
    """
    vol, crop_offset, forced_slice, source_n_slices = _load_volume_and_offset(
        volume, raw_loader=None, raw_extensions=())

    if model is None:
        model, device = load_unet(checkpoint, device=device)
    elif device is None:
        device = next(model.parameters()).device

    # Only worth loading when it will actually be consulted: a pinned slice makes the
    # ranking irrelevant, and that is the path every demo case takes.
    if classifier is None and use_classifier and forced_slice is None:
        classifier, _ = load_slice_classifier(device=device)

    return _localize_lesion(vol, model, device, image_size, threshold, crop_offset,
                            forced_slice, source_n_slices, classifier)


def _to_display_image(slice_array, window, rgb=True):
    """Percentile-stretch a z-normalised slice to an 8-bit PIL image.

    ``window`` is the ``(lo, hi)`` intensity pair to map onto 0..255. Passing the
    same window for every slice of a stack is what keeps brightness stable while
    scrolling; a per-slice window makes the whole image pulse and would let a lesion
    look like it is appearing when only the contrast changed.

    ``rgb=False`` keeps the image 8-bit grayscale, which encodes to roughly a third
    of the PNG bytes. Only the slice that carries the coloured lesion box needs the
    extra channels.
    """
    from PIL import Image

    lo, hi = window
    img = np.clip((slice_array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    return out.convert("RGB") if rgb else out


def _draw_box(pil_img, box_xywh):
    """Draw the lesion box in the brand accent (plan.md Partie 3), in place."""
    from PIL import ImageDraw

    x, y, w, h = box_xywh
    ImageDraw.Draw(pil_img).rectangle([x, y, x + w, y + h], outline=(255, 122, 89), width=2)


def _upscale(pil_img, target=320):
    """Enlarge small crops so the box is legible.

    Nearest-neighbour on purpose: it keeps the pixelation visible rather than
    implying a resolution the scan does not have.
    """
    from PIL import Image

    scale = max(1, target // max(pil_img.size))
    if scale > 1:
        pil_img = pil_img.resize((pil_img.width * scale, pil_img.height * scale), Image.NEAREST)
    return pil_img


def _fit_for_display(pil_img, max_side=384, target=320):
    """Scale a slice to a sensible on-screen size, both directions.

    Small crops are enlarged by :func:`_upscale`; native 512x512 slices are reduced.
    The reduction matters because the navigator embeds every slice in the page as a
    data URI: at 512x512 RGB a 25-slice strip is ~6.5 MB of HTML, which is exactly
    the lag the slice navigator is supposed not to have. The app panel is 440 px
    wide, so nothing visible is lost. Downscaling uses LANCZOS (proper resampling);
    the nearest-neighbour rule applies only to *upscaling*, where it is what keeps
    the pixelation honest.
    """
    from PIL import Image

    longest = max(pil_img.size)
    if longest > max_side:
        scale = max_side / longest
        return pil_img.resize((round(pil_img.width * scale), round(pil_img.height * scale)),
                              Image.LANCZOS)
    return _upscale(pil_img, target)


def _png_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def render_slice_strip(volume: Union[str, np.ndarray], best_slice: int, box_xywh=None,
                       max_slices: int = 41):
    """Render each slice of a small volume as a PNG, plus a MIP over the stack.

    Backs the results page's slice navigator. Scrolling through neighbouring slices
    is what makes a lesion read as a real three-dimensional finding -- it grows,
    peaks and fades -- instead of one frame the viewer has to take on trust.

    Returns ``{"slices": [{"index", "png"}...], "best_position": int, "mip": bytes}``,
    where ``index`` is the full-frame slice number (offset by the stored
    ``crop_offset``) and ``best_position`` is the scored slice's position in the list.
    ``None`` is returned when there is nothing to navigate (a single slice, or more
    than ``max_slices``: embedding a whole 176-slice volume as data URIs would add
    tens of megabytes to the page for no benefit).

    The box is drawn **only** on the scored slice, since that is the only slice the
    model actually looked at -- repeating it across the stack would imply a 3D
    detection that was never computed.
    """
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

    depth = vol.shape[0]
    if depth < 2 or depth > max_slices:
        return None

    window = tuple(np.percentile(vol, [1, 99]))
    local_best = max(0, min(best_slice - crop_offset[0], depth - 1))

    slices = []
    for z in range(depth):
        img = _to_display_image(vol[z], window, rgb=(z == local_best and box_xywh is not None))
        if z == local_best and box_xywh is not None:
            _draw_box(img, box_xywh)
        slices.append({"index": z + crop_offset[0], "png": _png_bytes(_fit_for_display(img))})

    # Maximum-intensity projection: the brightest value each pixel reaches anywhere in
    # the slab. Standard DCE-MRI reading practice -- an enhancing lesion survives the
    # projection while slice-level noise does not.
    mip = _to_display_image(vol.max(axis=0), window, rgb=False)
    return {"slices": slices, "best_position": local_best,
            "mip": _png_bytes(_fit_for_display(mip))}


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
    setup_logging()
    # Tiny smoke path for the imaging side: score the first preprocessed volume.
    from glob import glob

    npzs = sorted(glob(os.path.join(config.DBT_PREPROCESSED_DIR, "*.npz")))
    if npzs:
        log.info(f"Scoring {npzs[0]} ...")
        log.info(predict_dbt(npzs[0]))
    else:
        log.info("No preprocessed .npz volumes found to demo predict_dbt.")
