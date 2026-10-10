from __future__ import annotations
import io
import logging
import os
from typing import Union
import numpy as np
import config
from logging_setup import setup_logging

log = logging.getLogger(__name__)


def load_unet(checkpoint: str, base: int = 32, device=None):
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
            "python -m imaging.train --data-dir data/silver/dce_mri_p2"
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(base_channels=base)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model, device


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
        "slice_preselected": forced_slice is not None,
        "slice_selector": selector,
        "confidence": best_conf,
        "best_slice": best_slice + crop_offset[0],
        "box_xywh": box,
        "box_full_frame_xywh": box_full,
        "n_slices": int(source_n_slices) if source_n_slices is not None else depth,
    }


def _load_volume_and_offset(volume):
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
        else:
            raise ValueError(f"Unsupported volume file {volume!r}: expected .npz.")
    else:
        vol = np.asarray(volume, dtype=np.float32)

    if vol.ndim == 2:
        vol = vol[None]
    if vol.ndim != 3:
        raise ValueError(f"Expected a (depth, H, W) volume, got shape {vol.shape}.")
    return vol, crop_offset, forced_slice, source_n_slices


DEFAULT_MRI_UNET_CKPT = config.DCE_MRI_UNET_CKPT


def predict_dce_mri(volume: Union[str, np.ndarray],
                    checkpoint: str = DEFAULT_MRI_UNET_CKPT,
                    image_size: int = 256,
                    threshold: float = 0.5,
                    model=None,
                    device=None,
                    classifier=None,
                    use_classifier: bool = True):
    vol, crop_offset, forced_slice, source_n_slices = _load_volume_and_offset(volume)

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
    from PIL import Image

    lo, hi = window
    img = np.clip((slice_array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    return out.convert("RGB") if rgb else out


def _draw_box(pil_img, box_xywh):
    """Draw the lesion box in the brand accent (DOCUMENTATION.md Partie 3), in place."""
    from PIL import ImageDraw

    x, y, w, h = box_xywh
    ImageDraw.Draw(pil_img).rectangle([x, y, x + w, y + h], outline=(255, 122, 89), width=2)


def _upscale(pil_img, target=320):
    from PIL import Image

    scale = max(1, target // max(pil_img.size))
    if scale > 1:
        pil_img = pil_img.resize((pil_img.width * scale, pil_img.height * scale), Image.NEAREST)
    return pil_img


def _fit_for_display(pil_img, max_side=384, target=320):
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
        # DOCUMENTATION.md Partie 3) rather than an arbitrary red.
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

    npzs = sorted(glob(os.path.join(config.DCE_MRI_SILVER_DIR, "*.npz")))
    if npzs:
        log.info(f"Scoring {npzs[0]} ...")
        log.info(predict_dce_mri(npzs[0]))
    else:
        log.info("No preprocessed .npz volumes found to demo predict_dce_mri.")
