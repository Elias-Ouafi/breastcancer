"""Flask web app: upload an MRI/DBT study, get a cancer-detection verdict.

Run locally:

    python -m app.server           # then open http://127.0.0.1:5000

The prediction is produced by whatever :func:`app.predictor.get_predictor` returns.
By default that is a mock (no trained model needed), so this app is fully usable
before the real AI is connected. See ``app/predictor.py``.
"""
from __future__ import annotations

import base64
import os
import tempfile
import uuid

from flask import Flask, redirect, render_template, request, url_for

import config
from app.predictor import get_predictor

# Accepted upload extensions. .npz is what the U-Net backend expects; the others are
# allowed so the UI is usable with raw studies / preview images while mocking.
ALLOWED_EXTENSIONS = {".npz", ".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".nii", ".gz"}
MAX_CONTENT_LENGTH = 512 * 1024 * 1024  # 512 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "mri_app_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _allowed(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def _overlay_data_uri(file_path: str, result: dict) -> str | None:
    """Render the annotated slice for `result` as a data: URI, or None if it can't be.

    Best-effort: the mock backend's box/slice aren't grounded in real pixels, and any
    real upload that isn't a scoreable `.npz` (e.g. a raw image) has nothing to draw
    on, so failures here are swallowed -- the text result still renders without it.
    """
    if result.get("best_slice") is None or not file_path.lower().endswith(".npz"):
        return None
    try:
        from inference import render_overlay_png

        png_bytes = render_overlay_png(file_path, result["best_slice"], result.get("box_xywh"))
        return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    except Exception:
        return None


def _slice_strip(file_path: str, result: dict) -> dict | None:
    """Data URIs for the slice navigator, or None when there is nothing to navigate.

    Best-effort like `_overlay_data_uri`: the page falls back to the single annotated
    slice whenever this returns None (mock backend, non-`.npz` upload, one-slice file,
    or a volume too large to embed).
    """
    if result.get("best_slice") is None or not file_path.lower().endswith(".npz"):
        return None
    try:
        from inference import render_slice_strip

        strip = render_slice_strip(file_path, result["best_slice"], result.get("box_xywh"))
        if strip is None:
            return None
        def encode(png):
            return "data:image/png;base64," + base64.b64encode(png).decode("ascii")

        return {
            "slices": [{"index": s["index"], "uri": encode(s["png"])} for s in strip["slices"]],
            "best_position": strip["best_position"],
            "mip": encode(strip["mip"]),
        }
    except Exception:
        return None


def _overlay_box_pct(file_path: str, result: dict) -> dict | None:
    """Express `box_xywh` as percentages of the rendered slice, for the CSS marker.

    The overlay PNG already carries the drawn box; this only lets the results page
    position its pulsing marker on the *real* detection instead of a decorative
    spot. Percentages survive the uniform upscale `render_overlay_png` applies.
    Returns None whenever the box can't be located -- the page renders fine without.
    """
    box = result.get("box_xywh")
    if not box or not file_path.lower().endswith(".npz"):
        return None
    try:
        import numpy as np

        with np.load(file_path) as data:
            shape = data["volume"].shape
        height, width = shape[-2], shape[-1]
        x, y, w, h = box
        return {
            "left": round(100 * x / width, 2),
            "top": round(100 * y / height, 2),
            "width": round(100 * w / width, 2),
            "height": round(100 * h / height, 2),
        }
    except Exception:
        return None


@app.route("/", methods=["GET"])
def index():
    predictor = get_predictor()
    return render_template("index.html", backend=predictor.name, demo_cases=_demo_cases())


DEMO_DIR = config.DEMO_CASES_DIR


def _demo_cases():
    """The bundled demo cases, sorted. Empty list when the folder is missing."""
    if not os.path.isdir(DEMO_DIR):
        return []
    return sorted(f for f in os.listdir(DEMO_DIR) if f.endswith(".npz"))


@app.route("/demo/<int:case_number>", methods=["POST"])
def predict_demo(case_number: int):
    """Score a bundled demo case, so a pitch does not go through a file picker.

    The case is chosen by position in `_demo_cases()`, never by a client-supplied
    path, so there is nothing to traverse. The file is read in place and not deleted:
    it ships with the repo and contains no patient-identifying data (a slab of an
    already de-identified public TCIA study), unlike an upload.
    """
    cases = _demo_cases()
    if not 1 <= case_number <= len(cases):
        return redirect(url_for("index"))
    return _render_prediction(os.path.join(DEMO_DIR, cases[case_number - 1]),
                              cases[case_number - 1], cleanup=False)


@app.route("/comment-ca-marche", methods=["GET"])
def how_it_works():
    """Static explainer: the four pipeline steps, and what the model does not do."""
    return render_template("how.html", backend=get_predictor().name)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("mri")
    if file is None or file.filename == "":
        return render_template("index.html", backend=get_predictor().name,
                               demo_cases=_demo_cases(),
                               error="Veuillez sélectionner un fichier IRM à envoyer."), 400
    if not _allowed(file.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return render_template("index.html", backend=get_predictor().name,
                               demo_cases=_demo_cases(),
                               error=f"Type de fichier non pris en charge. Formats acceptés : {allowed}"), 400

    # Persist to a unique temp path (avoids collisions and path-traversal via filename).
    _, ext = os.path.splitext(file.filename)
    safe_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext.lower()}")
    file.save(safe_path)

    return _render_prediction(safe_path, file.filename)


def _render_prediction(path: str, display_name: str, cleanup: bool = True):
    """Score ``path`` and render the result page. Shared by /predict and /demo/<n>.

    ``cleanup`` deletes the file afterwards -- required for uploads (the privacy
    promise the UI makes), wrong for the bundled demo cases.
    """
    predictor = get_predictor()
    overlay_data_uri = None
    overlay_box_pct = None
    strip = None
    try:
        result = predictor.predict(path)
        overlay_data_uri = _overlay_data_uri(path, result)
        overlay_box_pct = _overlay_box_pct(path, result)
        strip = _slice_strip(path, result)
    except Exception as exc:  # surface backend errors in the UI instead of a 500 page
        return render_template("index.html", backend=predictor.name,
                               demo_cases=_demo_cases(),
                               error=f"Échec de la prédiction : {exc}"), 500
    finally:
        if cleanup:
            try:
                os.remove(path)
            except OSError:
                pass

    return render_template("result.html", result=result, overlay_data_uri=overlay_data_uri,
                           overlay_box_pct=overlay_box_pct, strip=strip,
                           filename=display_name, backend=predictor.name)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint mirroring /predict, for programmatic / future integrations."""
    file = request.files.get("mri")
    if file is None or file.filename == "" or not _allowed(file.filename):
        return {"error": "Missing or unsupported MRI file."}, 400
    _, ext = os.path.splitext(file.filename)
    safe_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext.lower()}")
    file.save(safe_path)
    predictor = get_predictor()
    try:
        result = predictor.predict(safe_path)
        result["overlay_data_uri"] = _overlay_data_uri(safe_path, result)
    except Exception as exc:
        return {"error": str(exc)}, 500
    finally:
        try:
            os.remove(safe_path)
        except OSError:
            pass
    return result


def main():
    # Loopback only, by design: the server binds to 127.0.0.1 and is therefore
    # unreachable from any other machine on the network. The host is intentionally
    # NOT configurable (no 0.0.0.0 / public binding), so the app can only ever run
    # locally — nothing is deployed to a remote/named server.
    host = "127.0.0.1"
    port = int(os.environ.get("MRI_APP_PORT", "5000"))
    # use_reloader=False keeps a single local process; the interactive debugger
    # (remote code execution surface) stays off.
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
