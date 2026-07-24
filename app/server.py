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


@app.route("/", methods=["GET"])
def index():
    predictor = get_predictor()
    return render_template("index.html", backend=predictor.name)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("mri")
    if file is None or file.filename == "":
        return render_template("index.html", backend=get_predictor().name,
                               error="Veuillez sélectionner un fichier IRM à envoyer."), 400
    if not _allowed(file.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return render_template("index.html", backend=get_predictor().name,
                               error=f"Type de fichier non pris en charge. Formats acceptés : {allowed}"), 400

    # Persist to a unique temp path (avoids collisions and path-traversal via filename).
    _, ext = os.path.splitext(file.filename)
    safe_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext.lower()}")
    file.save(safe_path)

    predictor = get_predictor()
    overlay_data_uri = None
    try:
        result = predictor.predict(safe_path)
        overlay_data_uri = _overlay_data_uri(safe_path, result)
    except Exception as exc:  # surface backend errors in the UI instead of a 500 page
        return render_template("index.html", backend=predictor.name,
                               error=f"Échec de la prédiction : {exc}"), 500
    finally:
        try:
            os.remove(safe_path)
        except OSError:
            pass

    return render_template("result.html", result=result, overlay_data_uri=overlay_data_uri,
                           filename=file.filename, backend=predictor.name)


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
