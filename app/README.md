# MRI Cancer-Detection Web App

A small Flask app: upload a breast **MRI/DBT** study, get a cancer-detection verdict
(present/absent + confidence). Ships with a **mock** backend so it runs *before* any
model is connected.

## Run

```bash
pip install -r ../requirements.txt   # or: pip install "Flask>=3.0"
python -m app.server                 # from the project root
# open http://127.0.0.1:5000
```

**Local-only by design.** The server binds to `127.0.0.1` (loopback) — it is not
reachable from any other machine, is not deployed to any remote/named server, and
cannot be exposed to the network (no `0.0.0.0` binding is configurable). Uploaded files
are written to a local temp folder and deleted right after scoring. Only the port is
configurable, via `MRI_APP_PORT`.

## Connecting the real AI (later)

The web layer only talks to a `Predictor` (see [`predictor.py`](predictor.py)).
Three backends exist:

| Backend | Selected by | What it does |
|---------|-------------|--------------|
| `mock` (default) | — | Fabricates a plausible result; ignores pixels. |
| `unet` | `MRI_APP_BACKEND=unet` | DBT lesion localisation via `inference.predict_dbt` (checkpoint `results/unet_best.pt`). |
| `dce_mri` | `MRI_APP_BACKEND=dce_mri` | DCE-MRI lesion localisation via `inference.predict_dce_mri` (checkpoint `results_mri/unet_best.pt`), scored on the post-minus-pre subtraction volume. |

To go live, set one env var (the matching checkpoint must exist and the upload must
be a preprocessed `.npz` volume -- for `dce_mri`, produced by
`TransformData.preprocess_dce_mri_with_boxes`):

```bash
# Windows PowerShell
$env:MRI_APP_BACKEND = "unet"; python -m app.server      # DBT
$env:MRI_APP_BACKEND = "dce_mri"; python -m app.server   # DCE-MRI
```

Nothing else in the app changes — the result contract
(`lesion_detected`, `confidence`, `best_slice`, `box_xywh`, `n_slices`) is identical.
If a future model needs different preprocessing or a different signature, wrap it in a
new `Predictor` subclass and add one branch in `get_predictor()`.

## Endpoints

- `GET /` — upload form
- `POST /predict` — HTML result page
- `POST /api/predict` — JSON result (field name `mri`)

> Research/demo tool only — not a medical device, not for clinical use.
