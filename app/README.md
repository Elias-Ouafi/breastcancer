# MRI Cancer-Detection Web App

A small Flask app: upload a breast **MRI/DBT** study, get a cancer-detection verdict
(present/absent + confidence). Ships with a **mock** backend so it runs *before* any
model is connected.

## Run

For the demo, use the launcher at the project root -- it preflights the checkpoint
and the demo cases, then starts this app with the real DCE-MRI backend:

```bash
python run_demo.py
```

To run the app itself (mock backend, no model needed):

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
| `dce_mri` | `MRI_APP_BACKEND=dce_mri` | DCE-MRI lesion localisation via `inference.predict_dce_mri` (checkpoint `results_mri_p2_negfix/unet_best.pt` -- 2nd post-contrast pass, scratch GroupNorm U-Net, 186-patient sample; see `plan.md` §4.1), scored on the post-minus-pre subtraction volume. |

To go live, set one env var (the matching checkpoint must exist and the upload must
be a preprocessed `.npz` volume -- for `dce_mri`, produced by
`TransformData.preprocess_dce_mri_with_boxes`):

```bash
# Windows PowerShell
$env:MRI_APP_BACKEND = "unet"; python -m app.server      # DBT
$env:MRI_APP_BACKEND = "dce_mri"; python -m app.server   # DCE-MRI
```

**Known limitation (`dce_mri` backend, see `plan.md` §4.2):** automatic slice
selection on a raw full-volume upload does not reliably find the lesion yet
(verified 0/186 on held-out patients — the model segments well once shown the right
slice, it just can't find that slice unassisted). For a demo that works every time,
upload one of the curated cases in `demo_cases/` (versioned in the repo; rebuild with
`python scripts/make_demo_cases.py`) rather than an arbitrary patient volume. Those
cases pin a verified-good slice via a `forced_slice` key read automatically by
`inference.predict_dce_mri`, which reports `slice_preselected: true` so the UI can
say the slice was *imposed* rather than found. Both screens carry a "Limites
connues" panel stating this, the bounding-box training masks and the absence of
clinical validation.

Nothing else in the app changes — the result contract
(`lesion_detected`, `slice_preselected`, `confidence`, `best_slice`, `box_xywh`,
`n_slices`) is unchanged apart from that added flag.
If a future model needs different preprocessing or a different signature, wrap it in a
new `Predictor` subclass and add one branch in `get_predictor()`.

## Endpoints

- `GET /` — upload form, plus one-click buttons for the bundled demo cases
- `GET /comment-ca-marche` — the four pipeline steps, and what the model does not do
- `POST /predict` — HTML result page (field name `mri`)
- `POST /demo/<n>` — same page for bundled case `n` (1-based), no upload needed
- `POST /api/predict` — JSON result (field name `mri`)

The HTML result page embeds every slice of the uploaded slab as a data URI so the
slice slider and the MIP toggle work with no round-trip — which matters because the
upload is deleted right after scoring and cannot be re-read.

> Research/demo tool only — not a medical device, not for clinical use.
