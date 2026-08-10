"""Every path the project reads or writes, in one place.

Before this module the same folder name was spelled out in a dozen files -- as an
``argparse`` default here, a function default there, a string literal in the app --
so moving a dataset meant hunting through the tree and hoping nothing was missed.

Two rules hold everything together:

* **Layers, not experiments.** Data flows ``raw_data`` (bytes as downloaded, never
  written to again) -> ``preprocessed_data`` (normalised volumes + masks) ->
  ``curated_data`` (slice banks, demo cases: derived, cheap to rebuild). A folder's
  name says which layer it belongs to, not which week it was produced.
* **Data and artefacts are separate trees.** ``data/`` is entirely gitignored;
  ``models/`` holds checkpoints and metrics, some of which *are* versioned because
  the demo must run from a fresh clone. Nesting the second inside the first would
  force a cascade of gitignore negations, since git cannot re-include a file whose
  parent directory is excluded.

Paths are absolute, derived from this file, so a script works the same whatever the
current working directory.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Layers -----------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "preprocessed_data")
CURATED_DATA_DIR = os.path.join(DATA_DIR, "curated_data")

MODELS_DIR = os.path.join(ROOT, "models")
PLOTS_DIR = os.path.join(ROOT, "plots")
REPORTS_DIR = os.path.join(ROOT, "reports")

# --- Raw: exactly what the source published, no preprocessing ---------------
TCIA_DIR = os.path.join(RAW_DATA_DIR, "tcia")
WISCONSIN_DIR = os.path.join(RAW_DATA_DIR, "wisconsin")
BREAKHIS_DIR = os.path.join(RAW_DATA_DIR, "breakhis")

# Annotation tables live beside the series they describe.
DBT_BOXES_TRAIN = os.path.join(TCIA_DIR, "BCS-DBT-boxes-train.csv")
DBT_BOXES_VALIDATION = os.path.join(TCIA_DIR, "BCS-DBT-boxes-validation.csv")
MRI_ANNOTATION_BOXES = os.path.join(TCIA_DIR, "Annotation_Boxes.xlsx")

# --- Preprocessed: z-normalised volumes + masks, one .npz per series --------
DBT_PREPROCESSED_DIR = os.path.join(PREPROCESSED_DATA_DIR, "dbt")
DCE_MRI_PREPROCESSED_DIR = os.path.join(PREPROCESSED_DATA_DIR, "dce_mri_p2")
WISCONSIN_PREPROCESSED_DIR = os.path.join(PREPROCESSED_DATA_DIR, "wisconsin")

# --- Curated: derived from the layer above, rebuildable ---------------------
DEMO_CASES_DIR = os.path.join(CURATED_DATA_DIR, "demo_cases")
SLICE_BANK_DIR = os.path.join(CURATED_DATA_DIR, "slice_bank_p2")

# --- Models: checkpoints and the metrics that justify them ------------------
# The DCE-MRI run and the slice classifier are versioned (see .gitignore): the demo
# has to work from `git clone` + `pip install`, with no dataset download.
DCE_MRI_MODEL_DIR = os.path.join(MODELS_DIR, "dce_mri_p2_negfix")
DCE_MRI_UNET_CKPT = os.path.join(DCE_MRI_MODEL_DIR, "unet_best.pt")

SLICE_CLF_DIR = os.path.join(MODELS_DIR, "sliceclf")
SLICE_CLF_CKPT = os.path.join(SLICE_CLF_DIR, "sliceclf_best.pt")

DBT_MODEL_DIR = os.path.join(MODELS_DIR, "dbt")
DBT_UNET_CKPT = os.path.join(DBT_MODEL_DIR, "unet_best.pt")

TABULAR_MODEL_DIR = os.path.join(MODELS_DIR, "tabular")

# --- Reports: tables and figures meant to be read, not loaded ---------------
TABULAR_RESULTS_CSV = os.path.join(REPORTS_DIR, "model_results.csv")
PCA_INFO_CSV = os.path.join(REPORTS_DIR, "pca_info.csv")
FEATURE_CONTRIBUTIONS_CSV = os.path.join(REPORTS_DIR, "feature_contributions.csv")
SCREE_PLOT_PNG = os.path.join(PLOTS_DIR, "scree_plot.png")
MODEL_COMPARISON_PNG = os.path.join(PLOTS_DIR, "model_comparison.png")


def ensure_dirs(*dirs):
    """Create `dirs` if needed and return them, so callers can inline the call."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs
