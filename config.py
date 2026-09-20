"""Every path the project reads or writes, in one place.

Before this module the same folder name was spelled out in a dozen files -- as an
``argparse`` default here, a function default there, a string literal in the app --
so moving a dataset meant hunting through the tree and hoping nothing was missed.

Two rules hold everything together:

* **Medallion layers, not experiments.** Data flows ``bronze`` (bytes as downloaded,
  never edited) -> ``silver`` (normalised volumes + masks, validated) -> ``gold``
  (slice banks, demo cases, the nnU-Net export: derived, cheap to rebuild). A folder's name says
  which layer it belongs to, not which week it was produced.
* **Bronze is a transit zone, not an archive.** Once a series is in silver in every
  corpus that reads it, its DICOM folder is deleted (``pipelines`` "purge" stage, opt out
  with ``--keep-bronze``): the data is then held once, in silver. Only the small
  annotation tables stay in bronze, since every plan is computed from them.
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
BRONZE_DIR = os.path.join(DATA_DIR, "bronze")
SILVER_DIR = os.path.join(DATA_DIR, "silver")
GOLD_DIR = os.path.join(DATA_DIR, "gold")

MODELS_DIR = os.path.join(ROOT, "models")
PLOTS_DIR = os.path.join(ROOT, "plots")
REPORTS_DIR = os.path.join(ROOT, "reports")

# --- Bronze: exactly what the source published, no preprocessing ------------
TCIA_DIR = os.path.join(BRONZE_DIR, "tcia")

# The annotation table lives beside the series it describes. It is the one part of bronze
# that is never purged: it is small, and the ingestion reads it.
MRI_ANNOTATION_BOXES = os.path.join(TCIA_DIR, "Annotation_Boxes.xlsx")

# --- Silver: z-normalised volumes + masks, one .npz per series --------------
DCE_MRI_SILVER_DIR = os.path.join(SILVER_DIR, "dce_mri_p2")
# The nnU-Net corpus has its own preprocessing (`mri_nnunet/`) and its own folder: the
# corpus above is what the served checkpoint was trained on (and a test pins it bit for
# bit), so nothing here may change it. Native-geometry NIfTI for every DCE phase come
# first, which is what lets bronze be purged without losing the ability to re-tune the
# spacing or the normalisation.
DCE_MRI_NNUNET_SILVER_DIR = os.path.join(SILVER_DIR, "dce_mri_nnunet")

# --- Gold: derived from the layer above, rebuildable ------------------------
DEMO_CASES_DIR = os.path.join(GOLD_DIR, "demo_cases")
SLICE_BANK_DIR = os.path.join(GOLD_DIR, "slice_bank_p2")
# The nnU-Net v2 `nnUNet_raw` layout (imagesTr / labelsTr / dataset.json), exported from the
# silver corpus above and rebuildable from it.
NNUNET_RAW_DIR = os.path.join(GOLD_DIR, "nnunet_raw")

# --- Models: checkpoints and the metrics that justify them ------------------
# The DCE-MRI run and the slice classifier are versioned (see .gitignore): the demo
# has to work from `git clone` + `pip install`, with no dataset download.
DCE_MRI_MODEL_DIR = os.path.join(MODELS_DIR, "dce_mri_p2_negfix")
DCE_MRI_UNET_CKPT = os.path.join(DCE_MRI_MODEL_DIR, "unet_best.pt")

SLICE_CLF_DIR = os.path.join(MODELS_DIR, "sliceclf")
SLICE_CLF_CKPT = os.path.join(SLICE_CLF_DIR, "sliceclf_best.pt")

# Where `imaging.train` writes by default. Deliberately not DCE_MRI_MODEL_DIR: that folder holds
# the versioned checkpoint the demo serves, and a training run (a smoke test once did) must
# not be able to overwrite it by being started with no flags.
DCE_MRI_TRAIN_DIR = os.path.join(MODELS_DIR, "dce_mri")


def ensure_dirs(*dirs):
    """Create `dirs` if needed and return them, so callers can inline the call."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs
