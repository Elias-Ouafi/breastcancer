# Breast Cancer Detection Project

This project develops an app for breast cancer detection.
The user provide their MRI data and the app returns the probability of cancer and if further medical analysis is required.
WARNING : IT IS NOT CURRENTLY A VALID DIAGNOSTIC USABLE BY END USERS.

My personal goal is to try to help the medical sector my own way and improve my data engineering skills.
The technical results are summarised in Final_Report.md

It uses two complementary angles:

1. **Quantitative / tabular** — the **Breast Cancer Wisconsin (Diagnostic)** dataset
   (UCI ML Repository), used to train and compare classical classifiers. This
   pipeline runs on **PySpark / Spark MLlib**.
2. **Medical imaging** — breast **MRI/DBT** DICOM series from **TCIA**
   (`Breast-Cancer-Screening-DBT` collection) with their segmentations, and the
   **BreakHis** histopathology images from Kaggle.

The long-term goal (see `Main.py`) is to combine the strongest features from both
the quantitative and imaging pipelines into a single model.

## Quick start — the demo

The DCE-MRI lesion-localisation demo runs from a fresh clone, with no dataset
download: the checkpoint and three curated cases are versioned.

```bash
pip install -r requirements.txt
python run_demo.py
```

Everything below the next two sections is the *training* side, which does need the
datasets. A French version of this walkthrough is in [DEMO.md](DEMO.md).

### Running it

**The day before**, confirm the machine is ready without occupying a port:

```bash
python run_demo.py --check
```

**On the day:**

1. `python run_demo.py`
2. Open <http://127.0.0.1:5000>
3. Click **Cas 1** — no file picker to fumble with. (Drag-and-drop still works for
   your own exam.)
4. Walk through the result, in this order:
   - the verdict and confidence, with the compute time (~110 ms)
   - the annotated slice, accent box on the enhancing region
   - **drag the slider** — the lesion appears, peaks and fades. This is the moment
     that convinces.
   - **Vue MIP** — the maximum-intensity projection, how a radiologist reads it
   - unfold *Détail technique* if you get questions
5. Follow with **Comment ça marche** (link at the bottom) if they want the pipeline.

### What to say yourself

Three sentences, before anyone has to ask:

> The slice was picked in advance by a human. The model segments a lesion well
> **once shown the right slice** — 88 % sensitivity. It cannot find that slice on its
> own yet: a dedicated classifier took that from 0 % to 43 %, and that work is
> ongoing.

> The 0.53 Dice is not comparable to the ~0.80 in the literature: our training masks
> are bounding boxes, not expert contours.

> Nothing is clinically validated. This is research on a 186-patient sample.

The app already states all of this in its "Limites connues" panel. Saying it before
they read it buys credibility rather than costing it.

### If it goes wrong

| Symptom | Fix |
|---------|-----|
| `run_demo.py` refuses to start | It names the missing prerequisite; follow the `->` line it prints |
| Port already in use | `python run_demo.py --port 5001` |
| Engine badge reads `mock` | The launcher was bypassed — start again with `run_demo.py` |
| No annotated slice shown | The upload is not a preprocessed `.npz`; use a `data/curated_data/demo_cases/` file |

**Safety net:** screenshot a successful result the day before. If the live run fails,
you carry on without a gap.

## Where things live

Data flows through three layers, and trained artefacts are kept apart from it:

```
data/
├── raw_data/          exactly what the source published, never written to again
│   ├── tcia/          DICOM series + annotation tables (~80 GB)
│   ├── wisconsin/     the UCI CSV
│   └── breakhis/      the Kaggle archive
├── preprocessed_data/ z-normalised volumes + masks, one .npz per series
│   ├── dbt/           from preprocess_dbt_with_boxes
│   └── dce_mri_p2/    from preprocess_dce_mri_with_boxes (the model in the demo)
└── curated_data/      derived and rebuildable: slice banks, the three demo cases
models/                checkpoints + the metrics that justify them
reports/               metric tables meant to be read
plots/                 figures
```

Every one of those paths is defined once, in [config.py](config.py) — no folder name
is spelled out in a pipeline script, so moving a dataset is a one-line change. `data/`
and `models/` are gitignored apart from the demo cases and the two checkpoints the
demo needs, which is what lets `git clone` + `pip install` replay it.

## Running the imaging pipeline as one flow

The four DCE-MRI stages — download, preprocess, train, evaluate — are wired together
as a [Prefect](https://www.prefect.io) flow in [pipelines/dce_mri.py](pipelines/dce_mri.py):

```bash
pip install -e ".[orchestration]"
python -m pipelines.dce_mri --dry-run          # print the plan, run nothing
python -m pipelines.dce_mri                    # run it
python -m pipelines.dce_mri --from preprocess  # resume, skipping the 60 GB download
```

Each stage checks its own output and skips when it is already there, so a run that
died in training resumes instead of redoing the two days before it; `--force` re-runs
anyway. Only the download retries — a TCIA fetch fails on timeouts, whereas re-running
a crashed training run would just burn another hour reaching the same exception.

The stages call the same functions as the manual commands documented below, so the two
cannot drift apart.

## Development

```bash
pip install -e ".[dev]"
ruff check .        # lint
pytest              # 198 tests, ~25 s, no GPU or dataset needed
```

[CI](.github/workflows/ci.yml) runs both on every push and pull request. The suite
covers the metric definitions, the storage-layout contract, the orchestration logic,
and the assets the demo needs — including whether git actually *tracks* the checkpoint
and the demo cases, which is the failure that would otherwise surface five minutes
before a pitch. On a machine with a JVM, one more test compares the exported tabular
model against Spark and adds ~80 s on its own; CI installs neither Spark nor Java, so
it skips there.

Pipelines log through `logging` (see [logging_setup.py](logging_setup.py)), with a
timestamp per line and a copy under `logs/`. Turn the volume up with
`BREASTCANCER_LOG_LEVEL=DEBUG`.

## Prerequisites

- Python 3.12 or higher
- **Java 17 or newer (a JVM)** on the `PATH` — required by **PySpark 4** for the
  quantitative pipeline (e.g. Temurin/OpenJDK 17; set `JAVA_HOME`).
- **TCIA** access via `tcia_utils` / `nbia` (for the MRI/DBT dataset)
- A **Kaggle** account and API token (for the BreakHis dataset)

## Setup

### Kaggle (for BreakHis)
1. Create a Kaggle account and generate an API token (`kaggle.json`).
2. Place it at `.kaggle/kaggle.json` in the project directory.

### TCIA (for MRI/DBT)
The MRI download uses `tcia_utils.nbia`; public collections such as
`Breast-Cancer-Screening-DBT` require no API key. Series land in `data/raw_data/tcia/`;
change `TCIA_DIR` in `config.py` to put them elsewhere.

## Usage

### Quantitative pipeline (Wisconsin)
Run the full extract → transform (PCA) → analyze pipeline:
```bash
python Main.py
```
This downloads the Wisconsin dataset, lifts it into a **Spark DataFrame** (via a
JVM-native `spark.read` of the fetched table), then uses **Spark MLlib** to
impute/standardize it, apply PCA (smallest number of components retaining 95% of the
variance), and train several classifiers (Logistic Regression, Random Forest, Linear
SVM, Gradient-Boosted Trees, and a Multilayer Perceptron). It writes metric tables to
`reports/` and figures to `plots/`. Requires a JVM (see Prerequisites). See
`Final_Report.md` for a summary of the outcomes.

> **Migration note:** the quantitative pipeline was moved from scikit-learn/XGBoost
> to PySpark / Spark MLlib. `KNeighborsClassifier` has no MLlib equivalent and was
> replaced by Gradient-Boosted Trees (which also subsumes the old XGBoost model), and
> `SVC` maps to MLlib's linear `LinearSVC`. Spark PCA takes a fixed component count,
> so the 95%-variance target is met by fitting once at full rank and selecting `k`.
> The imaging pipeline is intentionally **not** on Spark — DICOM I/O, image
> resampling (SimpleITK/ITK) and U-Net training (PyTorch) have no MLlib equivalent.

### Imaging pipeline

The imaging side trains a **2D U-Net** to *localise* lesions in breast scans. To
learn, it needs, for each image, a **mask** marking where the lesion is. That mask
can come from two sources depending on the dataset:

- **DBT (mammography / tomosynthesis)** — lesions are given as **bounding boxes** in
  a separate annotation CSV. This is the working path below.
- **MRI** — lesions may come as DICOM **SEG/RTSTRUCT** segmentations, handled by
  `preprocess_mri_data` (resamples to 1 mm, builds the mask from the SEG files).

#### DBT workflow (bounding-box annotations)

DBT (Digital Breast Tomosynthesis) is a 3D mammogram: a stack of X-ray "slices" of
the breast. In the `Breast-Cancer-Screening-DBT` collection, **most scans are
normal** — only a subset of patients have a biopsied lesion, listed with a box
(patient, view, slice, x/y/width/height) in the annotation CSV. So the pipeline is
**annotation-driven**: fetch the tables first, download the patients they name, then
turn each box into a mask. Everything downloaded that way carries a lesion, which is
why step 1b exists: a specificity cannot be measured on a corpus without negatives, and
"absent from the boxes CSV" is not the same statement as "normal exam".

```python
import config
from ExtractData import (
    download_annotated_dbt_series,
    download_dbt_tables,
    download_normal_dbt_series,
)
from TransformData import preprocess_dbt_with_boxes

# 0. Get the BCS-DBT tables once into data/raw_data/tcia/ (boxes, per-view labels, and
#    the file-paths inventory). Existing files are kept; pass overwrite=True to refresh.
download_dbt_tables()

# The boxes of the training set (101 patients) pool with the validation set (40
# disjoint patients) — same schema. Both functions take a path or a list of paths.
BOXES = [config.DBT_BOXES_TRAIN, config.DBT_BOXES_VALIDATION]

# 1. Download the DBT series of the annotated patients (cap the volume with max_gb).
#    max_patients=None fetches every annotated patient in the pooled CSVs.
download_annotated_dbt_series(
    BOXES, max_patients=None,
    download_dir="data/raw_data/tcia", max_gb=25,
)

# 1b. Exams *without* a cancer, which the boxes CSVs cannot give you: the per-view
#     labels table is the only place the collection says an exam is normal (4 581 of
#     its 5 060 patients). A patient is taken only if every one of its views is normal.
#     ~200 MB per patient; the cap is on what this call adds, not on the folder size.
download_normal_dbt_series(max_patients=150, max_gb_added=50)

# 2. Build a box mask per series and save compressed .npz (skips views with no box).
preprocess_dbt_with_boxes(
    root_dir="data/raw_data/tcia",
    boxes_csv=BOXES,
    file_paths_csv=config.DBT_FILE_PATHS,   # the inventory; this is the match
    output_dir="data/preprocessed_data/dbt",
)
```
```bash
# 3. Train the lesion-localisation U-Net on the preprocessed .npz volumes.
python -m imaging.train --data-dir data/preprocessed_data/dbt --epochs 25
```

Step 2 matches each downloaded series to its boxes with a **join**, not a guess:
`BCS-DBT-file-paths-*.csv` lists `(PatientID, StudyUID, View)` for every series folder,
and a box belongs to the series carrying its three values. A folder the inventory does
not list is skipped and counted, since nothing says which box is its own. The pixels
keep one job — the one the dataset's own reader gives them: a study stored rotated
relative to the frame its boxes live in is flipped (laterality read from whichever edge
carries signal), and the manifest records which cases were.

Two earlier versions inferred the view instead, and the measurements are why they are
gone: the DICOM laterality tag reads `L` on all 262 downloaded series (147 of 253 series
matched, 23 masks on background), and deriving laterality from the pixels was right 237
times out of 262, found 253 of the 260 annotated series, and took the box of the *other*
acquisition of a repeated view (`lmlo` vs `lmlo1`) 4 times — a case no pixel can decide
(plan.md §4.4). Preprocessing then z-normalises the image, paints the box(es) into a
binary mask with `create_mask`, crops to the lesion region of interest, and stores the
real `PatientID` inside the `.npz` (as `case_id`).

It also reads the boxes CSV's **`Class`** column and stores it: each `.npz` carries
`lesion_class` (`benign` or `cancer`) and a 0/1 `label`. The mask cannot carry that —
a benign lesion paints the same pixels as a cancer — so a file holding only a mask
cannot answer the exam-level question, which is *cancer or not*, not *lesion or not*.
Pass `mask_classes=("cancer",)` to paint only the cancer boxes; the default paints
both, since a benign lesion is still something to localise and benign patients are
two thirds of the annotated set. `Class` is required: a CSV without it is rejected
rather than treated as one undistinguished class. A completed run writes
`manifest.json` beside the volumes (`lineage.py`) with the per-class counts, so the
class balance of a corpus can be read without opening a single volume.

The `imaging/` package then trains the U-Net: it reads the `.npz` volumes, splits
them **by patient** (`case_id`) so no patient straddles train/val/test, serves axial
slices, and optimises a combined BCE + soft-Dice loss. Metrics (**Dice**, **IoU**)
go to `models/dbt/segmentation_metrics.csv` and the best checkpoint to
`models/dbt/unet_best.pt`. Because the masks are boxes rather than fine contours, this
targets lesion *localisation*, and the achievable Dice is inherently limited.
Requires `torch` (install the wheel matching your platform/CUDA). Validate the whole
loop without any data via `python -m imaging.train --smoke-test`.

#### Evaluating a trained model

`imaging.train` reports one number: mean Dice over lesion-bearing slices, with no
uncertainty and no view of what happens on a healthy slice. `imaging.evaluate` adds
what that leaves out — bootstrap confidence intervals (resampling *patients*, since
slices within a patient are correlated), lesion-level sensitivity, false positives
per whole volume, and measured inference time:

```bash
python -m imaging.evaluate --data-dir data/preprocessed_data/dce_mri_p2 \
    --checkpoint models/dce_mri_p2_negfix/unet_best.pt
```

It writes `eval_report.json` (summary) and `eval_per_patient.csv` (one row per
patient, so any figure can be traced back). Current results are in `plan.md` §4.3.

#### Slice classifier

The segmentation U-Net cannot pick a lesion's slice out of a full volume (see
`plan.md` §4.2/§4.3). `imaging.sliceclf` trains a separate model for that ranking
task alone, on *every* slice rather than a sampled subset of negatives:

```bash
python -m imaging.sliceclf --slice-bank data/curated_data/slice_bank_p2 --epochs 25
```

It is selected on top-1 accuracy — "is the volume's highest-scoring slice really
lesion-bearing?" — the metric the segmentation confidence scored 0 on.

#### Lesion classifier (benign vs cancer, step 2 from the image)

`imaging.lesionclf` answers step 2's question — is this lesion benign or malignant? —
from the DBT crops instead of from a cytology form, using the `label` that
`preprocess_dbt_with_boxes` stores:

```bash
python -m imaging.lesionclf --data-dir data/preprocessed_data/dbt --folds 5
```

It is **not** a detection model: the volume is cropped around the annotated box, so
the lesion's location is given. Nothing it reports says anything about finding a
cancer in a screening exam — that is the separate exam-level head, which needs full
frames and normal exams.

It cross-validates instead of holding out one test split, because 130 patients split
15 % leaves about 8 cancers in test and an interval that would cover nearly
everything; the folds are stratified by class over patients, and every patient is
scored once by a model that never saw it. No checkpoint or epoch is chosen on the
held-out patients. Writes `models/lesionclf/cv_report.json` (patient ROC-AUC with a
patient-bootstrap CI, plus sensitivity/specificity/PPV **and the prevalence they were
measured at**) and `cv_predictions.csv` (one row per patient). Validate the loop with
no dataset via `python -m imaging.lesionclf --smoke-test`.

### Histopathology (BreakHis)
```bash
python ExtractBreakHis.py
```

## Acknowledgments

- [TCIA](https://www.cancerimagingarchive.net/) — MRI/DBT imaging data
- [UCI ML Repository](https://archive.ics.uci.edu/) — Breast Cancer Wisconsin dataset
- [Kaggle](https://www.kaggle.com/) — BreakHis histopathology dataset
