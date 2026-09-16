# Training-side pipeline — reference

The commands and design notes behind each stage of the data pipeline. The overview and
the architecture diagram are in the [README](../README.md); the dated measurement log
(in French) is [plan.md](../plan.md). Unlike the demo, everything here needs the TCIA
datasets on disk.

## Running the DCE-MRI pipeline as one flow

The four DCE-MRI stages — download, preprocess, train, evaluate — are wired together
as a [Prefect](https://www.prefect.io) flow in [pipelines/dce_mri.py](../pipelines/dce_mri.py):

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

## Prerequisites

- Python 3.12 or higher
- **TCIA** access via `tcia_utils` / `nbia` (for the MRI/DBT dataset)

## Setup

### TCIA (for MRI/DBT)
The MRI download uses `tcia_utils.nbia`; public collections such as
`Breast-Cancer-Screening-DBT` require no API key. Series land in `data/raw_data/tcia/`;
change `TCIA_DIR` in `config.py` to put them elsewhere.

## What the models are trained to do

The imaging side trains a **2D U-Net** to *localise* lesions in breast scans. To
learn, it needs, for each image, a **mask** marking where the lesion is. That mask
can come from two sources depending on the dataset:

- **DBT (mammography / tomosynthesis)** — lesions are given as **bounding boxes** in
  a separate annotation CSV. This is the working path below.
- **MRI** — lesions may come as DICOM **SEG/RTSTRUCT** segmentations, handled by
  `preprocess_mri_data` (resamples to 1 mm, builds the mask from the SEG files).

## DBT workflow (bounding-box annotations)

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
#     ~310 MB per patient (46.8 GB measured for 150); the cap is on what this call
#     adds, not on the folder size.
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

## The exam-level corpus (cancer or no cancer)

Everything above is annotation-driven, so every volume it writes holds a lesion:
prevalence 100 %, and a specificity that cannot be measured at all. `preprocess_dbt_exams`
builds the other corpus — the one an exam-level decision is scored on:

```python
from TransformData import preprocess_dbt_exams

preprocess_dbt_exams(                      # writes data/preprocessed_data/dbt_exams/
    boxes_csv=BOXES,                       # optional: paints the masks, never the label
)
```

The label comes from `BCS-DBT-labels-*.csv`, the only table that says an exam is normal,
read at the patient's **worst view** — so a series with no box is a *negative* rather
than a skip, and `label` means one thing: 1 is cancer. `actionable` (recalled, not
biopsied) and `benign` are 0, and the word itself is stored as `exam_status`, so moving
that line later does not mean decoding 22 GB of DICOM again. A series with no labels row
is skipped and counted rather than assumed normal.

Both classes go through **one geometry**: the full frame resampled to 384×384 with its
aspect ratio kept, zero-padded, every slice retained, no cropping. This is the point of
the function. A corpus whose positives are lesion crops (45×72×70) and whose negatives
are full frames (2457×1890) is separable by array shape alone, which produces a splendid
AUC that measures the preprocessing. Downsampling averages rather than samples — at
2457 → 384 rows, picking one row in seven is how a small bright mass disappears — and the
same laterality flip is applied to negatives too, so "was flipped" cannot become a proxy
for "has a box", hence for the label. Measured: 4.9 MB per series compressed, ~14 s of
decoding each; `skip_existing=True` makes a pass resumable, which matters when a full one
runs for hours.

The `imaging/` package then trains the U-Net: it reads the `.npz` volumes, splits
them **by patient** (`case_id`) so no patient straddles train/val/test, serves axial
slices, and optimises a combined BCE + soft-Dice loss. Metrics (**Dice**, **IoU**)
go to `models/dbt/segmentation_metrics.csv` and the best checkpoint to
`models/dbt/unet_best.pt`. Because the masks are boxes rather than fine contours, this
targets lesion *localisation*, and the achievable Dice is inherently limited.
Requires `torch` (install the wheel matching your platform/CUDA). Validate the whole
loop without any data via `python -m imaging.train --smoke-test`.

## Evaluating a trained model

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

## Slice classifier

The segmentation U-Net cannot pick a lesion's slice out of a full volume (see
`plan.md` §4.2/§4.3). `imaging.sliceclf` trains a separate model for that ranking
task alone, on *every* slice rather than a sampled subset of negatives:

```bash
python -m imaging.sliceclf --slice-bank data/curated_data/slice_bank_p2 --epochs 25
```

It is selected on top-1 accuracy — "is the volume's highest-scoring slice really
lesion-bearing?" — the metric the segmentation confidence scored 0 on.

## Exam classifier (cancer / no-cancer, step 1 from the image)

`imaging.examclf` is step 1's actual decision head: cancer or not, from a **full**
DBT exam rather than a crop already centred on a lesion, using the label
`TransformData.preprocess_dbt_exams` writes for every series, annotated or not:

```bash
python -m imaging.examclf --data-dir data/preprocessed_data/dbt_exams --folds 5
```

The label is exam-level but the signal is not — most slices of a cancer exam show
nothing — so a bag's score is the **max** over a sample of its slices at training
time and over every one of its slices at evaluation (multiple-instance learning,
`imaging.exambank` paying the decompression cost of ~870 full exams once). Cross-
validates rather than holding out one test split, for the same reason as above: 5
folds stratified by patient, no checkpoint chosen on held-out patients, a patient's
score is the max over their own exams. Writes `models/examclf/cv_report.json` and
`cv_predictions.csv`; validate the loop with no dataset via
`python -m imaging.examclf --smoke-test`.

**Measured on 870 exams, 272 patients, 56 with a cancer, and it does not work**:
patient ROC-AUC 0.457 [0.369-0.544], and at threshold 0.5 the model calls every
single patient negative (sensitivity 0.0) — the same accuracy as always answering
"no cancer". A bigger MIL bag (32 vs 16 slices) was tried next and made it slightly
worse (0.414 [0.334-0.497]), not better. Detail, diagnosis and next leads: `plan.md`
§4.7 and "Prochaines pistes pour l'étape 1".

## The operating point, which is what an AUC does not tell you

An AUC is not a decision. A tool that answers "is there a cancer?" answers at **one
threshold**, and the pair to be judged on is the screening programme's — sensitivity
82.8 %, specificity 91.4 % (`plan.md`, "Cible chiffrée"). `imaging.oppoint` turns the
stored out-of-fold predictions into that pair, without re-training anything:

```bash
python -m imaging.oppoint          # writes reports/examclf_operating_point.{json,md}
```

The threshold is fixed at the target **sensitivity** — never chosen to maximise an
accuracy — and taken **out of fold**: each patient is judged by a threshold the other
four folds produced. Fitting it on the same 272 scores it then grades would report how
well a rule fitted to these patients describes these patients; `--naive` prints that
number too, labelled as not citable, so the gap can be read rather than asserted.

**The result closes the file on this head.** Sensitivity 78.6 % [67.2-88.9],
specificity 20.4 % [15.3-25.9], **PPV 20.4 % at a prevalence of 20.6 %** — the PPV *is*
the prevalence. Learning that the model said "cancer" does not change the probability
that there is one. At the sensitivity actually reached, chance would give 21.4 %
specificity; the model gives 20.4 %, below it. Full table and method: `plan.md` §4.11.
