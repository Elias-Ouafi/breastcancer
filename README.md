# Breast Cancer Detection Project

This project develops an app for breast cancer detection.
The user provide their MRI data and the app returns the probability of cancer.
WARNING : IT IS NOT CURRENTLY A VALID DIAGNOSTIC USABLE BY END USERS.

It uses two complementary angles:

1. **Quantitative / tabular** — the **Breast Cancer Wisconsin (Diagnostic)** dataset
   (UCI ML Repository), used to train and compare classical classifiers. This
   pipeline runs on **PySpark / Spark MLlib**.
2. **Medical imaging** — breast **MRI/DBT** DICOM series from **TCIA**
   (`Breast-Cancer-Screening-DBT` collection) with their segmentations, and the
   **BreakHis** histopathology images from Kaggle.

The long-term goal (see `Main.py`) is to combine the strongest features from both
the quantitative and imaging pipelines into a single model.

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
`Breast-Cancer-Screening-DBT` require no API key. Set the download location by
editing `DOWNLOAD_DIR` in `ExtractData.py`.

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
SVM, Gradient-Boosted Trees, and a Multilayer Perceptron). It writes results to
`results/` and figures to `plots/`. Requires a JVM (see Prerequisites). See
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
**annotation-driven**: fetch the boxes first, download only the annotated patients,
then turn each box into a mask.

```python
from ExtractData import download_annotated_dbt_series
from TransformData import preprocess_dbt_with_boxes

# 0. Get the boxes CSV(s) once into tciaDownload/ from TCIA. The training set
#    (BCS-DBT-boxes-train.csv, 101 patients) can be grown with the validation set
#    (BCS-DBT-boxes-validation.csv, 40 disjoint patients) — same schema. Both
#    functions accept a single path or a list of paths and pool them.
BOXES = [
    "tciaDownload/BCS-DBT-boxes-train.csv",
    "tciaDownload/BCS-DBT-boxes-validation.csv",
]

# 1. Download the DBT series of the annotated patients (cap the volume with max_gb).
#    max_patients=None fetches every annotated patient in the pooled CSVs.
download_annotated_dbt_series(
    BOXES, max_patients=None,
    download_dir="tciaDownload", max_gb=25,
)

# 2. Build a box mask per series and save compressed .npz (skips views with no box).
preprocess_dbt_with_boxes(
    root_dir="tciaDownload",
    boxes_csv=BOXES,
    output_dir="preprocessed_data",
)
```
```bash
# 3. Train the lesion-localisation U-Net on the preprocessed .npz volumes.
python -m imaging.train --data-dir preprocessed_data --epochs 25
```

Step 2 matches each downloaded series to its boxes by **PatientID + view**
(laterality from `FrameLaterality` + `ViewPosition`, e.g. `lmlo`), z-normalises the
image, paints the box(es) into a binary mask with `create_mask`, crops to the lesion
region of interest, and stores the real `PatientID` inside the `.npz` (as `case_id`).

The `imaging/` package then trains the U-Net: it reads the `.npz` volumes, splits
them **by patient** (`case_id`) so no patient straddles train/val/test, serves axial
slices, and optimises a combined BCE + soft-Dice loss. Metrics (**Dice**, **IoU**)
go to `results/segmentation_metrics.csv` and the best checkpoint to
`results/unet_best.pt`. Because the masks are boxes rather than fine contours, this
targets lesion *localisation*, and the achievable Dice is inherently limited.
Requires `torch` (install the wheel matching your platform/CUDA). Validate the whole
loop without any data via `python -m imaging.train --smoke-test`.

### Histopathology (BreakHis)
```bash
python ExtractBreakHis.py
```

## Acknowledgments

- [TCIA](https://www.cancerimagingarchive.net/) — MRI/DBT imaging data
- [UCI ML Repository](https://archive.ics.uci.edu/) — Breast Cancer Wisconsin dataset
- [Kaggle](https://www.kaggle.com/) — BreakHis histopathology dataset
