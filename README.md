# Breast Cancer Detection Project

This project develops machine learning models for breast cancer detection from two
complementary angles:

1. **Quantitative / tabular** — the **Breast Cancer Wisconsin (Diagnostic)** dataset
   (UCI ML Repository), used to train and compare classical classifiers. This
   pipeline runs on **PySpark / Spark MLlib**.
2. **Medical imaging** — breast **MRI/DBT** DICOM series from **TCIA**
   (`Breast-Cancer-Screening-DBT` collection) with their segmentations, and the
   **BreakHis** histopathology images from Kaggle.

The long-term goal (see `Main.py`) is to combine the strongest features from both
the quantitative and imaging pipelines into a single model.

## Project Structure

```
breastcancer/
├── data/                   # Tabular data & outputs (gitignored)
├── plots/                  # Generated figures (model comparison, etc.)
├── results/                # Analysis results
├── ExtractData.py          # Wisconsin dataset + TCIA MRI/DBT download & segmentations
├── ExtractBreakHis.py      # BreakHis histopathology download (Kaggle)
├── TransformData.py        # Spark MLlib scaling + PCA (tabular) + MRI preprocessing/compression
├── AnalyzeData.py          # Train & compare Spark MLlib classifiers on tabular data
├── Main.py                 # Orchestrates the tabular (PySpark) pipeline end-to-end
├── Final_Report.md         # Report for the quantitative pipeline
├── requirements.txt        # Core Python dependencies
├── .gitignore
└── README.md               # This file
```

> **Note:** Downloaded imaging data (DICOM series, preprocessed arrays) is **not**
> stored in the repository. The default download location for MRI series is set in
> `ExtractData.py` (`DOWNLOAD_DIR`), currently an external drive (`D:\`), because the
> raw DICOM data is large.

## Prerequisites

- Python 3.8 or higher
- **Java 17 or newer (a JVM)** on the `PATH` — required by **PySpark 4** for the
  quantitative pipeline (e.g. Temurin/OpenJDK 17; set `JAVA_HOME`).
- **TCIA** access via `tcia_utils` / `nbia` (for the MRI/DBT dataset)
- A **Kaggle** account and API token (for the BreakHis dataset)

## Installation

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/yourusername/breastcancer.git
   cd breastcancer
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Unix / macOS
   source .venv/bin/activate
   ```
3. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The **MRI/DBT** pipeline additionally requires imaging libraries not in
   `requirements.txt`:
   ```bash
   pip install SimpleITK itk itkwidgets tcia_utils openpyxl
   ```

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

# 0. Get the boxes CSV once (BCS-DBT-boxes-train.csv) into tciaDownload/ from TCIA.

# 1. Download the DBT series of the annotated patients (cap the volume with max_gb).
download_annotated_dbt_series(
    "tciaDownload/BCS-DBT-boxes-train.csv", max_patients=101,
    download_dir="tciaDownload", max_gb=10,
)

# 2. Build a box mask per series and save compressed .npz (skips views with no box).
preprocess_dbt_with_boxes(
    root_dir="tciaDownload",
    boxes_csv="tciaDownload/BCS-DBT-boxes-train.csv",
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

## Managing data size

Medical imaging data is large, so the pipeline is set up to keep the footprint down:

- **Download cap.** Both `extract_dicom_mri_images` and `download_segmentations`
  accept a `max_gb` argument (default 30 GB) and stop once the target directory
  reaches that size. Lower it to download less, e.g. `extract_dicom_mri_images(max_gb=5)`.
- **Compressed, reduced-precision output.** `save_preprocessed` in `TransformData.py`
  stores each series as a single `*.npz` file that is:
  - **cropped** to the region of interest around the segmentation (`crop_to_roi`),
    with the crop origin saved as `crop_offset` so it can be mapped back;
  - stored as **float16** instead of float32 (half the size, negligible impact on
    z-normalised intensities);
  - **zlib-compressed** via `np.savez_compressed`, which shrinks the near-empty mask
    dramatically.

  Load a preprocessed series with:
  ```python
  import numpy as np
  data = np.load("preprocessed_data/<patient_id>.npz")
  volume, mask, offset = data["volume"], data["mask"], data["crop_offset"]
  ```

### Further reductions, if still too large
- **Delete raw DICOM after preprocessing.** Pass `delete_source=True` to
  `process_all_mri_data` (or `preprocess_mri_data`) to remove each original DICOM
  folder once it has been successfully written to `.npz`, reclaiming the bulk of the
  space. It is **destructive and off by default**: the deletion only happens after
  the `.npz` is confirmed present and non-empty, so a failed save never loses the
  source data. Verify the outputs first, then enable it.
- **Downsample.** Resampling to a coarser spacing (e.g. 1.5–2 mm) in
  `preprocess_mri_data` reduces voxel counts roughly with the cube of the spacing.
- **Fewer series.** Process a subset rather than the full collection.
- **Disable cropping** by calling `save_preprocessed(..., crop=False)` if you need the
  full field of view (files will be larger).

## Roadmap

The direction below is informed by the current state of the field (2025–2026). For
context, the **MASAI** randomised controlled trial (>100,000 women, published in *The
Lancet*, 2026) showed AI-supported mammography screening lifting cancer detection by
~29% with no rise in false positives and a ~44% cut in radiologist reading workload —
strong evidence that the detection/localisation → classification direction this
project follows is clinically worthwhile.

- **Graphical interface for results visualization.** Build a GUI to explore the
  analysis outputs visually — model performance comparisons, PCA scree plots and
  feature contributions for the quantitative pipeline, and side-by-side viewing of
  MRI/DBT volumes with their segmentation masks (and predicted boxes) for the imaging
  pipeline. The goal is an interactive dashboard so results can be reviewed without
  reading the raw CSVs or regenerating the static figures in `plots/`.
- **Imaging model, next steps.** The first brick is a 2D U-Net for lesion
  localisation (`imaging/`). Follow-ups, in roughly increasing effort:
  - **Two-stage detection.** The current single U-Net conflates localisation and
    scoring. Recent DBT pipelines separate the two: a detector (e.g. YOLO-family with
    an attention block such as CBAM) proposes lesion regions, then a CNN classifies
    each region. This tends to handle the heavy normal/lesion class imbalance in
    `Breast-Cancer-Screening-DBT` better than dense segmentation over box masks.
  - **Data-efficient 3D.** Move from per-slice 2D to volumetric modelling of the DBT
    stack. Rather than a full 3D U-Net trained from scratch (data-hungry on the small
    annotated subset), recent work adapts pretrained 2D backbones to 3D with little
    extra cost ("2D-to-3D" inflation / slice-aggregation), which is a better fit for
    the limited number of annotated patients here.
  - **Benign/malignant classifier** once pathology/biopsy labels are wired in, so the
    pipeline outputs a diagnosis and not only a location.
- **BreakHis histopathology pipeline.** A dedicated classification pipeline for the
  BreakHis images. The field has moved from training CNNs from scratch to reusing
  **pathology foundation-model features** (self-supervised models such as UNI, CONCH,
  Virchow) as frozen encoders with a lightweight classifier head — reported to reach
  ~97–98% accuracy on BreakHis with far less labelled data than end-to-end training.
  Start there rather than a bespoke CNN.
- **Combined model.** Merge the strongest quantitative and imaging features into a
  single predictive model (see `Main.py`). Prefer **intermediate, attention-based
  fusion** (e.g. cross-attention between imaging embeddings and tabular/clinical
  features) over naive concatenation of final scores — recent multimodal breast-cancer
  reviews consistently find representation-level fusion outperforms early- or
  late-fusion baselines, and it keeps the modalities interpretable.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## Acknowledgments

- [TCIA](https://www.cancerimagingarchive.net/) — MRI/DBT imaging data
- [UCI ML Repository](https://archive.ics.uci.edu/) — Breast Cancer Wisconsin dataset
- [Kaggle](https://www.kaggle.com/) — BreakHis histopathology dataset
