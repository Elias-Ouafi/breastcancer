# Breast Cancer Detection Project

This project develops machine learning models for breast cancer detection from two
complementary angles:

1. **Quantitative / tabular** — the **Breast Cancer Wisconsin (Diagnostic)** dataset
   (UCI ML Repository), used to train and compare classical classifiers.
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
├── TransformData.py        # PCA on tabular data + MRI preprocessing/compression
├── AnalyzeData.py          # Train & compare classifiers on tabular data
├── Analysis.py             # Ensemble models (incl. XGBoost, VotingClassifier)
├── Main.py                 # Orchestrates the tabular pipeline end-to-end
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
This downloads the Wisconsin dataset, standardizes it, applies PCA (95% variance
retained), trains several classifiers, and writes results to `results/` and figures
to `plots/`. See `Final_Report.md` for a summary of the outcomes.

### Imaging pipeline (MRI/DBT)
```bash
# 1. Download DICOM series (and, separately, segmentations) from TCIA
python ExtractData.py

# 2. Preprocess every downloaded series into compressed arrays
python TransformData.py
```
Preprocessing resamples each series to isotropic 1 mm spacing, z-normalises the
intensities, builds a binary mask from the SEG/RTSTRUCT segmentations, and saves the
result.

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

- **Graphical interface for results visualization.** Build a GUI to explore the
  analysis outputs visually — model performance comparisons, PCA scree plots and
  feature contributions for the quantitative pipeline, and side-by-side viewing of
  MRI volumes with their segmentation masks for the imaging pipeline. The goal is an
  interactive dashboard so results can be reviewed without reading the raw CSVs or
  regenerating the static figures in `plots/`.
- **Combined model.** Merge the strongest quantitative and imaging features into a
  single predictive model (see `Main.py`).

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
