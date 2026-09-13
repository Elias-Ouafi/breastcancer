import logging
import os

import pandas as pd
import pydicom
from tcia_utils import nbia

import config
from logging_setup import setup_logging

log = logging.getLogger(__name__)

# Optional dependencies used only by specific extractors (Wisconsin fetch, plotting).
# Imported lazily so the DBT download path works without the full stack installed.
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None
try:
    import requests
except ImportError:
    requests = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def dir_size_bytes(path):
    """Return total size in bytes under `path`. If the path does not exist return 0."""
    total = 0
    if not os.path.exists(path):
        return 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
            except OSError:
                # Skip files that can't be accessed
                continue
    return total


def extract_breast_cancer_wisconsin_diagnostic_data(max_gb=30):
    """
    Fetches the Breast Cancer Wisconsin (Diagnostic) dataset and saves it to CSV
    under `config.WISCONSIN_DIR`, the raw layer for this source.
    Stops and does not save if the cumulative size there would exceed `max_gb` gigabytes.
    Also returns the raw data as a pandas DataFrame for future use (or None if not saved).
    """
    # Ensure the raw directory for this source exists
    target_dir = config.WISCONSIN_DIR
    os.makedirs(target_dir, exist_ok=True)

    # Respect storage cap
    max_bytes = int(max_gb * 1024 ** 3)
    current_size = dir_size_bytes(target_dir)
    if current_size >= max_bytes:
        log.info(f"Storage limit reached: {current_size} bytes >= {max_bytes} bytes ({max_gb} GB). Dataset will not be saved.")
        return None

    # Fetch the dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    
    # Extract features and target
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    
    # Combine features and target into a single DataFrame
    data = pd.concat([X, y], axis=1)

    # Save to a temp file first to measure size
    filename = 'raw_breast_cancer_data.csv'
    temp_path = os.path.join(target_dir, filename + '.tmp')
    final_path = os.path.join(target_dir, filename)

    try:
        data.to_csv(temp_path, index=False)
        file_size = os.path.getsize(temp_path)
    except Exception as e:
        log.error(f"Failed to write temporary CSV: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return None

    # Check if adding this file would exceed the cap
    current_size = dir_size_bytes(target_dir)
    if current_size + file_size > max_bytes:
        log.info(f"Saving this dataset would exceed the storage cap ({max_gb} GB). File of size {file_size} bytes will not be kept.")
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return None

    # Move temp file to final path
    try:
        os.replace(temp_path, final_path)
    except Exception as e:
        log.error(f"Failed to move temporary file into place: {e}")
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return None

    log.info("Wisconsin (Diagnostic) dataset extracted and saved to %s", final_path)
    return data

# Downloaded DICOM series land in the raw layer, untouched (see config.py).
DOWNLOAD_DIR = config.TCIA_DIR

def extract_dicom_mri_images(max_gb=30, max_series=None):
    """Extract breast cancer MRI images and store them in `DOWNLOAD_DIR`, skipping already downloaded series.

    Downloads one series at a time and stops when either the cumulative size in
    `DOWNLOAD_DIR` reaches `max_gb` gigabytes (default 30 GB) or `max_series` new
    series have been downloaded (default: no count limit). Use `max_series` to grab
    just a handful of cases for a quick test.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Get list of series to download
    all_series = nbia.getSeries(collection="Breast-Cancer-Screening-DBT")

    # Identify already downloaded series
    existing_series = {
        name for name in os.listdir(DOWNLOAD_DIR)
        if os.path.isdir(os.path.join(DOWNLOAD_DIR, name))
    }

    # Filter out already downloaded series
    new_series = [s for s in all_series if s['SeriesInstanceUID'] not in existing_series]

    if not new_series:
        log.info("All available series are already downloaded.")
        return

    max_bytes = int(max_gb * 1024 ** 3)
    current_size = dir_size_bytes(DOWNLOAD_DIR)
    if current_size >= max_bytes:
        log.info(f"Storage limit reached: {current_size} bytes >= {max_bytes} bytes ({max_gb} GB). No downloads will be performed.")
        return

    downloaded_count = 0
    for s in new_series:
        if max_series is not None and downloaded_count >= max_series:
            log.info(f"Reached series limit of {max_series}. Stopping further downloads.")
            break

        current_size = dir_size_bytes(DOWNLOAD_DIR)
        if current_size >= max_bytes:
            log.info(f"Reached download cap of {max_gb} GB. Stopping further downloads.")
            break

        series_uid = s.get('SeriesInstanceUID', '<unknown>')
        try:
            log.info(f"Attempting to download series {series_uid} to {DOWNLOAD_DIR}...")
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            nbia.downloadSeries([s], path=DOWNLOAD_DIR)

            downloaded_count += 1
            current_size = dir_size_bytes(DOWNLOAD_DIR)
            log.info(f"Downloaded series {series_uid}. Current storage used: {current_size} bytes.")
        except Exception as e:
            log.error(f"Failed to download series {series_uid}: {e}")
            continue

    log.info(f"Downloaded {downloaded_count} series to {DOWNLOAD_DIR} (cap was {max_gb} GB).")

def view_dicom_series(series_path):
    """View a DICOM series using pydicom and matplotlib with slice navigation."""
    # Get all DICOM files in the directory
    dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
    
    if not dicom_files:
        log.info(f"No DICOM files found in {series_path}")
        return
    
    # Read the first DICOM file
    dicom_path = os.path.join(series_path, dicom_files[0])
    ds = pydicom.dcmread(dicom_path)
    
    # Get the pixel data
    pixel_data = ds.pixel_array
    
    # Check if we have a 3D array (stack of images)
    if len(pixel_data.shape) == 3:
        num_slices = pixel_data.shape[0]
        log.info(f"Found {num_slices} slices in the DICOM stack")
        
        # Create a figure with a slider
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.2)  # Make room for the slider
        
        # Display the middle slice initially
        current_slice = num_slices // 2
        im = ax.imshow(pixel_data[current_slice], cmap=plt.cm.gray)
        ax.set_title(f'Slice {current_slice + 1} of {num_slices}')
        ax.axis('off')
        
        # Add a slider for slice navigation
        from matplotlib.widgets import Slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(
            ax=ax_slider,
            label='Slice',
            valmin=0,
            valmax=num_slices - 1,
            valinit=current_slice,
            valstep=1
        )
        
        def update(val):
            slice_idx = int(slider.val)
            im.set_data(pixel_data[slice_idx])
            ax.set_title(f'Slice {slice_idx + 1} of {num_slices}')
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        plt.show()
    else:
        # If it's a single 2D image, display it directly
        plt.figure(figsize=(10, 10))
        plt.imshow(pixel_data, cmap=plt.cm.gray)
        plt.title(f"DICOM Image: {os.path.basename(dicom_path)}")
        plt.axis('off')
        plt.show()

def clean_mri_annotation(folder_path=None, filename="Annotation_Boxes.xlsx"):
    """
    Convert Annotation_Boxes.xlsx into a csv, in place beside the series it annotates.
    """
    folder_path = folder_path or config.TCIA_DIR
    excel_path = os.path.join(folder_path, filename)
    csv_filename = filename.replace(".xlsx", ".csv")
    csv_path = os.path.join(folder_path, csv_filename)
    
    if not os.path.exists(excel_path):
        log.info(f"File not found: {excel_path}")
        return

    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False)

def download_segmentations(
    download_dir=DOWNLOAD_DIR,
    output_dir=DOWNLOAD_DIR,
    collection="Breast-Cancer-Screening-DBT",
    max_gb=30
):
    """
    Download RTSTRUCT/SEG segmentations for each DICOM series from TCIA.
    Uses patientId to get all series, then filters by StudyInstanceUID + modality.

    Stops downloading when `output_dir` reaches `max_gb` gigabytes.
    """

    os.makedirs(output_dir, exist_ok=True)
    existing = {d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d))}

    max_bytes = int(max_gb * 1024 ** 3)
    current_size = dir_size_bytes(output_dir)
    if current_size >= max_bytes:
        log.info(f"Storage limit reached ({current_size} bytes >= {max_bytes} bytes). No segmentations will be downloaded.")
        return

    for series_folder in os.listdir(download_dir):
        folder_path = os.path.join(download_dir, series_folder)
        if not os.path.isdir(folder_path):
            continue

        dicoms = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
        if not dicoms:
            log.warning(f"No DICOMs in {folder_path}, skipping.")
            continue

        try:
            dcm = pydicom.dcmread(os.path.join(folder_path, dicoms[0]), stop_before_pixels=True)
            series_uid = dcm.SeriesInstanceUID
            study_uid = dcm.StudyInstanceUID
            patient_id = dcm.PatientID
        except Exception as e:
            log.error(f"Reading DICOM in {folder_path}: {e}")
            continue

        log.info(f"[INFO] Series UID: {series_uid} | Study UID: {study_uid} | Patient ID: {patient_id}")

        try:
            all_series = nbia.getSeries(collection=collection, patientId=patient_id)
        except Exception as e:
            log.error(f"getSeries() failed for patient {patient_id}: {e}")
            continue

        # Filter to segmentations in the same study
        segmentations = [
            s for s in all_series
            if s['StudyInstanceUID'] == study_uid and s['Modality'] in {"SEG", "RTSTRUCT"}
        ]

        if not segmentations:
            log.info(f"No segmentations found for study {study_uid}")
            continue

        for seg in segmentations:
            seg_uid = seg['SeriesInstanceUID']
            if seg_uid in existing:
                log.info(f"Skipped: Segmentation {seg_uid} already downloaded.")
                continue

            current_size = dir_size_bytes(output_dir)
            if current_size >= max_bytes:
                log.info(f"Reached download cap of {max_gb} GB while downloading segmentations. Stopping.")
                return

            try:
                log.info(f"Segmentation {seg_uid} ({seg['Modality']})...")
                os.makedirs(output_dir, exist_ok=True)
                nbia.downloadSeries([seg], path=output_dir)

                existing.add(seg_uid)
                log.info(f"Downloaded {seg_uid}")
            except Exception as e:
                log.error(f"Failed to download {seg_uid}: {e}")

    log.info("✅ download_segmentations completed.")

def _read_boxes(boxes_csv):
    """Read one CSV path, or a list/tuple of paths, into a single boxes DataFrame.

    Accepting several CSVs lets the annotated-training set be grown by combining the
    BCS-DBT ``boxes-train`` and ``boxes-validation`` files (disjoint patients, same
    schema) without any per-file bookkeeping downstream.
    """
    paths = [boxes_csv] if isinstance(boxes_csv, (str, os.PathLike)) else list(boxes_csv)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


# The BCS-DBT tables, published beside the images rather than through the NBIA API, so
# `nbia` cannot fetch them. They were downloaded by hand until 2026-09-13, which meant a
# clone could not preprocess DBT at all: the inventory is now required for the box match.
# The published file name is not the local one: the collection page links
# `BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv` for what this repo keeps as
# `BCS-DBT-boxes-validation.csv` (checked byte-for-byte against the copy on disk).
DBT_TABLE_URLS = {
    config.DBT_BOXES_TRAIN: "BCS-DBT-boxes-train",
    config.DBT_BOXES_VALIDATION: "BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024",
    config.DBT_BOXES_TEST: "BCS-DBT-boxes-test-v2-PHASE-2-Jan-2024",
    config.DBT_FILE_PATHS_TRAIN: "BCS-DBT-file-paths-train-v2",
    config.DBT_FILE_PATHS_VALIDATION: "BCS-DBT-file-paths-validation-v2",
    config.DBT_FILE_PATHS_TEST: "BCS-DBT-file-paths-test-v2",
    config.DBT_LABELS_TRAIN: "BCS-DBT-labels-train-v2",
    config.DBT_LABELS_VALIDATION: "BCS-DBT-labels-validation-PHASE-2-Jan-2024",
    config.DBT_LABELS_TEST: "BCS-DBT-labels-test-PHASE-2",
}
DBT_TABLE_BASE_URL = "https://www.cancerimagingarchive.net/wp-content/uploads"


def download_dbt_tables(dest_dir=None, overwrite=False, base_url=DBT_TABLE_BASE_URL):
    """Fetch the BCS-DBT annotation tables: boxes, per-view labels, and file paths.

    Three kinds of table, and the project needs all three. The **boxes** give lesion
    coordinates and the ``Class`` (benign / cancer) that makes a positive a cancer. The
    **file paths** are the inventory that matches a box to a series folder -- without
    it, :func:`TransformData.preprocess_dbt_with_boxes` cannot say which box belongs to
    which series. The **labels** carry the per-view status, and are the only place that
    says an exam is *normal*: 4 581 of the 5 060 patients, against 89 with a cancer.

    Each response is checked to carry a ``PatientID`` header line before it is written:
    a wrong name answers 404 here (measured), but a CMS that answers a styled 200 page
    instead would otherwise land on disk as a ``.csv`` and fail much later, somewhere
    that says nothing about the download. Existing files are kept unless ``overwrite``.

    Returns the list of paths present afterwards.
    """
    if requests is None:
        raise ImportError("requests is required to download the BCS-DBT tables")
    dest_dir = dest_dir or config.TCIA_DIR
    os.makedirs(dest_dir, exist_ok=True)

    present = []
    for default_path, stem in DBT_TABLE_URLS.items():
        path = os.path.join(dest_dir, os.path.basename(default_path))
        if os.path.exists(path) and not overwrite:
            log.info(f"{os.path.basename(path)} already here; keeping it.")
            present.append(path)
            continue
        url = f"{base_url}/{stem}.csv"
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            text = response.content.decode("utf-8-sig")
        except Exception as e:
            log.error(f"{stem}: download failed ({e})")
            continue
        header = text.split("\n", 1)[0]
        if "PatientID" not in header:
            log.error(f"{stem}: the response is not a BCS-DBT table (first line: "
                      f"{header[:80]!r}); not written.")
            continue
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        rows = text.count("\n")
        log.info(f"{os.path.basename(path)}: {rows} rows -> {path}")
        present.append(path)

    missing = [os.path.basename(p) for p in DBT_TABLE_URLS
               if not os.path.exists(os.path.join(dest_dir, os.path.basename(p)))]
    if missing:
        log.warning(f"Still missing: {missing}")
    return present


def download_annotated_dbt_series(boxes_csv, max_patients=3, download_dir=DOWNLOAD_DIR,
                                  collection="Breast-Cancer-Screening-DBT", max_gb=None):
    """Download the DBT series for the first `max_patients` annotated patients.

    Most BCS-DBT volumes are normal (no lesion); only patients listed in the boxes
    CSV carry annotations. This reads that CSV, takes the first `max_patients`
    annotated patients, and downloads all their series (every view) so the box
    masks can later be matched in `preprocess_dbt_with_boxes`.

    `boxes_csv` may be a single path or a list of paths; passing both the train and
    validation boxes CSVs pools their (disjoint) annotated patients into one set.
    `max_patients=None` fetches every annotated patient. Downloading stops early once
    `download_dir` reaches `max_gb` gigabytes (if set), so you can cap the total
    volume regardless of the patient count.
    """
    boxes = _read_boxes(boxes_csv)
    patients = list(dict.fromkeys(boxes["PatientID"].tolist()))[:max_patients]
    log.info(f"Annotated patients to fetch: {len(patients)} (cap: {max_gb} GB)")
    return download_dbt_series_for(patients, download_dir=download_dir,
                                   collection=collection, max_gb=max_gb)


def download_dbt_series_for(patients, download_dir=DOWNLOAD_DIR,
                            collection="Breast-Cancer-Screening-DBT",
                            max_gb=None, max_gb_added=None):
    """Download every series of each patient in `patients`, series by series.

    Two different caps, because they answer different questions. `max_gb` is the size
    the whole `download_dir` may reach -- the historical behaviour, and a trap when the
    folder already holds other collections. `max_gb_added` caps what *this call* adds,
    which is what you want when the destination is shared. Either stops the run cleanly
    between series, leaving a partial but usable set.

    The size is measured once and then tracked by the bytes each series writes: walking
    the tree per series meant re-reading tens of thousands of files on every step.
    Returns the number of new series downloaded.
    """
    os.makedirs(download_dir, exist_ok=True)
    existing = {name for name in os.listdir(download_dir)
                if os.path.isdir(os.path.join(download_dir, name))}
    start_bytes = dir_size_bytes(download_dir)
    total_bytes = start_bytes
    max_bytes = int(max_gb * 1024 ** 3) if max_gb else None
    added_limit = int(max_gb_added * 1024 ** 3) if max_gb_added else None

    def over_budget():
        if max_bytes is not None and total_bytes >= max_bytes:
            log.info(f"Reached the {max_gb} GB total cap "
                     f"({total_bytes / 1024 ** 3:.1f} GB). Stopping downloads.")
            return True
        if added_limit is not None and total_bytes - start_bytes >= added_limit:
            log.info(f"Added {(total_bytes - start_bytes) / 1024 ** 3:.1f} GB, the cap "
                     f"for this run being {max_gb_added} GB. Stopping downloads.")
            return True
        return False

    downloaded = 0
    for pid in patients:
        if over_budget():
            break
        try:
            series = nbia.getSeries(collection=collection, patientId=pid)
        except Exception as e:
            log.error(f"getSeries failed for {pid}: {e}")
            continue
        for s in series:
            if over_budget():
                break
            uid = s.get("SeriesInstanceUID")
            if uid in existing:
                continue
            try:
                log.info(f"{pid} series {uid} -> {download_dir}")
                nbia.downloadSeries([s], path=download_dir)
                downloaded += 1
                existing.add(uid)
                total_bytes += dir_size_bytes(os.path.join(download_dir, uid))
            except Exception as e:
                log.error(f"download {uid} failed: {e}")

    log.info(f"Downloaded {downloaded} new series into {download_dir} "
             f"({(total_bytes - start_bytes) / 1024 ** 3:.1f} GB added, "
             f"{total_bytes / 1024 ** 3:.1f} GB total).")
    return downloaded


def download_normal_dbt_series(labels_csv=None, max_patients=150,
                               download_dir=DOWNLOAD_DIR,
                               collection="Breast-Cancer-Screening-DBT",
                               max_gb_added=50, seed=0):
    """Download the series of patients whose every view is labelled **normal**.

    This is the half of the corpus the project never had. Everything downloaded so far
    came from the boxes CSVs, so 100 % of the patients on disk carry a lesion, and a
    specificity cannot be measured on a corpus without negatives (plan.md, "Cible
    chiffrée"). The per-view labels table is what says an exam is normal: 4 581 of the
    5 060 patients, against 89 with a cancer.

    A patient is taken only if its **worst** status over every view of every study is
    ``normal`` (`TransformData.dbt_patient_status`), so a patient with one actionable
    view is not counted as a negative. The sample is drawn with `seed` rather than by
    ascending PatientID: the IDs are ordered by site and date, so the first N would be
    one corner of the collection. Downloading stops at `max_gb_added` gigabytes added by
    this call -- roughly 340 MB per patient (4 views) on what is on disk here.

    Returns the number of new series downloaded.
    """
    import random

    import TransformData

    labels_csv = labels_csv or [config.DBT_LABELS_TRAIN, config.DBT_LABELS_VALIDATION,
                                config.DBT_LABELS_TEST]
    status = TransformData.dbt_patient_status(labels_csv)
    normals = sorted(pid for pid, value in status.items() if value == "normal")
    random.Random(seed).shuffle(normals)
    chosen = normals[:max_patients]
    log.info(f"Normal patients available: {len(normals)}; fetching {len(chosen)} "
             f"(cap: {max_gb_added} GB added, seed {seed}).")
    return download_dbt_series_for(chosen, download_dir=download_dir,
                                   collection=collection, max_gb_added=max_gb_added)


def download_dce_mri_series(patient_ids=None, max_patients=10,
                            download_dir=os.path.join(config.TCIA_DIR, "duke_mri"),
                            collection="Duke-Breast-Cancer-MRI", series_filter="dyn",
                            max_gb=None):
    """Download the DCE-MRI dynamic series (pre + post-contrast passes) for patients
    in the Duke-Breast-Cancer-MRI collection.

    Each patient carries several series (a T1 pre-contrast pass and multiple
    post-contrast passes, e.g. ``ax dyn pre``, ``ax dyn 1st pass`` .. ``ax dyn 4th
    pass``) plus unrelated series (T1 TSE, SEG). ``series_filter`` keeps only series
    whose ``SeriesDescription`` contains this substring (case-insensitive), so only
    the multiphase DCE stack is pulled.

    ``patient_ids`` lets you target specific patients; when omitted the first
    ``max_patients`` patients in the collection (by ``PatientID`` sort order) are
    used. Downloading stops early once `download_dir` reaches `max_gb` gigabytes (if
    set). Each series lands in its own ``SeriesInstanceUID`` subfolder, same layout
    as `download_annotated_dbt_series`.
    """
    os.makedirs(download_dir, exist_ok=True)

    if patient_ids is None:
        all_series = nbia.getSeries(collection=collection)
        patient_ids = sorted({s["PatientID"] for s in all_series})[:max_patients]
    log.info(f"Patients to fetch: {len(patient_ids)} (cap: {max_gb} GB)")

    existing = {name for name in os.listdir(download_dir)
                if os.path.isdir(os.path.join(download_dir, name))}
    max_bytes = int(max_gb * 1024 ** 3) if max_gb else None

    downloaded = 0
    for pid in patient_ids:
        if max_bytes is not None and dir_size_bytes(download_dir) >= max_bytes:
            log.info(f"Reached {max_gb} GB cap. Stopping downloads.")
            break
        try:
            series = nbia.getSeries(collection=collection, patientId=pid)
        except Exception as e:
            log.error(f"getSeries failed for {pid}: {e}")
            continue

        dyn_series = [s for s in series
                     if series_filter.lower() in (s.get("SeriesDescription") or "").lower()]
        log.info(f"{pid}: {len(dyn_series)} DCE series matching '{series_filter}'")

        for s in dyn_series:
            if max_bytes is not None and dir_size_bytes(download_dir) >= max_bytes:
                log.info(f"Reached {max_gb} GB cap. Stopping downloads.")
                break
            uid = s.get("SeriesInstanceUID")
            if uid in existing:
                continue
            try:
                log.info(f"{pid} | {s.get('SeriesDescription')} | {uid} -> {download_dir}")
                nbia.downloadSeries([s], path=download_dir)
                existing.add(uid)
                downloaded += 1
            except Exception as e:
                log.error(f"download {uid} failed: {e}")

    total_gb = dir_size_bytes(download_dir) / 1024 ** 3
    log.info(f"Downloaded {downloaded} new series into {download_dir} ({total_gb:.1f} GB total).")
    return downloaded


if __name__ == "__main__":
    setup_logging(logfile="extract.log")
    # Download to the configured DOWNLOAD_DIR. Pass max_series/max_gb to limit volume.
    extract_dicom_mri_images()