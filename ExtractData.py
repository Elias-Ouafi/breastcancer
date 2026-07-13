import os
import pandas as pd
import numpy as np
import pydicom
from tcia_utils import nbia

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
    Fetches the Breast Cancer Wisconsin (Diagnostic) dataset and saves it to CSV.
    Stops and does not save if the cumulative size in `DOWNLOAD_DIR` would exceed `max_gb` gigabytes.
    Also returns the raw data as a pandas DataFrame for future use (or None if not saved).
    """
    # Ensure download directory exists
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Respect storage cap
    max_bytes = int(max_gb * 1024 ** 3)
    current_size = dir_size_bytes(DOWNLOAD_DIR)
    if current_size >= max_bytes:
        print(f"Storage limit reached: {current_size} bytes >= {max_bytes} bytes ({max_gb} GB). Dataset will not be saved.")
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
    temp_path = os.path.join(DOWNLOAD_DIR, filename + '.tmp')
    final_path = os.path.join(DOWNLOAD_DIR, filename)

    try:
        data.to_csv(temp_path, index=False)
        file_size = os.path.getsize(temp_path)
    except Exception as e:
        print(f"[ERROR] Failed to write temporary CSV: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return None

    # Check if adding this file would exceed the cap
    current_size = dir_size_bytes(DOWNLOAD_DIR)
    if current_size + file_size > max_bytes:
        print(f"Saving this dataset would exceed the storage cap ({max_gb} GB). File of size {file_size} bytes will not be kept.")
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return None

    # Move temp file to final path
    try:
        os.replace(temp_path, final_path)
    except Exception as e:
        print(f"[ERROR] Failed to move temporary file into place: {e}")
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return None

    print("Data from Breast Cancer Wisconsin (Diagnostic) dataset has been extracted and saved to", final_path)
    return data

# Default location for downloaded DICOM series (set by user request)
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

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
        print("All available series are already downloaded.")
        return

    max_bytes = int(max_gb * 1024 ** 3)
    current_size = dir_size_bytes(DOWNLOAD_DIR)
    if current_size >= max_bytes:
        print(f"Storage limit reached: {current_size} bytes >= {max_bytes} bytes ({max_gb} GB). No downloads will be performed.")
        return

    downloaded_count = 0
    for s in new_series:
        if max_series is not None and downloaded_count >= max_series:
            print(f"Reached series limit of {max_series}. Stopping further downloads.")
            break

        current_size = dir_size_bytes(DOWNLOAD_DIR)
        if current_size >= max_bytes:
            print(f"Reached download cap of {max_gb} GB. Stopping further downloads.")
            break

        series_uid = s.get('SeriesInstanceUID', '<unknown>')
        try:
            print(f"[DOWNLOAD] Attempting to download series {series_uid} to {DOWNLOAD_DIR}...")
            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
            nbia.downloadSeries([s], path=DOWNLOAD_DIR)

            downloaded_count += 1
            current_size = dir_size_bytes(DOWNLOAD_DIR)
            print(f"[INFO] Downloaded series {series_uid}. Current storage used: {current_size} bytes.")
        except Exception as e:
            print(f"[ERROR] Failed to download series {series_uid}: {e}")
            continue

    print(f"Downloaded {downloaded_count} series to {DOWNLOAD_DIR} (cap was {max_gb} GB).")

def view_dicom_series(series_path):
    """View a DICOM series using pydicom and matplotlib with slice navigation."""
    # Get all DICOM files in the directory
    dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
    
    if not dicom_files:
        print(f"No DICOM files found in {series_path}")
        return
    
    # Read the first DICOM file
    dicom_path = os.path.join(series_path, dicom_files[0])
    ds = pydicom.dcmread(dicom_path)
    
    # Get the pixel data
    pixel_data = ds.pixel_array
    
    # Check if we have a 3D array (stack of images)
    if len(pixel_data.shape) == 3:
        num_slices = pixel_data.shape[0]
        print(f"Found {num_slices} slices in the DICOM stack")
        
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

def clean_mri_annotation(folder_path="tciaDownload", filename="Annotation_Boxes.xlsx"):
    """
    Convert Annotation_Boxes.xlsx into a csv.
    """
    excel_path = os.path.join(folder_path, filename)
    csv_filename = filename.replace(".xlsx", ".csv")
    csv_path = os.path.join(folder_path, csv_filename)
    
    if not os.path.exists(excel_path):
        print(f"File not found: {excel_path}")
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
        print(f"Storage limit reached ({current_size} bytes >= {max_bytes} bytes). No segmentations will be downloaded.")
        return

    for series_folder in os.listdir(download_dir):
        folder_path = os.path.join(download_dir, series_folder)
        if not os.path.isdir(folder_path):
            continue

        dicoms = [f for f in os.listdir(folder_path) if f.lower().endswith(".dcm")]
        if not dicoms:
            print(f"[WARN] No DICOMs in {folder_path}, skipping.")
            continue

        try:
            dcm = pydicom.dcmread(os.path.join(folder_path, dicoms[0]), stop_before_pixels=True)
            series_uid = dcm.SeriesInstanceUID
            study_uid = dcm.StudyInstanceUID
            patient_id = dcm.PatientID
        except Exception as e:
            print(f"[ERROR] Reading DICOM in {folder_path}: {e}")
            continue

        print(f"\n[INFO] Series UID: {series_uid} | Study UID: {study_uid} | Patient ID: {patient_id}")

        try:
            all_series = nbia.getSeries(collection=collection, patientId=patient_id)
        except Exception as e:
            print(f"[ERROR] getSeries() failed for patient {patient_id}: {e}")
            continue

        # Filter to segmentations in the same study
        segmentations = [
            s for s in all_series
            if s['StudyInstanceUID'] == study_uid and s['Modality'] in {"SEG", "RTSTRUCT"}
        ]

        if not segmentations:
            print(f"[INFO] No segmentations found for study {study_uid}")
            continue

        for seg in segmentations:
            seg_uid = seg['SeriesInstanceUID']
            if seg_uid in existing:
                print(f"[SKIP] Segmentation {seg_uid} already downloaded.")
                continue

            current_size = dir_size_bytes(output_dir)
            if current_size >= max_bytes:
                print(f"Reached download cap of {max_gb} GB while downloading segmentations. Stopping.")
                return

            try:
                print(f"[DOWNLOAD] Segmentation {seg_uid} ({seg['Modality']})...")
                os.makedirs(output_dir, exist_ok=True)
                nbia.downloadSeries([seg], path=output_dir)

                existing.add(seg_uid)
                print(f"[SUCCESS] Downloaded {seg_uid}")
            except Exception as e:
                print(f"[ERROR] Failed to download {seg_uid}: {e}")

    print("\n✅ download_segmentations completed.")

def _read_boxes(boxes_csv):
    """Read one CSV path, or a list/tuple of paths, into a single boxes DataFrame.

    Accepting several CSVs lets the annotated-training set be grown by combining the
    BCS-DBT ``boxes-train`` and ``boxes-validation`` files (disjoint patients, same
    schema) without any per-file bookkeeping downstream.
    """
    paths = [boxes_csv] if isinstance(boxes_csv, (str, os.PathLike)) else list(boxes_csv)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


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
    os.makedirs(download_dir, exist_ok=True)
    boxes = _read_boxes(boxes_csv)
    patients = list(dict.fromkeys(boxes["PatientID"].tolist()))[:max_patients]
    print(f"Annotated patients to fetch: {len(patients)} (cap: {max_gb} GB)")

    existing = {name for name in os.listdir(download_dir)
                if os.path.isdir(os.path.join(download_dir, name))}
    max_bytes = int(max_gb * 1024 ** 3) if max_gb else None

    downloaded = 0
    for pid in patients:
        if max_bytes is not None and dir_size_bytes(download_dir) >= max_bytes:
            print(f"Reached {max_gb} GB cap. Stopping downloads.")
            break
        try:
            series = nbia.getSeries(collection=collection, patientId=pid)
        except Exception as e:
            print(f"[ERROR] getSeries failed for {pid}: {e}")
            continue
        for s in series:
            if max_bytes is not None and dir_size_bytes(download_dir) >= max_bytes:
                print(f"Reached {max_gb} GB cap. Stopping downloads.")
                break
            uid = s.get("SeriesInstanceUID")
            if uid in existing:
                continue
            try:
                print(f"[DOWNLOAD] {pid} series {uid} -> {download_dir}")
                nbia.downloadSeries([s], path=download_dir)
                downloaded += 1
            except Exception as e:
                print(f"[ERROR] download {uid} failed: {e}")

    total_gb = dir_size_bytes(download_dir) / 1024 ** 3
    print(f"Downloaded {downloaded} new series into {download_dir} ({total_gb:.1f} GB total).")
    return downloaded


if __name__ == "__main__":
    # Download to the configured DOWNLOAD_DIR. Pass max_series/max_gb to limit volume.
    extract_dicom_mri_images()