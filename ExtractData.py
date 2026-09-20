import logging
import os

import pandas as pd
import pydicom
from tcia_utils import nbia

import config
import http_timeouts

log = logging.getLogger(__name__)

# tcia_utils sends every request without a timeout, so one stalled response hangs a
# download forever (2 h 30 on 2026-09-14). Installed at import so every download path
# of this module inherits it; see http_timeouts for why a socket default does not work.
http_timeouts.install(nbia)


# Optional dependency used only by the series viewer. Imported lazily so the download path
# works without the plotting stack installed.
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


# Downloaded DICOM series land in bronze, untouched (see config.py).
DOWNLOAD_DIR = config.TCIA_DIR

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
    set). Each series lands in its own ``SeriesInstanceUID`` subfolder, which is the
    layout ``TransformData.group_dce_series_by_patient`` and ``mri_nnunet`` read.
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
