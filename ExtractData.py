import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import requests
from tcia_utils import nbia
import pydicom
import matplotlib.pyplot as plt

def extract_breast_cancer_wisconsin_diagnostic_data():
    """
    Fetches the Breast Cancer Wisconsin (Diagnostic) dataset and saves it to CSV.
    Also returns the raw data as a pandas DataFrame for future use.
    """
    # Fetch the dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    
    # Extract features and target
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    
    # Combine features and target into a single DataFrame
    data = pd.concat([X, y], axis=1)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save raw data to CSV
    data.to_csv('data/raw_breast_cancer_data.csv', index=False)
    
    print("Data from Breast Cancer Wisconsin (Diagnostic) dataset has been extracted!")
    
    return data

DOWNLOAD_DIR = 'tciaDownload'

def extract_dicom_mri_images():
    """Extract breast cancer MRI images and store them in the folder tciaDownload, skipping already downloaded series."""
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

    # Download only new series
    nbia.downloadSeries(new_series)
    print(f"Downloaded {len(new_series)} new series to {DOWNLOAD_DIR}")

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
    download_dir="tciaDownload",
    output_dir="tciaSegmentations",
    collection="Breast-Cancer-Screening-DBT"
):
    """
    Download RTSTRUCT/SEG segmentations for each DICOM series from TCIA.
    Uses patientId to get all series, then filters by StudyInstanceUID + modality.
    """

    os.makedirs(output_dir, exist_ok=True)
    existing = {d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d))}

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

            try:
                print(f"[DOWNLOAD] Segmentation {seg_uid} ({seg['Modality']})...")
                nbia.downloadSeries([seg], output_dir=output_dir)
                existing.add(seg_uid)
                print(f"[SUCCESS] Downloaded {seg_uid}")
            except Exception as e:
                print(f"[ERROR] Failed to download {seg_uid}: {e}")

    print("\n✅ download_segmentations completed.")

# Use default directories
download_segmentations()