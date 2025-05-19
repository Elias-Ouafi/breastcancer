import requests
import pandas as pd
from tcia_utils import nbia
import pydicom
import matplotlib.pyplot as plt
import os
import numpy as np


def extract_dicom_series():
    """Extract breast cancer MRI images and store them in the folder tciaDownload"""
    data = nbia.getSeries(collection = "Breast-Cancer-Screening-DBT")
    nbia.downloadSeries(data)


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