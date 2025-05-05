import os
import pandas as pd
import requests
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/download.log'),
        logging.StreamHandler()
    ]
)

class TCIAAPI:
    """Class to interact with TCIA REST API."""
    
    def __init__(self):
        self.base_url = "https://services.cancerimagingarchive.net/services/v4"
        self.collection = "CBIS-DDSM"
        self.api_key = self._get_api_key()
    
    def _get_api_key(self):
        """Get API key from environment variable or prompt user."""
        api_key = os.getenv('TCIA_API_KEY')
        if not api_key:
            logging.warning("TCIA_API_KEY environment variable not set.")
            logging.info("Please register at https://www.cancerimagingarchive.net/tcia-portal/access-tcia/")
            logging.info("After registration, set your API key as an environment variable:")
            logging.info("Windows: set TCIA_API_KEY=your_api_key")
            logging.info("Linux/Mac: export TCIA_API_KEY=your_api_key")
            return None
        return api_key
    
    def get_series(self):
        """Get all series for the CBIS-DDSM collection."""
        endpoint = f"{self.base_url}/TCIA/query/getSeries"
        params = {
            "Collection": self.collection,
            "format": "json"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            logging.info(f"Fetching series from {endpoint}")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            series_data = response.json()
            logging.info(f"Found {len(series_data)} series")
            return series_data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching series: {str(e)}")
            if response.status_code == 401:
                logging.error("Unauthorized access. Please check your API key.")
            elif response.status_code == 403:
                logging.error("Access forbidden. Please check your API key and permissions.")
            return []
    
    def download_series(self, series_uid, output_dir):
        """Download a specific series."""
        endpoint = f"{self.base_url}/TCIA/query/getImage"
        params = {
            "SeriesInstanceUID": series_uid,
            "format": "json"
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            logging.info(f"Fetching images for series {series_uid}")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            image_data = response.json()
            if not image_data:
                logging.warning(f"No images found for series {series_uid}")
                return False
            
            # Create series directory
            series_dir = os.path.join(output_dir, series_uid)
            os.makedirs(series_dir, exist_ok=True)
            
            # Download each image in the series
            for image in tqdm(image_data, desc=f"Downloading series {series_uid}"):
                try:
                    image_url = image['URL']
                    image_filename = os.path.join(series_dir, f"{image['SOPInstanceUID']}.dcm")
                    
                    # Skip if file already exists
                    if os.path.exists(image_filename):
                        logging.info(f"File {image_filename} already exists, skipping...")
                        continue
                    
                    # Download image
                    image_response = requests.get(image_url)
                    image_response.raise_for_status()
                    
                    # Save DICOM file
                    with open(image_filename, 'wb') as f:
                        f.write(image_response.content)
                    
                    logging.info(f"Downloaded {image_filename}")
                    
                except Exception as e:
                    logging.error(f"Error downloading image {image['SOPInstanceUID']}: {str(e)}")
                    continue
            
            return True
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading series {series_uid}: {str(e)}")
            if response.status_code == 401:
                logging.error("Unauthorized access. Please check your API key.")
            elif response.status_code == 403:
                logging.error("Access forbidden. Please check your API key and permissions.")
            return False

def download_cbis_ddsm(output_dir="data/CBIS_DDSM", max_series=5):
    """Download CBIS-DDSM dataset from TCIA."""
    logging.info("\nStarting CBIS-DDSM dataset download...")
    
    # Create download directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize TCIA API client
        tcia = TCIAAPI()
        
        # Get all series
        series = tcia.get_series()
        if not series:
            logging.error("No series found!")
            return False
        
        # Download specified number of series
        successful_downloads = 0
        for i, entry in enumerate(series[:max_series]):
            series_uid = entry['SeriesInstanceUID']
            logging.info(f"\nProcessing series {i+1}/{max_series}: {series_uid}")
            
            success = tcia.download_series(series_uid, output_dir)
            if success:
                successful_downloads += 1
        
        logging.info(f"\nDownload complete! {successful_downloads}/{max_series} series downloaded successfully.")
        logging.info(f"Data saved to {output_dir}")
        
        return successful_downloads > 0
        
    except Exception as e:
        logging.error(f"Error downloading CBIS-DDSM dataset: {str(e)}")
        return False

def preprocess_images(image_dir, output_dir="data/processed_images"):
    """Preprocess downloaded DICOM images."""
    logging.info("\nStarting image preprocessing...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    processed_data = []
    for root, _, files in tqdm(list(os.walk(image_dir)), desc="Processing images"):
        for file in files:
            if file.endswith('.dcm'):
                try:
                    # Read DICOM image
                    dicom_path = os.path.join(root, file)
                    dicom = pydicom.dcmread(dicom_path)
                    
                    # Convert to numpy array
                    image_array = dicom.pixel_array
                    
                    # Normalize to 0-255
                    image_array = ((image_array - image_array.min()) / 
                                 (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                    
                    # Convert to PIL Image
                    image = Image.fromarray(image_array)
                    
                    # Save processed image
                    output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.png")
                    image.save(output_path)
                    
                    # Add metadata to processed_data list
                    processed_data.append({
                        'image_path': output_path,
                        'original_path': dicom_path,
                        'patient_id': dicom.PatientID if 'PatientID' in dicom else os.path.basename(root),
                        'study_date': dicom.StudyDate if 'StudyDate' in dicom else None,
                        'modality': dicom.Modality if 'Modality' in dicom else None,
                        'body_part': dicom.BodyPartExamined if 'BodyPartExamined' in dicom else None
                    })
                    
                    logging.info(f"Processed {file}")
                    
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(processed_data)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    
    logging.info(f"\nPreprocessing complete! Processed images saved to {output_dir}")
    logging.info(f"Metadata saved to {metadata_path}")
    
    return metadata_df

def extract_cbis_ddsm_data(max_series=5):
    """Main function to download and preprocess CBIS-DDSM data."""
    # Download data
    download_success = download_cbis_ddsm(max_series=max_series)
    
    if not download_success:
        logging.error("Failed to download data. Check the logs for details.")
        return None
    
    # Preprocess images
    metadata = preprocess_images("data/CBIS_DDSM")
    
    return metadata

if __name__ == "__main__":
    # Example usage
    metadata = extract_cbis_ddsm_data(max_series=5)
    if metadata is not None:
        logging.info("\nCBIS-DDSM data extraction complete!")
        logging.info(f"Total processed images: {len(metadata)}") 