import os
import kaggle
import pandas as pd
import shutil
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np
import time


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/breakhis_download.log'),
        logging.StreamHandler()
    ]
)

def download_breakhis(output_dir="data/BreakHis"):
    """
    Download BreakHis dataset from Kaggle.
    Note: This requires a Kaggle account and API credentials.
    Instructions for setting up Kaggle API:
    1. Go to https://www.kaggle.com/account
    2. Click on 'Create New API Token'
    3. Save kaggle.json to .kaggle/kaggle.json in the project directory
    """
    logging.info("Starting BreakHis dataset download...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Set Kaggle credentials path
        kaggle_credentials_path = os.path.join(os.getcwd(), '.kaggle', 'kaggle.json')
        if not os.path.exists(kaggle_credentials_path):
            raise Exception(f"Kaggle credentials not found at {kaggle_credentials_path}")
        
        # Set environment variable for Kaggle credentials
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')
        
        # Download dataset from Kaggle
        logging.info("Downloading BreakHis dataset from Kaggle...")
        start_time = time.time()
        
        # Download using Kaggle API
        kaggle.api.dataset_download_files(
            'ambarish/breakhis',
            path=output_dir,
            unzip=True
        )
        
        download_time = time.time() - start_time
        logging.info(f"Download completed in {download_time:.1f} seconds")
        
        # Process images and create metadata
        logging.info("Processing images and creating metadata...")
        process_images(output_dir)
        
        logging.info("BreakHis dataset processing complete!")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading BreakHis dataset: {str(e)}")
        return False

def process_images(base_dir):
    """Process images and create metadata CSV."""
    # Initialize metadata list
    metadata = []
    
    # Count total images first
    total_images = sum(1 for root, _, files in os.walk(base_dir) 
                      for file in files if file.endswith(('.png', '.jpg', '.jpeg')))
    logging.info(f"Found {total_images} images to process")
    
    # Process each image with progress bar
    processed = 0
    for root, _, files in tqdm(list(os.walk(base_dir)), desc="Processing directories"):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Get image path
                    image_path = os.path.join(root, file)
                    
                    # Extract information from filename
                    # Format: SOB_B_TA-14-4659-100-001.png
                    # SOB: Source of biopsy
                    # B: Benign
                    # TA: Tumor type
                    # 14: Patient ID
                    # 4659: Image ID
                    # 100: Magnification factor
                    # 001: Image number
                    filename_parts = os.path.splitext(file)[0].split('-')
                    if len(filename_parts) >= 4:
                        tumor_type = filename_parts[0].split('_')[2]
                        patient_id = filename_parts[1]
                        magnification = filename_parts[2]
                        image_number = filename_parts[3]
                        
                        # Determine if benign or malignant
                        is_benign = 'B' in filename_parts[0]
                        
                        # Read image to get dimensions
                        with Image.open(image_path) as img:
                            width, height = img.size
                        
                        # Add to metadata
                        metadata.append({
                            'image_path': image_path,
                            'tumor_type': tumor_type,
                            'patient_id': patient_id,
                            'magnification': magnification,
                            'image_number': image_number,
                            'is_benign': is_benign,
                            'width': width,
                            'height': height
                        })
                    
                    processed += 1
                    if processed % 100 == 0:  # Log progress every 100 images
                        logging.info(f"Processed {processed}/{total_images} images ({(processed/total_images*100):.1f}%)")
                    
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
                    continue
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(base_dir, "metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    
    # Print summary statistics
    logging.info(f"\nProcessing complete!")
    logging.info(f"Total images processed: {len(metadata)}")
    logging.info(f"Benign images: {len(metadata_df[metadata_df['is_benign']])}")
    logging.info(f"Malignant images: {len(metadata_df[~metadata_df['is_benign']])}")
    logging.info(f"Unique tumor types: {metadata_df['tumor_type'].nunique()}")
    logging.info(f"Unique patients: {metadata_df['patient_id'].nunique()}")
    logging.info(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    download_breakhis() 