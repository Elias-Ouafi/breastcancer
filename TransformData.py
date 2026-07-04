import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import pydicom
import SimpleITK as sitk
from tcia_utils import nbia
import logging
from pathlib import Path
import itk
import itkwidgets
from itkwidgets import view

def clean_data(X, y):
    """Clean and preprocess the data."""
    # Check for missing values
    missing_values = X.isnull().sum()
    if missing_values.any():
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        # Handle missing values (if any)
        X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def analyze_feature_contributions(pca, feature_names):
    """Analyze and return feature contributions to principal components."""
    # Get the absolute values of the components
    components = np.abs(pca.components_)
    
    # Create a DataFrame to store feature contributions
    feature_contributions = pd.DataFrame(
        components,
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    
    # For each PC, get the top 3 contributing features
    top_features = {}
    for pc in feature_contributions.index:
        top_features[pc] = feature_contributions.loc[pc].nlargest(3).to_dict()
    
    return feature_contributions, top_features

def create_scree_plot(pca, save_path='data/scree_plot.png'):
    """Create and save a scree plot of explained variance."""
    plt.figure(figsize=(10, 6))
    
    # Plot individual explained variance
    plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_,
            alpha=0.5, align='center', label='Individual explained variance')
    
    # Plot cumulative explained variance
    plt.step(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_),
             where='mid', label='Cumulative explained variance')
    
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Scree Plot')
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def apply_pca(X_scaled, feature_names):
    """Apply PCA and analyze component importance."""
    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    
    # Analyze PCA results
    print("\nPCA Analysis:")
    print(f"Number of components: {pca.n_components_}")
    print("\nExplained variance ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"Component {i+1}: {ratio:.4f}")
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print("\nCumulative explained variance:")
    for i, var in enumerate(cumulative_variance):
        print(f"Components 1-{i+1}: {var:.4f}")
    
    # Analyze feature contributions
    feature_contributions, top_features = analyze_feature_contributions(pca, feature_names)
    
    # Print top contributing features for each component
    print("\nTop contributing features for each principal component:")
    for pc, features in top_features.items():
        print(f"\n{pc}:")
        for feature, contribution in features.items():
            print(f"  {feature}: {contribution:.4f}")
    
    # Create scree plot
    create_scree_plot(pca)
    
    return X_pca, pca, feature_contributions, top_features

def save_transformed_data(X_pca, y, pca, feature_contributions):
    """Save the transformed data and PCA information."""
    # Save transformed data
    transformed_data = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    transformed_data['Diagnosis'] = y
    transformed_data.to_csv('data/transformed_data.csv', index=False)
    
    # Save PCA information
    pca_info = pd.DataFrame({
        'component': range(1, pca.n_components_ + 1),
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    })
    pca_info.to_csv('data/pca_info.csv', index=False)
    
    # Save feature contributions
    feature_contributions.to_csv('data/feature_contributions.csv')
    
    return transformed_data

def transform_data(data):
    """Transform the data through cleaning and PCA."""
    # Separate features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    feature_names = X.columns.tolist()
    
    # Clean and preprocess data
    X_scaled, y, scaler = clean_data(X, y)
    
    # Apply PCA
    X_pca, pca, feature_contributions, top_features = apply_pca(X_scaled, feature_names)
    
    # Save transformed data
    transformed_data = save_transformed_data(X_pca, y, pca, feature_contributions)
    
    print("\nData transformation complete!")
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after PCA: {X_pca.shape[1]}")
    print("\nTransformed data saved to 'data/transformed_data.csv'")
    print("PCA information saved to 'data/pca_info.csv'")
    print("Feature contributions saved to 'data/feature_contributions.csv'")
    print("Scree plot saved to 'data/scree_plot.png'")
    
    return transformed_data, pca, feature_contributions, top_features 


def load_dicom_volume(dicom_dir):
    """
    Load a 3D volume from a folder of DICOM slices.
    Sorts slices by InstanceNumber.
    """
    dicom_files = sorted(
        [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')],
        key=lambda f: int(pydicom.dcmread(os.path.join(dicom_dir, f)).InstanceNumber)
    )
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)).pixel_array for f in dicom_files]
    volume = np.stack(slices, axis=0)  # Shape: (depth, height, width)
    return volume

def create_mask(volume_shape, bbox):
    """
    Create a binary mask from bounding box coordinates.
    bbox keys: Start Slice, End Slice, Start Row, End Row, Start Column, End Column
    """
    mask = np.zeros(volume_shape, dtype=np.uint8)
    z0, z1 = bbox['Start Slice'], bbox['End Slice']
    y0, y1 = bbox['Start Row'], bbox['End Row']
    x0, x1 = bbox['Start Column'], bbox['End Column']
    mask[z0:z1, y0:y1, x0:x1] = 1
    return mask


def crop_to_roi(volume, mask, margin=16):
    """Crop `volume` and `mask` to the mask's bounding box plus a voxel `margin`.

    Segmentation masks are almost entirely background, so storing the full volume
    wastes space. When the mask is empty we cannot infer a region of interest, so
    the arrays are returned unchanged.
    Returns (cropped_volume, cropped_mask, offset) where `offset` is the (z, y, x)
    index of the crop origin in the original volume, so the crop can be located
    back in the full image later.
    """
    if not mask.any():
        return volume, mask, (0, 0, 0)

    nonzero = np.argwhere(mask)
    start = np.maximum(nonzero.min(axis=0) - margin, 0)
    end = np.minimum(nonzero.max(axis=0) + margin + 1, mask.shape)

    slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
    return volume[slices], mask[slices], tuple(int(s) for s in start)


def save_preprocessed(patient_id, volume, mask, output_dir, dtype=np.float16, crop=True):
    """Save a preprocessed MRI volume + mask as a single compressed .npz file.

    Three levers keep the files small:
      - `crop`: keep only the region of interest around the segmentation.
      - `dtype`: store intensities as float16 (half the size of float32); the
        precision loss is negligible for z-normalised MRI data.
      - `np.savez_compressed`: zlib-compresses the arrays; the mostly-empty mask
        shrinks by orders of magnitude.
    """
    os.makedirs(output_dir, exist_ok=True)

    offset = (0, 0, 0)
    if crop:
        volume, mask, offset = crop_to_roi(volume, mask)

    volume = volume.astype(dtype)
    mask = mask.astype(np.uint8)

    out_path = os.path.join(output_dir, f"{patient_id}.npz")
    np.savez_compressed(
        out_path,
        volume=volume,
        mask=mask,
        crop_offset=np.asarray(offset, dtype=np.int32),
    )
    return out_path


def delete_dicom_source(dicom_dir, npz_path):
    """Delete the raw DICOM folder `dicom_dir` after preprocessing.

    Destructive — this permanently removes the original series. As a safety net the
    deletion is skipped (with a warning) unless `npz_path` exists and is non-empty,
    so a failed or partial save never costs you the source data.
    Returns True if the folder was removed.
    """
    if not npz_path or not os.path.exists(npz_path) or os.path.getsize(npz_path) == 0:
        logging.warning(
            f"Skipping deletion of {dicom_dir}: preprocessed file "
            f"{npz_path} is missing or empty."
        )
        return False
    try:
        shutil.rmtree(dicom_dir)
        logging.info(f"🗑️  Removed raw DICOM source {dicom_dir} (kept {npz_path}).")
        return True
    except OSError as e:
        logging.error(f"Failed to remove {dicom_dir}: {e}")
        return False


root_directory = '/tciaDownload'

def extract_patient_ids(root_dir):
    patient_ids = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                filepath = os.path.join(dirpath, filename)
                try:
                    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                    patient_ids.add(ds.PatientID)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    return patient_ids

# Usage

patient_ids = extract_patient_ids(root_directory)
print("Extracted patient IDs:")
for pid in patient_ids:
    print(pid)


def process_all_mri_data(root_dir="tciaDownload", output_dir="preprocessed_data", delete_source=False):
    """
    Loop through all MRI data to preprocess them.

    Set `delete_source=True` to remove each raw DICOM folder after it has been
    successfully preprocessed into a compressed .npz (reclaims most of the disk
    space). It is off by default because it is destructive.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    
    # Store all subdirectories in a list to loop through
    mri_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    processed_count = 0
    failed_count = 0
    
    for mri_dir in mri_dirs:
        try:
            logging.info(f"Processing MRI directory: {mri_dir}")
            local_dicom_path = os.path.join(root_dir, mri_dir)
            
            # Process the MRI data
            volume, mask = preprocess_mri_data(
                series_instance_uid=mri_dir,
                local_dicom_path=local_dicom_path,
                output_dir=output_dir,
                delete_source=delete_source
            )
            
            if volume is not None and mask is not None:
                processed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            logging.error(f"Failed to process {mri_dir}: {str(e)}")
            failed_count += 1
            continue
    
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Failed to process: {failed_count}")
    return processed_count, failed_count

def preprocess_mri_data(series_instance_uid, local_dicom_path, output_dir="preprocessed_data", delete_source=False):
    """
    Preprocess MRI data to add its segmentations.

    If `delete_source` is True, the raw DICOM folder is deleted once the compressed
    .npz has been written successfully (see `delete_dicom_source`).
    """
    try:
        # STEP 1 - Load the MRI series
        logging.info(f"\n Loading MRI series...")
        dicom_files = [f for f in os.listdir(local_dicom_path) if f.endswith('.dcm')]
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {local_dicom_path}")
            
        # STEP 2 -Read DICOM images with single or multiple files
        if len(dicom_files) == 1:
            logging.info("Single DICOM file.")
            image = sitk.ReadImage(os.path.join(local_dicom_path, dicom_files[0]))
        else:
            logging.info("Multiple DICOM files.")
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(local_dicom_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        
        # STEP 3 - Convert images to numpy array first, then to float32
        image_array = sitk.GetArrayFromImage(image)
        image_array = image_array.astype(np.float32)
        image = sitk.GetImageFromArray(image_array)
        # Use the right spacing and size
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        standard_spacing = (1.0, 1.0, 1.0)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(standard_spacing)
        resampler.SetSize([int(sz * spc / nspc) for sz, spc, nspc in zip(original_size, original_spacing, standard_spacing)])
        resampler.SetInterpolator(sitk.sitkLinear)
        resampled_image = resampler.Execute(image)
        
        # STEP 4 - Convert to numpy array for processing
        image_array = sitk.GetArrayFromImage(resampled_image)
        
        # STEP 5 - Normalize the array
        mean = np.mean(image_array)
        std = np.std(image_array)
        normalized_array = (image_array - mean) / (std + 1e-8)
        mask = np.zeros_like(normalized_array, dtype=np.uint8)
        
        # STEP 6 - Segment the data
        seg_files = [f for f in os.listdir(local_dicom_path) if f.endswith('.dcm')]
        for seg_file in seg_files:
            seg_path = os.path.join(local_dicom_path, seg_file)
            try:
                # Read DICOM file
                ds = pydicom.dcmread(seg_path)
                
                # Check if it's a segmentation file
                if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4':
                    logging.info(f"Processing SEG file: {seg_file}")
                    seg_image = sitk.ReadImage(seg_path)
                    # Convert to numpy array first, then to float32
                    seg_array = sitk.GetArrayFromImage(seg_image)
                    seg_array = seg_array.astype(np.float32)
                    seg_image = sitk.GetImageFromArray(seg_array)
                    resampled_seg = resampler.Execute(seg_image)
                    seg_array = sitk.GetArrayFromImage(resampled_seg)
                    mask = np.logical_or(mask, seg_array > 0).astype(np.uint8)
                    
                elif ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3':  # RTSTRUCT
                    logging.info(f"Processing RTSTRUCT file: {seg_file}")
                    # Convert RTSTRUCT to binary mask
                    rtstruct = itk.imread(seg_path)
                    rtstruct_resampled = itk.resample_image_filter(
                        rtstruct,
                        size=resampled_image.GetSize(),
                        spacing=standard_spacing
                    )
                    rtstruct_array = itk.GetArrayFromImage(rtstruct_resampled)
                    mask = np.logical_or(mask, rtstruct_array > 0).astype(np.uint8)
                    
            except Exception as e:
                logging.warning(f"Could not process segmentation file {seg_file}: {str(e)}")
                continue
        
        # STEP 7 - Save preprocessed data (compressed to keep files small)
        patient_id = os.path.basename(local_dicom_path)
        npz_path = save_preprocessed(patient_id, normalized_array, mask, output_dir)
        if mask.any():
            view(normalized_array, mask, ui_collapsed=True)

        # STEP 8 - Optionally reclaim disk space by deleting the raw DICOM source
        if delete_source:
            delete_dicom_source(local_dicom_path, npz_path)

        # The end
        logging.info(f"✅ Successfully processed {patient_id}")
        return normalized_array, mask
        
    except Exception as e:
        logging.error(f"❌ Failed to process series {series_instance_uid}: {str(e)}")
        return None, None

# Example usage
if __name__ == "__main__":
    # delete_source=True also removes each raw DICOM folder after it is safely
    # preprocessed. It is destructive, so keep it False until you have verified the
    # compressed .npz outputs are correct.
    processed_count, failed_count = process_all_mri_data(
        root_dir="tciaDownload",
        output_dir="preprocessed_data",
        delete_source=False
    )