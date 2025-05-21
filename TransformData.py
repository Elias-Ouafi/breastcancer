import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os
import pydicom

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

def preprocess_mri_data(data_dir="tciaDownload", annotation_csv="tciaDownload/Annotation_Boxes.csv", output_dir="preprocessed_data"):
    """
    Process all patient MRI volumes in `data_dir` using bounding boxes from `annotation_csv`,
    normalize them, generate binary tumor masks, and save as NumPy arrays in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    annotations = pd.read_csv(annotation_csv)

    for patient_id in annotations['Patient ID'].unique():
        patient_annotations = annotations[annotations['Patient ID'] == patient_id]
        patient_path = os.path.join(data_dir, str(patient_id))
        
        if not os.path.isdir(patient_path):
            print(f"⚠️ Skipping {patient_id}: Folder not found.")
            continue

        try:
            volume = load_dicom_volume(patient_path)
            mask = np.zeros_like(volume, dtype=np.uint8)

            for _, row in patient_annotations.iterrows():
                bbox = {
                    'Start Slice': int(row['Start Slice']),
                    'End Slice': int(row['End Slice']),
                    'Start Row': int(row['Start Row']),
                    'End Row': int(row['End Row']),
                    'Start Column': int(row['Start Column']),
                    'End Column': int(row['End Column']),
                }
                mask += create_mask(volume.shape, bbox)

            # Normalize volume to [0, 1]
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-5)

            # Save outputs
            np.save(os.path.join(output_dir, f"{patient_id}_volume.npy"), volume)
            np.save(os.path.join(output_dir, f"{patient_id}_mask.npy"), mask)

            print(f"✅ Processed {patient_id}")
        
        except Exception as e:
            print(f"❌ Failed {patient_id}: {e}")
        

preprocess_mri_data()