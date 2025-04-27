import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

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
    
    return X_scaled, y

def apply_pca(X_scaled):
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
    
    return X_pca, pca

def save_transformed_data(X_pca, y, pca):
    """Save the transformed data and PCA information."""
    # Save transformed data
    transformed_data = pd.DataFrame(X_pca)
    transformed_data['Diagnosis'] = y
    transformed_data.to_csv('data/transformed_data.csv', index=False)
    
    # Save PCA information
    pca_info = pd.DataFrame({
        'component': range(1, pca.n_components_ + 1),
        'explained_variance': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    })
    pca_info.to_csv('data/pca_info.csv', index=False)
    
    return transformed_data

def transform_data(data):
    """Transform the data through cleaning and PCA."""
    # Separate features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    
    # Clean and preprocess data
    X_scaled, y = clean_data(X, y)
    
    # Apply PCA
    X_pca, pca = apply_pca(X_scaled)
    
    # Save transformed data
    transformed_data = save_transformed_data(X_pca, y, pca)
    
    print("\nData transformation complete!")
    print(f"Original number of features: {X.shape[1]}")
    print(f"Number of features after PCA: {X_pca.shape[1]}")
    print("\nTransformed data saved to 'data/transformed_data.csv'")
    print("PCA information saved to 'data/pca_info.csv'")
    
    return transformed_data, pca 