import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
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