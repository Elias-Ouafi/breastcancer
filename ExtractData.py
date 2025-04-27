import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

def extract_data():
    """
    Fetches the Breast Cancer Wisconsin (Diagnostic) dataset and saves it to CSV.
    Returns the raw data as a pandas DataFrame.
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
    
    print("Data extraction complete!")
    print(f"Raw data saved to 'data/raw_breast_cancer_data.csv'")
    
    return data
