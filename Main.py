import os
import pandas as pd
from ExtractData import extract_breast_cancer_wisconsin_diagnostic_data
from TransformData import transform_data
from AnalyzeData import analyze_data

def main():
    """Main function to run the entire breast cancer analytical project.
    It is composed of 2 main parts:
    1. Train the best AI model to predict breast cancer based quantitative values.
    2. Train an AI model to predict breast cancer based images.

    A future extension of this project will be to train an AI model to predict breast cancer using the best features from both parts.
    """
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("PART 1: Breast cancer analysis project based on quantitative values...")
    
    # Step 1: Extract data
    print("\nStep 1: Extract the data")
    raw_data = extract_breast_cancer_wisconsin_diagnostic_data()
    
    # Step 2: Transform data
    print("\nStep 2: Transform the data")
    transformed_data, pca, feature_contributions, top_features = transform_data(raw_data)
    
    # Step 3: Analyze data
    print("\nStep 3: Analyze the data")
    results = analyze_data(transformed_data)
    print("Analysis complete!")
    
    # Print final results
    print("\nAnalysis complete! Results summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name} Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    print("\nResults of the analysis have been saved to the 'results' directory.")

if __name__ == "__main__":
    main() 