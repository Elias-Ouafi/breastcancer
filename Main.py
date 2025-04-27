import os
from ExtractData import extract_data
from TransformData import transform_data
from Analysis import analyze_data

def main():
    """Main function to run the entire breast cancer analysis project."""
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("Starting breast cancer analysis project...")
    
    # Step 1: Extract data
    print("\nStep 1: Extracting data...")
    raw_data = extract_data()
    
    # Step 2: Transform data
    print("\nStep 2: Transforming data...")
    transformed_data, pca = transform_data(raw_data)
    
    # Step 3: Analyze data
    print("\nStep 3: Analyzing data...")
    results = analyze_data(transformed_data)
    
    # Print final results
    print("\nAnalysis complete! Results summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name} Model Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    print("\nAll results and visualizations have been saved to their respective directories.")

if __name__ == "__main__":
    main() 