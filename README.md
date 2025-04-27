# Breast Cancer Prediction System

This project implements a machine learning system for predicting breast cancer based on diagnostic data from the UCI Machine Learning Repository's Breast Cancer Wisconsin (Diagnostic) dataset.

## Project Structure

- `TransformData.py`: Handles data loading, cleaning, preprocessing, and PCA
- `Analysis.py`: Implements model training, evaluation, and ensemble methods
- `requirements.txt`: Python dependencies
- `data/`: Directory for storing data files
- `plots/`: Directory for storing visualization plots

## Features

### Data Transformation (`TransformData.py`)
- Data loading and cleaning
- Feature standardization
- Principal Component Analysis (PCA)
- Data saving to CSV format

### Analysis (`Analysis.py`)
- Multiple model implementations:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - Neural Network
- Ensemble method (Voting Classifier)
- Comprehensive model evaluation metrics
- Visualization of model performance

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. First, run the data transformation script:
```bash
python TransformData.py
```
This will:
- Load and clean the data
- Apply PCA
- Save transformed data to CSV files

2. Then, run the analysis script:
```bash
python Analysis.py
```
This will:
- Train and evaluate individual models
- Train and evaluate an ensemble model
- Generate performance metrics and visualizations
- Save all results to CSV files

## Output Files

### Data Files (`data/` directory)
- `raw_breast_cancer_data.csv`: Original dataset
- `transformed_data.csv`: Preprocessed data after PCA
- `pca_info.csv`: PCA analysis results
- `individual_model_results.csv`: Performance metrics for individual models
- `ensemble_results.csv`: Performance metrics for ensemble model
- `predictions_*.csv`: Prediction results for each model

### Visualization Files (`plots/` directory)
- `model_comparison.png`: Performance comparison of all models

## Results Analysis

The analysis generates comprehensive metrics for each model:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The ensemble model combines the predictions of all individual models to potentially improve overall performance.

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License. 