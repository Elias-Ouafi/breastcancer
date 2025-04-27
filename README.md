# Breast Cancer Prediction System

This project implements a machine learning system for predicting breast cancer based on diagnostic data from the UCI Machine Learning Repository's Breast Cancer Wisconsin (Diagnostic) dataset.

## Project Structure

- `breast_cancer_analysis.py`: Main script for data processing and model training
- `database_operations.py`: Database operations using SQLAlchemy
- `database_setup.sql`: SQL script for database schema setup
- `requirements.txt`: Python dependencies

## Features

- Data preprocessing and cleaning
- Principal Component Analysis (PCA) for dimensionality reduction
- Multiple model implementations:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - Neural Network
- Comprehensive model evaluation metrics
- SQL database integration for data storage and retrieval
- Visualization of model performance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd breast-cancer-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the database:
```bash
psql -f database_setup.sql
```

## Usage

1. Run the main analysis script:
```bash
python breast_cancer_analysis.py
```

2. The script will:
   - Load and preprocess the data
   - Train multiple models
   - Generate performance metrics
   - Save results to CSV and database
   - Create visualization plots

## Database Configuration

Update the database connection URL in `database_operations.py`:
```python
DATABASE_URL = "postgresql://username:password@localhost/breast_cancer_db"
```

## Results

The analysis generates:
- `model_results.csv`: Detailed performance metrics for all models
- `model_comparison.png`: Visualization of model performance
- Database tables with patient data and model results

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License. 