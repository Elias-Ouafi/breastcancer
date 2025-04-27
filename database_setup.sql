-- Create database
CREATE DATABASE breast_cancer_db;

-- Connect to the database
\c breast_cancer_db;

-- Create table for patient data
CREATE TABLE patient_data (
    id SERIAL PRIMARY KEY,
    radius_mean FLOAT,
    texture_mean FLOAT,
    perimeter_mean FLOAT,
    area_mean FLOAT,
    smoothness_mean FLOAT,
    compactness_mean FLOAT,
    concavity_mean FLOAT,
    concave_points_mean FLOAT,
    symmetry_mean FLOAT,
    fractal_dimension_mean FLOAT,
    diagnosis CHAR(1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for model results
CREATE TABLE model_results (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX idx_diagnosis ON patient_data(diagnosis);
CREATE INDEX idx_model_name ON model_results(model_name);

-- Create view for model performance comparison
CREATE VIEW model_performance_comparison AS
SELECT 
    model_name,
    accuracy,
    precision,
    recall,
    f1_score,
    roc_auc,
    created_at
FROM model_results
ORDER BY accuracy DESC; 