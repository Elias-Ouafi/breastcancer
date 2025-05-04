import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_data(data):
    """Prepare data for model training."""
    # Separate features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }
        
        # Add ROC AUC if probabilities are available
        if y_prob is not None:
            metrics['ROC AUC'] = roc_auc_score(y_test, y_prob)
        
        results[name] = metrics
    
    return results

def create_model_comparison_plot(results, save_path='plots/model_comparison.png'):
    """Create and save a comparison plot of model performance."""
    # Convert results to DataFrame
    metrics_df = pd.DataFrame(results).T
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', rot=45)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def analyze_data(data):
    """Main function to analyze the data and evaluate models."""
    # Prepare data
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(data)
    
    # Train and evaluate models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Create visualization
    create_model_comparison_plot(results)
    
    return results 