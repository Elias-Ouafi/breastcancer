import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def prepare_data(train_df, test_df):
    """Prepare already-split, already-scaled/PCA'd train and test data for
    model training. The split and the scaling/PCA fitting both happened
    upstream in TransformData.transform_data(), fit on the training set
    only - this function does not re-split or re-fit anything on the test
    set, it just separates features/target and encodes the label (fit on
    train, applied to test) so no test information leaks into training."""
    X_train = train_df.drop('Diagnosis', axis=1)
    y_train_raw = train_df['Diagnosis']
    X_test = test_df.drop('Diagnosis', axis=1)
    y_test_raw = test_df['Diagnosis']

    # Encode target variable: fit on train only, apply same mapping to test
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    return X_train.values, X_test.values, y_train, y_test, label_encoder

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

def analyze_data(train_df, test_df):
    """Main function to analyze the data and evaluate models.

    train_df/test_df must already be the leak-free, PCA-transformed splits
    produced by TransformData.transform_data() - test_df is only ever used
    here for final evaluation, never for fitting."""
    # Prepare data
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(train_df, test_df)

    # Train and evaluate models
    results = train_models(X_train, X_test, y_train, y_test)

    # Create visualization
    create_model_comparison_plot(results)

    return results 