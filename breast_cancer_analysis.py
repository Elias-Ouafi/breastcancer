import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import os

def load_data():
    """Load and preprocess the breast cancer dataset."""
    # Fetch dataset
    breast_cancer_data = fetch_ucirepo(id=17)
    
    # Extract features and target
    X = breast_cancer_data.data.features
    y = breast_cancer_data.data.targets
    
    # Convert target to binary (0 for benign, 1 for malignant) and ensure 1D array
    y = (y == 'M').astype(int).values.ravel()
    
    # Save raw data to CSV
    raw_data = pd.concat([X, pd.Series(y, name='diagnosis')], axis=1)
    os.makedirs('data', exist_ok=True)
    raw_data.to_csv('data/raw_breast_cancer_data.csv', index=False)
    
    return X, y

def preprocess_data(X, y):
    """Preprocess the data including scaling and PCA."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Number of components after PCA: {pca.n_components_}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Save preprocessed data
    os.makedirs('data', exist_ok=True)
    
    # Save training data
    pd.DataFrame(X_train_pca).to_csv('data/X_train.csv', index=False)
    pd.Series(y_train, name='diagnosis').to_csv('data/y_train.csv', index=False)
    
    # Save test data
    pd.DataFrame(X_test_pca).to_csv('data/X_test.csv', index=False)
    pd.Series(y_test, name='diagnosis').to_csv('data/y_test.csv', index=False)
    
    return X_train_pca, X_test_pca, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple classification models."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        # Save predictions
        predictions[name] = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'probability': y_prob
        })
        
        print(f"Results for {name}:")
        for metric, value in results[name].items():
            print(f"{metric}: {value:.4f}")
    
    # Save model results and predictions
    results_df = pd.DataFrame(results).T
    results_df.to_csv('data/model_results.csv')
    
    for name, pred_df in predictions.items():
        pred_df.to_csv(f'data/predictions_{name.lower().replace(" ", "_")}.csv', index=False)
    
    return results

def plot_results(results):
    """Plot the performance metrics of different models."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    models = list(results.keys())
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = [results[model][metric] for model in models]
        sns.barplot(x=models, y=values)
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0.8, 1.0)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/model_comparison.png')
    plt.close()

def main():
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train and evaluate all models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Plot and save results
    plot_results(results)
    
    print("\nAll results and data have been saved to the 'data' and 'plots' directories.")

if __name__ == "__main__":
    main() 