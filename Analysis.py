import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import os

def load_transformed_data():
    """Load the transformed data from CSV."""
    data = pd.read_csv('data/transformed_data.csv')
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    return X, y

def train_individual_models(X_train, X_test, y_train, y_test):
    """Train and evaluate individual models."""
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
    
    return results, predictions, models

def train_ensemble_model(models, X_train, X_test, y_train, y_test):
    """Train an ensemble model using voting classifier."""
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    
    print("\nTraining Ensemble Model (Voting Classifier)...")
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    y_prob = voting_clf.predict_proba(X_test)[:, 1]
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_prob
    })
    
    print("Results for Ensemble Model:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results, predictions

def plot_results(individual_results, ensemble_results):
    """Plot the performance metrics of different models."""
    # Combine individual and ensemble results
    all_results = {**individual_results, 'Ensemble': ensemble_results}
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    models = list(all_results.keys())
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        values = [all_results[model][metric] for model in models]
        sns.barplot(x=models, y=values)
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0.8, 1.0)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/model_comparison.png')
    plt.close()

def save_results(individual_results, ensemble_results, individual_predictions, ensemble_predictions):
    """Save all results to CSV files."""
    # Save individual model results
    results_df = pd.DataFrame(individual_results).T
    results_df.to_csv('data/individual_model_results.csv')
    
    # Save ensemble results
    ensemble_df = pd.DataFrame([ensemble_results], index=['Ensemble'])
    ensemble_df.to_csv('data/ensemble_results.csv')
    
    # Save predictions
    for name, pred_df in individual_predictions.items():
        pred_df.to_csv(f'data/predictions_{name.lower().replace(" ", "_")}.csv', index=False)
    
    ensemble_predictions.to_csv('data/predictions_ensemble.csv', index=False)

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models on the data."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
    
    return results

def plot_feature_importance(model, feature_names, output_path):
    """Plot feature importance for the Random Forest model."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_data(data):
    """Analyze the data and train models."""
    # Separate features and target
    X = data.drop('Diagnosis', axis=1)
    y = data['Diagnosis']
    
    # Train and evaluate models
    results = train_and_evaluate_models(X, y)
    
    # Plot feature importance for Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X, y)
    plot_feature_importance(rf_model, X.columns, 'results/feature_importance.png')
    
    return results 