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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def load_data():
    """Load and preprocess the breast cancer dataset."""
    # Fetch dataset
    breast_cancer_data = fetch_ucirepo(id=17)
    
    # Extract features and target
    X = breast_cancer_data.data.features
    y = breast_cancer_data.data.targets
    
    # Convert target to binary (0 for benign, 1 for malignant)
    y = (y == 'M').astype(int)
    
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
    
    return X_train_pca, X_test_pca, y_train, y_test

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple classification models."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    results = {}
    
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
        
        print(f"Results for {name}:")
        for metric, value in results[name].items():
            print(f"{metric}: {value:.4f}")
    
    return results

def create_neural_network(input_dim):
    """Create and compile a neural network model."""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

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
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train and evaluate traditional ML models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Train and evaluate neural network
    nn_model = create_neural_network(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    
    y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
    y_prob_nn = nn_model.predict(X_test)
    
    results['Neural Network'] = {
        'accuracy': accuracy_score(y_test, y_pred_nn),
        'precision': precision_score(y_test, y_pred_nn),
        'recall': recall_score(y_test, y_pred_nn),
        'f1': f1_score(y_test, y_pred_nn),
        'roc_auc': roc_auc_score(y_test, y_prob_nn)
    }
    
    # Plot and save results
    plot_results(results)
    
    # Save results to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results.csv')
    print("\nResults saved to 'model_results.csv'")

if __name__ == "__main__":
    main() 