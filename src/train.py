import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os
import json

from .data_generator import generate_loan_data
from .features import engineer_features
from .model_registry import ModelRegistry

def train_model():
    """
    End-to-end training pipeline for the Loan Approval model.
    """
    # 1. Setup paths
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, 'data', 'loan_data.csv')
    model_dir = os.path.join(project_root, 'models')
    
    # 2. Data Preparation
    if not os.path.exists(data_path):
        generate_loan_data(data_path)
    
    df = pd.read_csv(data_path)
    X = engineer_features(df)
    y = df['approved']
    
    # 3. Scaling & Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Training
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluation
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }
    
    # 6. Registry
    registry = ModelRegistry(model_dir)
    registry.save_model(model, scaler, metrics, version="v1")
    
    print("Training complete. Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    train_model()
