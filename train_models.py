import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from generate_dataset import generate_synthetic_ohlcv
from data_processor import compute_technical_indicators, FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_and_evaluate():
    dataset_path = os.path.join(BASE_DIR, "dataset", "btc_usdt_ohlcv.csv")
    if not os.path.exists(dataset_path):
        print("[Train] Generating fresh dataset...")
        df_raw = generate_synthetic_ohlcv(dataset_path)
    else:
        df_raw = pd.read_csv(dataset_path)
        
    print("[Train] Computing technical indicators...")
    df = compute_technical_indicators(df_raw)
    df = df.dropna().reset_index(drop=True)
    
    X = df[FEATURE_COLUMNS]
    y = df['Target']
    
    # Strictly Chronological Train/Test Split (80% Train, 20% Test) - Anti-Leakage Requirement
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"[Train] Chronological Split: Train size={len(X_train)}, Test size={len(X_test)}")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds).tolist()
        
        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    print(f"[Train] Best performing model: {best_model_name} (F1 Score: {best_f1})")
    
    # Feature Importances of Best Model
    feature_importances = {}
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        for feat, imp in zip(FEATURE_COLUMNS, importances):
            feature_importances[feat] = float(round(imp, 4))
    elif hasattr(best_model_obj, 'coef_'):
        importances = np.abs(best_model_obj.coef_[0])
        importances = importances / np.sum(importances)
        for feat, imp in zip(FEATURE_COLUMNS, importances):
            feature_importances[feat] = float(round(imp, 4))
            
    # Sort feature importances descending
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

    # Save artifacts
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.pkl"))
    
    metrics_payload = {
        "best_model_name": best_model_name,
        "model_comparison": results,
        "feature_importances": sorted_importances,
        "feature_columns": FEATURE_COLUMNS,
        "test_timestamps": df['Date'].iloc[split_idx:].tolist(),
        "test_close_prices": df['Close'].iloc[split_idx:].tolist(),
        "test_actual_targets": y_test.tolist(),
        "test_predictions": [int(p) for p in best_model_obj.predict(X_test)],
        "dataset_split_info": {
            "total_rows": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_range": f"{df['Date'].iloc[0]} to {df['Date'].iloc[split_idx-1]}",
            "test_range": f"{df['Date'].iloc[split_idx]} to {df['Date'].iloc[-1]}"
        }
    }
    
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)
        
    print("[Train] Successfully trained models and saved artifacts to backend/models/")

if __name__ == "__main__":
    train_and_evaluate()
