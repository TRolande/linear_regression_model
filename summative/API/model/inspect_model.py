"""
Script to inspect trained model artifacts, scaler, encoders, features, and metadata.
Run this script to view details of binary model files (.joblib) and metadata (.json).
"""

import json
from pathlib import Path
import joblib

# Determine model directory path
MODEL_DIR = Path(__file__).parent

def inspect_all():
    print("=" * 60)
    print("        LINEAR REGRESSION / MODEL ARTIFACT INSPECTOR        ")
    print("=" * 60)
    print()

    # 1. Load and display Model Metadata
    metadata_path = MODEL_DIR / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("1. MODEL METADATA ([model_metadata.json])")
        print("-" * 50)
        print(f"  Best Model Selected: {metadata.get('best_model_name')}")
        print(f"  Target Variable:     {metadata.get('target')}")
        print("\n  Evaluation Results across models:")
        for model_name, metrics in metadata.get("results", {}).items():
            print(f"    - {model_name:35s} | MSE: {metrics['mse']:.4f} | R^2: {metrics['r2']:.4f}")
        print()

    # 2. Load and display Feature Names
    features_path = MODEL_DIR / "feature_names.json"
    if features_path.exists():
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        print("2. INPUT FEATURES ([feature_names.json])")
        print("-" * 50)
        print(f"  Total Features Required: {len(features)}")
        for idx, feat in enumerate(features, 1):
            print(f"    {idx:2d}. {feat}")
        print()

    # 3. Load Best Model (.joblib)
    model_path = MODEL_DIR / "best_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        print("3. TRAINED MODEL DETAILS ([best_model.joblib])")
        print("-" * 50)
        print(f"  Model Type:        {type(model).__name__}")
        if hasattr(model, "n_estimators"):
            print(f"  Number of Trees:   {model.n_estimators}")
        if hasattr(model, "criterion"):
            print(f"  Criterion:         {model.criterion}")
        if hasattr(model, "feature_importances_") and 'features' in locals():
            print("\n  Top 5 Most Important Features:")
            importances = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
            for feat, imp in importances[:5]:
                print(f"    - {feat:30s}: {imp:.4f} ({imp*100:.1f}%)")
        print()

    # 4. Load Scaler (.joblib)
    scaler_path = MODEL_DIR / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print("4. FEATURE SCALER DETAILS ([scaler.joblib])")
        print("-" * 50)
        print(f"  Scaler Type:       {type(scaler).__name__}")
        if hasattr(scaler, "n_features_in_"):
            print(f"  Features Scaled:   {scaler.n_features_in_}")
        print()

    # 5. Load Banking Encoder (.joblib)
    encoder_path = MODEL_DIR / "banking_encoder.joblib"
    if encoder_path.exists():
        encoder = joblib.load(encoder_path)
        print("5. BANKING ENCODER DETAILS ([banking_encoder.joblib])")
        print("-" * 50)
        print(f"  Encoder Type:      {type(encoder).__name__}")
        if hasattr(encoder, "classes_"):
            print(f"  Encoded Classes:   {list(encoder.classes_)}")
        print()

    print("=" * 60)
    print("  Inspection Complete! All model components loaded cleanly.")
    print("=" * 60)

if __name__ == "__main__":
    inspect_all()
