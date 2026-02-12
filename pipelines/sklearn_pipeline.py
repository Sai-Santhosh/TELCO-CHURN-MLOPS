"""
Scikit-learn ML pipeline for Telco Churn - trains and saves model for inference + experimentation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare(data_path: str):
    """Load CSV and prepare for training."""
    df = pd.read_csv(data_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"].map(lambda x: 1 if str(x).strip().lower() in ("yes", "1") else 0)
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), cat_cols),
    ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ])
    
    feature_names = num_cols + [f"cat_{i}" for i in range(len(cat_cols) * 5)]
    return X, y, pipeline, num_cols, cat_cols


def main():
    data_path = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
    if not data_path.exists():
        data_path = Path("data/raw/Telco-Customer-Churn.csv")
    
    logger.info(f"Loading from {data_path}")
    X, y, pipeline, num_cols, cat_cols = load_and_prepare(str(data_path))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    weights = compute_sample_weight("balanced", y_train)
    pipeline.fit(X_train, y_train, classifier__sample_weight=weights)
    
    out_dir = ROOT / "artifacts" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = out_dir / "sklearn_pipeline_mlflow.joblib"
    joblib.dump(pipeline, model_path)
    
    feat_path = out_dir / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump({"numeric_features": num_cols, "categorical_features": cat_cols}, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    logger.info(f"Test ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
