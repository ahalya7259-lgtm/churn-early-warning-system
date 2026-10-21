"""
Train churn prediction models and save the best one.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

from data_processing import load_raw, clean, get_feature_target, save_processed

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def train():
    print("Loading and cleaning data...")
    raw = load_raw()
    df = clean(raw)
    save_processed(df)

    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    candidates = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
    }

    best_name = None
    best_auc = 0
    best_pipeline = None

    for name, model in candidates.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"\n=== {name.upper()} ===")
        print(f"ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, pipe.predict(X_test), digits=3))

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipe

    model_path = MODELS_DIR / "churn_model.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")
    print(f"Saved → {model_path}")

    # Also save feature names for later explanation
    meta = {
        "model_name": best_name,
        "roc_auc": best_auc,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(meta, MODELS_DIR / "model_meta.joblib")
    print("Metadata saved.")


if __name__ == "__main__":
    train()
