"""
Load the trained model and score new customers.
"""

from pathlib import Path
import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python src/train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def predict_proba(df: pd.DataFrame) -> pd.Series:
    """
    df must contain the same feature columns used during training
    (excluding the target 'Churn').
    Returns a Series of churn probabilities.
    """
    model = load_model()
    proba = model.predict_proba(df)[:, 1]
    return pd.Series(proba, index=df.index, name="churn_probability")


def predict_with_risk(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    proba = predict_proba(df)
    result = df.copy()
    result["churn_probability"] = proba
    result["risk_label"] = (proba >= threshold).map({True: "High", False: "Low"})
    return result.sort_values("churn_probability", ascending=False)


if __name__ == "__main__":
    # Example: score the processed training data (for demo)
    processed = Path(__file__).resolve().parents[1] / "data" / "processed" / "telco_clean.csv"
    if not processed.exists():
        print("No processed data found. Run train_model.py first.")
    else:
        df = pd.read_csv(processed)
        features = df.drop(columns=["Churn"], errors="ignore")
        scored = predict_with_risk(features)
        print(scored[["churn_probability", "risk_label"]].head(10))
