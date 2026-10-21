"""
Simple feature importance extraction from the trained pipeline.
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"


def get_feature_importance(top_n: int = 15) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    prep = model.named_steps["prep"]
    clf = model.named_steps["model"]

    # Get feature names after one-hot encoding
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(clf.n_features_in_)]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).ravel()
    else:
        raise ValueError("Model does not expose feature importances or coefficients")

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    imp = get_feature_importance()
    print(imp.to_string(index=False))
