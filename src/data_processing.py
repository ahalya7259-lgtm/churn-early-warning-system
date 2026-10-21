"""
Data loading and preprocessing for the Telco Customer Churn dataset.
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "telco_clean.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download Telco Customer Churn from Kaggle and save it as data/raw/telco_churn.csv"
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # TotalCharges is often stored as object with blanks
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Binary target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID if present (not useful for modeling)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Simple binary encoding for common Yes/No columns
    yes_no_cols = [
        c for c in df.columns
        if df[c].dropna().isin(["Yes", "No"]).all() and c != "Churn"
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df


def get_feature_target(df: pd.DataFrame):
    """Return X (features) and y (target)."""
    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' not found in dataframe")
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    return X, y


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data → {path}")


if __name__ == "__main__":
    raw = load_raw()
    print("Raw shape:", raw.shape)
    cleaned = clean(raw)
    print("Clean shape:", cleaned.shape)
    print("Churn rate: {:.1%}".format(cleaned["Churn"].mean()))
    save_processed(cleaned)
