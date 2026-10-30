from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "churn_v1.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"

NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]


def train():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC + CATEGORICAL]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessing = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

    model = Pipeline([
        ("prep", preprocessing),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"F1: {f1_score(y_test, probabilities > 0.5):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
