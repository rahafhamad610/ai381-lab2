"""The data contract for the churn model: one place that says which columns the model needs."""

import pandas as pd

NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"

# Columns that an upstream team renamed, mapped back to the names the model expects.
# Lab 1, Twist 1: churn_v2.csv renamed monthly_charges to monthly_fee.
RENAMES: dict[str, str] = {}


class SchemaError(ValueError):
    """Raised when a data file does not match the contract."""


def validate(df: pd.DataFrame, need_target: bool = True) -> pd.DataFrame:
    """Apply known renames, then fail loudly and clearly if a required column is missing."""
    df = df.rename(columns=RENAMES)
    required = FEATURES + ([TARGET] if need_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Schema check failed: missing columns {missing}. Columns found: {list(df.columns)}")
    return df
