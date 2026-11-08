"""Lab 3: train several candidate models and record every run in MLflow.

    docker compose -f docker-compose.mlflow.yml up -d --wait     # the tracking server
    python -m src.train.train_mlflow

Each candidate becomes one MLflow run that records:
  parameters  which model and which settings          metrics   AUC, F1 overall and per slice, p95 latency, size
  data hash   which exact data file it learned from   signature what goes in and what comes out
  the model   so it can be loaded and served later    tags      which batch and which commit produced it

Your work is marked TODO. Everything else is plumbing that is already done, so spend your time on the decisions.
"""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import schema

DATA_FILE = os.getenv("DATA_FILE", "data/churn_v1.csv")
EXPERIMENT = "churn"
SLICE_COLUMN = "contract"  # F1 is also reported for each contract type; TODO: change it if you prefer another slice
BATCH = os.getenv("BATCH", os.getenv("GIT_SHA", "local")[:7])  # runs trained together share a batch tag


def preprocessing() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), schema.NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), schema.CATEGORICAL),
        ]
    )


# TODO 1: choose at least three candidates and justify them in MODEL_SELECTION.md.
# Change the settings, not just the names: a different model family or a clearly different setting each time.
CANDIDATES = {
    "logreg_c05": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    "forest_depth6": RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=1
    ),
    "boosting_depth2": GradientBoostingClassifier(
        n_estimators=80, max_depth=2, learning_rate=0.1, random_state=42
    ),
}


def p95_latency_ms(model, X: pd.DataFrame, n: int = 200) -> float:
    """How long one prediction takes, as the service would do it: one row at a time."""
    times = []
    for i in range(n):
        row = X.iloc[[i % len(X)]]
        start = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 95))


def size_kb(model) -> float:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.joblib"
        joblib.dump(model, path)
        return path.stat().st_size / 1024


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)

    df = schema.validate(pd.read_csv(DATA_FILE))
    X, y = df[schema.FEATURES], df[schema.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    data_sha = hashlib.sha256(Path(DATA_FILE).read_bytes()).hexdigest()[:12]

    for name, clf in CANDIDATES.items():
        model = Pipeline([("prep", preprocessing()), ("clf", clf)]).fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = proba > 0.5

        with mlflow.start_run(run_name=name) as run:
            # parameters: what this run is
            mlflow.log_param("model", name)
            mlflow.log_params({f"clf__{k}": v for k, v in clf.get_params().items()})
            mlflow.log_param("data_file", DATA_FILE)
            mlflow.log_param("data_sha256", data_sha)  # TODO 2 (explain in your video): why log a hash of the data?
            mlflow.set_tags({"batch": BATCH, "git_sha": os.getenv("GIT_SHA", "local")})

            # metrics: how good it is
            metrics = {
                "auc": roc_auc_score(y_te, proba),
                "f1": f1_score(y_te, pred),
                "p95_latency_ms": p95_latency_ms(model, X_te),
                "size_kb": size_kb(model),
            }
            for value in X_te[SLICE_COLUMN].unique():
                mask = (X_te[SLICE_COLUMN] == value).to_numpy()
                metrics[f"f1_{SLICE_COLUMN}_{value}"] = f1_score(y_te[mask], pred[mask], zero_division=0)
            mlflow.log_metrics({k: round(float(v), 4) for k, v in metrics.items()})

            # the model, with its signature (inputs and outputs) and an example row
            signature = infer_signature(X_te, proba)
            from importlib.metadata import version
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_te.head(3),
                serialization_format="cloudpickle",
                pip_requirements=[
                    f"{pkg}=={version(pkg)}"
                    for pkg in [
                        "scikit-learn", "cloudpickle", "pandas",
                        "numpy", "scipy", "joblib"
                    ]
                ],
            )
            mlflow.set_tag("model_uri", info.model_uri)

            print(f"{name:18s} run {run.info.run_id[:8]}  auc {metrics['auc']:.3f}  f1 {metrics['f1']:.3f}  "
                  f"p95 {metrics['p95_latency_ms']:.1f} ms  size {metrics['size_kb']:.0f} KB")

    print(json.dumps({"batch": BATCH, "experiment": EXPERIMENT, "data_sha256": data_sha}))


if __name__ == "__main__":
    main()
