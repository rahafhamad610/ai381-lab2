"""Lab 3: the promotion gate. Only a better model becomes the champion.

    python scripts/promote.py                      # compares this batch's best candidate with the champion
    python scripts/promote.py --batch abc1234      # a specific batch (default: the BATCH or GIT_SHA environment variable)

How it decides (you write the thresholds, see the TODOs):
  1. Pick the best candidate of the batch by F1 among those that are fast enough (p95 latency budget).
  2. Compare it with the current champion, whose numbers live in artifacts/champion.json (a CI runner forgets
     everything between runs, so the champion's numbers are kept in Git). In CI the workflow passes the champion
     from the base branch with --champion-file, so a pull request is judged against what is on main today.
  3. Better by at least MIN_GAIN F1 points -> promote: register the model in MLflow, set the alias "champion",
     export it to artifacts/ (this is what the Dockerfile packages) and write the new artifacts/champion.json.
     Not better -> exit code 1. In CI that is a red run: the gate blocked the model.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

REGISTERED_NAME = "churn-model"
ARTIFACTS = Path("artifacts")
CHAMPION_FILE = ARTIFACTS / "champion.json"
LATENCY_BUDGET_MS = 50.0
MIN_GAIN = 0.01


def best_candidate(batch: str) -> dict:
    runs = mlflow.search_runs(
        experiment_names=["churn"], filter_string=f"tags.batch = '{batch}'", output_format="list"
    )
    fast = [r for r in runs if r.info.status == "FINISHED" and r.data.tags.get("model_uri") and r.data.metrics.get("p95_latency_ms", 1e9) <= LATENCY_BUDGET_MS]
    if not runs:
        sys.exit(f"No runs found for batch '{batch}'. Did train_mlflow.py run against the same MLflow server?")
    if not fast:
        sys.exit(f"BLOCKED: no candidate is within the {LATENCY_BUDGET_MS} ms latency budget.")
    run = max(fast, key=lambda r: r.data.metrics["f1"])
    return {
        "run_id": run.info.run_id,
        "model": run.data.params["model"],
        "f1": run.data.metrics["f1"],
        "auc": run.data.metrics["auc"],
        "p95_latency_ms": run.data.metrics["p95_latency_ms"],
        "data_sha256": run.data.params["data_sha256"],
        "model_uri": run.data.tags["model_uri"],
    }


def current_champion(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def promote(candidate: dict) -> None:
    client = MlflowClient()
    version = mlflow.register_model(candidate["model_uri"], REGISTERED_NAME)
    client.set_registered_model_alias(REGISTERED_NAME, "champion", version.version)

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib")
    meta = {
        "model_version": os.getenv("GIT_SHA", "local")[:7],  # the Dockerfile overwrites this with the image's commit
        "mlflow_run_id": candidate["run_id"],
        "mlflow_model_version": int(version.version),
        "model": candidate["model"],
        "data_sha256": candidate["data_sha256"],
        "auc": round(candidate["auc"], 3),
        "f1": round(candidate["f1"], 3),
        "p95_latency_ms": round(candidate["p95_latency_ms"], 2),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(meta, indent=2))
    CHAMPION_FILE.write_text(json.dumps(meta, indent=2))
    print(f"PROMOTED: {candidate['model']} is the new champion (MLflow version {version.version}).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=os.getenv("BATCH", os.getenv("GIT_SHA", "local")[:7]))
    ap.add_argument("--champion-file", type=Path, default=CHAMPION_FILE, help="the champion to beat")
    a = ap.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    candidate, champion = best_candidate(a.batch), current_champion(a.champion_file)
    print(f"best candidate: {candidate['model']}  f1 {candidate['f1']:.3f}  p95 {candidate['p95_latency_ms']:.1f} ms")

    if champion is None:
        print("No champion yet: the first model inside the latency budget is promoted.")
        promote(candidate)
        return 0
    print(f"champion:       {champion['model']}  f1 {champion['f1']:.3f}  p95 {champion['p95_latency_ms']:.1f} ms")
    if candidate["f1"] >= champion["f1"] + MIN_GAIN:
        promote(candidate)
        return 0
    print(f"BLOCKED: F1 {candidate['f1']:.3f} is not at least {MIN_GAIN} better than the champion's {champion['f1']:.3f}.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
