#!/usr/bin/env bash
# Smoke test for the running service (Azure public URL or your laptop). Exit code 0 = the service works.
#   bash scripts/smoke_test.sh http://localhost:8000
#   bash scripts/smoke_test.sh https://<your-app>.<region>.azurecontainerapps.io
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
BASE_URL="${BASE_URL%/}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Waiting for $BASE_URL/health (up to 60 seconds; a Container App that scaled to zero needs a moment)..."
for i in $(seq 1 30); do
  if curl -fsS "$BASE_URL/health" -o /tmp/health.json 2>/dev/null; then break; fi
  sleep 2
  if [ "$i" -eq 30 ]; then echo "FAIL: /health did not answer"; exit 1; fi
done
echo "health:  $(cat /tmp/health.json)"

PRED="$(curl -fsS -X POST "$BASE_URL/predict" -H 'Content-Type: application/json' -d @"$HERE/sample_request.json")"
echo "predict: $PRED"

# the prediction must be a probability between 0 and 1, and the service must say which model it is
python3 - "$PRED" "$(cat /tmp/health.json)" <<'PY'
import json, sys
pred, health = json.loads(sys.argv[1]), json.loads(sys.argv[2])
p = pred["churn_probability"]
assert 0.0 <= p <= 1.0, f"probability out of range: {p}"
assert health.get("status") == "ok", "health is not ok"
assert health.get("model_version"), "health does not say which model is running"
print("PASS: smoke test ok, model_version =", health["model_version"])
PY
