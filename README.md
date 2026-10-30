# AI381 Lab 2 — Customer Churn API

[![CI](https://github.com/rahafhamad610/ai381-lab2/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rahafhamad610/ai381-lab2/actions/workflows/ci.yml)

A customer churn prediction service using the Lab 1 starter dataset,
a scikit-learn pipeline, FastAPI, Docker, and GitHub Actions.

## Start the published service

With Docker running and port 8000 available, run:

```bash
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/rahafhamad610/ai381-lab2@sha256:c0eb0433123f9c81dd602005634b29a6f14fa4fa8468f6d0fe99d52c186fc2d5
```

Keep the terminal open. Visit:
- API documentation: http://localhost:8000/docs
- Health endpoint: http://localhost:8000/health

Press Ctrl+C to stop the service.

## Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months":5,"monthly_charges":89.9,"contract":"month-to-month","internet_service":"fiber","support_calls":3}'
```

Expected response:

```json
{"churn_probability":0.866}
```

Inputs are validated. Missing fields, unexpected fields, invalid categories,
and invalid numeric values are rejected.

## Model

The pipeline uses StandardScaler for numeric features, OneHotEncoder for
categorical features, and LogisticRegression. It is trained on churn_v1.csv
with a stratified 80/20 split and random_state=42.

Observed test-set metrics:
- F1: 0.512
- ROC AUC: 0.783

The complete fitted pipeline is saved during the Docker build and loaded
when the API starts.

## Build and test

From the repository root:

```bash
docker build --target builder -t lab2-tests .
docker run --rm lab2-tests python -m pytest -v
docker run --rm lab2-tests python -m ruff check src tests
```

The 13 test cases cover health and prediction contracts, model loading,
invalid inputs, missing and extra fields, zero values, and repeatable predictions.

## Container and CI

The multi-stage Dockerfile pins the Python base image by digest, runs tests
during the build, and runs the API as a non-root user with a health check.
Package installers are removed from the final runtime image.

GitHub Actions runs linting, tests, image building, and Trivy scanning on
pull requests. On main, a successful scan permits publication to GHCR with
the commit SHA as the image tag. Third-party actions are pinned by full SHA.

The scan blocks HIGH and CRITICAL vulnerabilities with available fixes.
Unfixed vulnerabilities are excluded by the configured ignore-unfixed option.

Main is protected with pull requests and the required Lint and test check.

## Published image

```text
ghcr.io/rahafhamad610/ai381-lab2@sha256:c0eb0433123f9c81dd602005634b29a6f14fa4fa8468f6d0fe99d52c186fc2d5
```
