import pytest
from fastapi.testclient import TestClient
from sklearn.utils.validation import check_is_fitted

from src.app.main import app

SAMPLE = {
    "tenure_months": 5,
    "monthly_charges": 89.9,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_contract(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert set(body) == {"churn_probability"}
    assert isinstance(body["churn_probability"], float)
    assert 0 <= body["churn_probability"] <= 1


def test_model_loads(client):
    model = app.state.model
    check_is_fitted(model.named_steps["clf"])
    assert model.classes_.tolist() == [0, 1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenure_months", -1),
        ("tenure_months", 1.5),
        ("monthly_charges", -10),
        ("contract", "invalid"),
        ("internet_service", "invalid"),
        ("support_calls", "three"),
    ],
)
def test_invalid_input_rejected(client, field, value):
    customer = {**SAMPLE, field: value}
    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_missing_field_rejected(client):
    customer = SAMPLE.copy()
    customer.pop("contract")
    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_extra_field_rejected(client):
    customer = {**SAMPLE, "unexpected": 123}
    response = client.post("/predict", json=customer)
    assert response.status_code == 422


def test_zero_values_edge_case(client):
    customer = {
        **SAMPLE,
        "tenure_months": 0,
        "monthly_charges": 0,
        "support_calls": 0,
    }
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1


def test_deterministic_prediction(client):
    first = client.post("/predict", json=SAMPLE)
    second = client.post("/predict", json=SAMPLE)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["churn_probability"] == pytest.approx(
        0.866, abs=0.001
    )
