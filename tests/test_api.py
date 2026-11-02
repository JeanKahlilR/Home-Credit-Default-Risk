import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import classify
from app.schemas import CreditApplication

LOW_RISK = {
    "EXT_SOURCE_2": 0.85,
    "EXT_SOURCE_3": 0.85,
    "AMT_CREDIT": 150000.0,
    "YEARS_EMPLOYED": 15.0,
    "AGE_YEARS": 48.0,
    "DTI_RATIO": 0.1,
    "AMT_INCOME_TOTAL": 300000.0,
    "BUREAU_TOTAL_CREDIT_DEBT": 0.0,
    "BUREAU_ACTIVE_LOANS": 0,
    "FLAG_OWN_CAR": 1,
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_INCOME_TYPE": "State servant",
}

HIGH_RISK = {
    "EXT_SOURCE_2": 0.05,
    "EXT_SOURCE_3": 0.05,
    "AMT_CREDIT": 1200000.0,
    "YEARS_EMPLOYED": 0.2,
    "AGE_YEARS": 21.0,
    "DTI_RATIO": 0.9,
    "AMT_INCOME_TOTAL": 40000.0,
    "BUREAU_TOTAL_CREDIT_DEBT": 500000.0,
    "BUREAU_ACTIVE_LOANS": 8,
    "FLAG_OWN_CAR": 0,
    "NAME_EDUCATION_TYPE": "Lower secondary",
    "NAME_INCOME_TYPE": "Working",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_metadata(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    assert r.json()["decision_thresholds"]["optimal_threshold"] == 0.5027


def test_low_risk_profile(client):
    r = client.post("/predict", json=LOW_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["probabilidade_default"] < 0.5027
    assert body["decisao_binaria"] == 0
    assert body["faixa_risco"] == "APROVADO_AUTOMATICO"
    assert body["threshold_utilizado"] == 0.5027


def test_high_risk_profile(client):
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["probabilidade_default"] >= 0.64
    assert body["decisao_binaria"] == 1
    assert body["faixa_risco"] == "REPROVADO_AUTOMATICO"


def test_null_tolerance(client):
    payload = {**LOW_RISK, "EXT_SOURCE_3": None, "YEARS_EMPLOYED": None, "DTI_RATIO": None}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probabilidade_default"] <= 1.0

    payload = {k: v for k, v in LOW_RISK.items() if k != "EXT_SOURCE_3"}
    assert client.post("/predict", json=payload).status_code == 200


def test_invalid_types_422(client):
    r = client.post("/predict", json={**LOW_RISK, "AMT_CREDIT": "abc"})
    assert r.status_code == 422
    r = client.post("/predict", json={**LOW_RISK, "BUREAU_ACTIVE_LOANS": "muitos"})
    assert r.status_code == 422


@pytest.mark.parametrize("field", ["EXT_SOURCE_2", "EXT_SOURCE_3"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_ext_source_range_422(client, field, value):
    r = client.post("/predict", json={**LOW_RISK, field: value})
    assert r.status_code == 422


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"applications": [LOW_RISK, HIGH_RISK, LOW_RISK]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    assert body["summary"]["total_processado"] == 3
    counts = body["summary"]["contagem_por_faixa"]
    assert sum(counts.values()) == 3
    assert counts["APROVADO_AUTOMATICO"] == 2
    assert counts["REPROVADO_AUTOMATICO"] == 1
    assert counts["ANALISE_ADICIONAL_OU_CONDICIONAL"] == 0
    # consistência com o endpoint unitário
    single = client.post("/predict", json=HIGH_RISK).json()
    assert body["predictions"][1] == single


def test_batch_empty_422(client):
    assert client.post("/predict/batch", json={"applications": []}).status_code == 422


def test_batch_invalid_item_422(client):
    bad = {**LOW_RISK, "EXT_SOURCE_2": 5}
    assert client.post("/predict/batch", json={"applications": [LOW_RISK, bad]}).status_code == 422


def test_swagger_example_is_valid(client):
    example = CreditApplication.model_json_schema()["example"]
    assert client.post("/predict", json=example).status_code == 200


def test_missing_required_422(client):
    payload = {k: v for k, v in LOW_RISK.items() if k != "EXT_SOURCE_2"}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "score,band,binary",
    [
        (0.30, "APROVADO_AUTOMATICO", 0),
        (0.5026, "APROVADO_AUTOMATICO", 0),
        (0.5027, "ANALISE_ADICIONAL_OU_CONDICIONAL", 1),
        (0.6399, "ANALISE_ADICIONAL_OU_CONDICIONAL", 1),
        (0.64, "REPROVADO_AUTOMATICO", 1),
        (0.95, "REPROVADO_AUTOMATICO", 1),
    ],
)
def test_traffic_light_boundaries(score, band, binary):
    out = classify(score, 0.5027, 0.64)
    assert out["faixa_risco"] == band
    assert out["decisao_binaria"] == binary
