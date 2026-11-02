import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "credit_risk_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

DEFAULT_THRESHOLD = 0.5027
REJECT_THRESHOLD = 0.64

# Ordem exata de colunas esperada pelo pipeline treinado.
FEATURE_ORDER = [
    "EXT_SOURCE_3", "EXT_SOURCE_2", "AGE_YEARS", "YEARS_EMPLOYED",
    "BUREAU_ACTIVE_LOANS", "BUREAU_TOTAL_CREDIT_DEBT", "DTI_RATIO",
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "FLAG_OWN_CAR",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
]

CATEGORICAL_FEATURES = ("NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE")
BANDS = (
    "APROVADO_AUTOMATICO",
    "ANALISE_ADICIONAL_OU_CONDICIONAL",
    "REPROVADO_AUTOMATICO",
)


def classify(score: float, approve_below: float, reject_from: float) -> Dict[str, Any]:
    """Aplica a regra de 3 faixas (semáforo de risco)."""
    if score < approve_below:
        band = "APROVADO_AUTOMATICO"  # Verde
    elif score < reject_from:
        band = "ANALISE_ADICIONAL_OU_CONDICIONAL"  # Amarelo
    else:
        band = "REPROVADO_AUTOMATICO"  # Vermelho
    return {
        "probabilidade_default": score,
        "decisao_binaria": int(score >= approve_below),
        "faixa_risco": band,
        "threshold_utilizado": approve_below,
    }


class ModelService:
    """Mantém pipeline e metadados em memória."""

    def __init__(self) -> None:
        self.pipeline: Optional[Any] = None
        self.metrics: Dict[str, Any] = {}
        self.approve_below = DEFAULT_THRESHOLD
        self.reject_from = REJECT_THRESHOLD

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        self.pipeline = joblib.load(PIPELINE_PATH)
        with open(METRICS_PATH, encoding="utf-8") as f:
            self.metrics = json.load(f)
        bands = self.metrics.get("decision_thresholds", {}).get("traffic_light_bands", {})
        self.approve_below = float(bands.get("green_auto_approve_below", DEFAULT_THRESHOLD))
        self.reject_from = float(bands.get("red_auto_reject_above_or_equal", REJECT_THRESHOLD))

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado.")
        return self.predict_batch([payload])["predictions"][0]

    def predict_batch(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predição vetorizada: um único DataFrame e uma única chamada predict_proba."""
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado.")
        rows = [{k: p.get(k) for k in FEATURE_ORDER} for p in payloads]
        df = pd.DataFrame(rows, columns=FEATURE_ORDER)
        # None -> NaN para os imputers do pipeline
        for col in FEATURE_ORDER:
            if col not in CATEGORICAL_FEATURES:
                df[col] = df[col].astype(float)
        scores = self.pipeline.predict_proba(df)[:, 1]
        predictions = [
            classify(round(float(s), 6), self.approve_below, self.reject_from)
            for s in scores
        ]
        counts = {band: 0 for band in BANDS}
        for p in predictions:
            counts[p["faixa_risco"]] += 1
        return {
            "predictions": predictions,
            "summary": {
                "total_processado": len(predictions),
                "contagem_por_faixa": counts,
            },
        }


service = ModelService()
