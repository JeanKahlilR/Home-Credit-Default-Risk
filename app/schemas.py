from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

FaixaRisco = Literal[
    "APROVADO_AUTOMATICO",
    "ANALISE_ADICIONAL_OU_CONDICIONAL",
    "REPROVADO_AUTOMATICO",
]

EXAMPLE_APPLICATION: Dict[str, Any] = {
    "EXT_SOURCE_2": 0.55,
    "EXT_SOURCE_3": 0.50,
    "AMT_CREDIT": 450000.0,
    "YEARS_EMPLOYED": 6.5,
    "AGE_YEARS": 38.0,
    "DTI_RATIO": 0.28,
    "AMT_INCOME_TOTAL": 180000.0,
    "BUREAU_TOTAL_CREDIT_DEBT": 35000.0,
    "BUREAU_ACTIVE_LOANS": 2,
    "FLAG_OWN_CAR": 1,
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_INCOME_TYPE": "Working",
}


class CreditApplication(BaseModel):
    """12 variáveis do MVP (5 Cs do Crédito)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_APPLICATION},
    )

    EXT_SOURCE_2: float = Field(..., ge=0.0, le=1.0)
    # Nulo legítimo: mantém default None (imputado pelo pipeline).
    EXT_SOURCE_3: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    AMT_CREDIT: float = Field(..., gt=0)
    YEARS_EMPLOYED: Optional[float] = None
    AGE_YEARS: float = Field(..., ge=18, le=120)
    DTI_RATIO: Optional[float] = None
    AMT_INCOME_TOTAL: float = Field(..., gt=0)
    BUREAU_TOTAL_CREDIT_DEBT: float = Field(..., ge=0)
    BUREAU_ACTIVE_LOANS: int = Field(..., ge=0)
    FLAG_OWN_CAR: int = Field(..., ge=0, le=1)
    NAME_EDUCATION_TYPE: str
    NAME_INCOME_TYPE: str


class PredictionResponse(BaseModel):
    probabilidade_default: float
    decisao_binaria: int = Field(..., ge=0, le=1)
    faixa_risco: FaixaRisco
    threshold_utilizado: float


class BatchCreditApplication(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"applications": [EXAMPLE_APPLICATION]}}
    )

    applications: List[CreditApplication] = Field(..., min_length=1)


class BatchSummary(BaseModel):
    total_processado: int
    contagem_por_faixa: Dict[FaixaRisco, int]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: BatchSummary


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
