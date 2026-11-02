from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from app.model import service
from app.schemas import (
    BatchCreditApplication,
    BatchPredictionResponse,
    CreditApplication,
    HealthResponse,
    PredictionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="Credit Risk API",
    description="Inferência em tempo real de risco de crédito (Home Credit).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if service.is_loaded else "unavailable",
        model_loaded=service.is_loaded,
    )


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return service.metrics


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return service.predict(application.model_dump())


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchCreditApplication):
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return service.predict_batch([a.model_dump() for a in batch.applications])
