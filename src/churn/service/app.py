import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from churn import db
from churn.model_store import load_model

PREDICTIONS = Counter("churn_predictions_total", "Predictions by class", ["churn"])
SCORE = Histogram("churn_score", "Predicted churn probability", buckets=[i / 10 for i in range(11)])
MODEL_INFO = Gauge("churn_model_info", "Model loaded by this pod", ["version"])
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)  # штатные 0.1, 0.5, 1 с слишком грубые


class Features(BaseModel):
    model_config = {"extra": "forbid"}

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=120)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float | None = None


class Prediction(BaseModel):
    #model_config = {"protected_namespaces": ()}

    score: float
    churn: bool
    model_version: str
    request_id: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline, app.state.meta, app.state.version = load_model()
    MODEL_INFO.labels(app.state.version).set(1)

    db.init()
    yield
    app.state.pipeline = None


app = FastAPI(title="churn-service", version="1.0", lifespan=lifespan)
Instrumentator().instrument(app, latency_lowr_buckets=LATENCY_BUCKETS).expose(app)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": getattr(app.state, "version", "unknown")}

@app.get("/ready")
def ready():
    if getattr(app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "ready"}



@app.post("/v1/predict")
def predict(x: Features, bg: BackgroundTasks) -> Prediction:
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    payload = x.model_dump()  # x.dict() in pydantic v1
    frame = pd.DataFrame([payload]).reindex(columns=app.state.meta["features"])

    score = float(app.state.pipeline.predict_proba(frame)[0, 1])

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    bg.add_task(db.save_prediction, request_id, payload, score, app.state.version, latency_ms)

    churn = score >= app.state.meta["threshold"]
    PREDICTIONS.labels(str(churn).lower()).inc()
    SCORE.observe(score)

    return Prediction(score=score, churn=churn, model_version = app.state.version, request_id=request_id, latency_ms=latency_ms)








