from __future__ import annotations
import os
import time
import math
import logging
import inspect
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple

import pymysql
from fastapi import FastAPI, HTTPException, Depends, Form, Query, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from settings import get_settings
from model_manager import get_model
from monitoring_metrics import (
    prediction_latency,
    model_prediction_counter,
    get_monitor,
)

logger = logging.getLogger("games-api-ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# ----------------- Compliance (optionnel) -----------------
try:
    from compliance.standards_compliance import SecurityValidator, AccessibilityValidator
    COMPLIANCE_AVAILABLE = True
    logger.info("Compliance module loaded successfully")
except ImportError as e:
    COMPLIANCE_AVAILABLE = False
    logger.warning(f"Compliance module not available: {e}")

settings = get_settings()

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
pwd_ctx = CryptContext(schemes=["bcrypt", "pbkdf2_sha256", "sha256_crypt"], deprecated="auto")

DB_READY: bool = False
DB_LAST_ERROR: Optional[str] = None

app = FastAPI(
    title="Games API ML (C9→C13)",
    version="2.6.0",
    description="API avec modèle ML, monitoring avancé et MySQL (AlwaysData) pour les utilisateurs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in settings.ALLOW_ORIGINS.split(",") if o] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Compliance helpers -----------------
def setup_compliance():
    if COMPLIANCE_AVAILABLE:
        try:
            app.state.security_validator = SecurityValidator()
            app.state.accessibility_validator = AccessibilityValidator()
            logger.info("Compliance validators attached to app state")
            return True
        except Exception as e:
            logger.error(f"Failed to setup compliance validators: {e}")
            return False
    logger.info("Compliance not available - continuing without")
    return False

COMPLIANCE_ENABLED = setup_compliance()

def get_security_validator():
    if COMPLIANCE_ENABLED and hasattr(app.state, 'security_validator'):
        return app.state.security_validator
    return None

def get_accessibility_validator():
    if COMPLIANCE_ENABLED and hasattr(app.state, 'accessibility_validator'):
        return app.state.accessibility_validator
    return None

# ----------------- Models -----------------
class TrainRequest(BaseModel):
    version: Optional[str] = None
    force_retrain: bool = False

class PredictionRequest(BaseModel):
    query: str
    k: int = 10
    min_confidence: float = 0.1
    platforms: Optional[List[str]] = None
    min_price: Optional[float] = None
    genres: Optional[List[str]] = None  # ✅ nouveau

class SimilarGameRequest(BaseModel):
    game_id: Optional[int] = None
    title: Optional[str] = None
    k: int = 10

class ModelMetricsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_version: str
    is_trained: bool
    total_predictions: int
    avg_confidence: float
    last_training: Optional[str]
    games_count: int
    feature_dimension: int

class TrainResponse(BaseModel):
    status: str
    version: Optional[str] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

# ----------------- Utils -----------------
def get_setting_bool(name: str, default: bool = False) -> bool:
    try:
        v = getattr(settings, name, None)
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        return default

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default

def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return int(x)
    except Exception:
        return default

# ----------------- Auth helpers -----------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub: str = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token")
        return sub
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ----------------- Decorator (latency) -----------------
def measure_latency(endpoint: str):
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    res = await func(*args, **kwargs)
                    prediction_latency.labels(endpoint=endpoint).observe(time.time() - start)
                    return res
                except Exception:
                    prediction_latency.labels(endpoint=endpoint).observe(time.time() - start)
                    model_prediction_counter.labels(endpoint=endpoint, status="error").inc()
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    res = func(*args, **kwargs)
                    prediction_latency.labels(endpoint=endpoint).observe(time.time() - start)
                    return res
                except Exception:
                    prediction_latency.labels(endpoint=endpoint).observe(time.time() - start)
                    model_prediction_counter.labels(endpoint=endpoint, status="error").inc()
                    raise
            return sync_wrapper
    return decorator

# ----------------- System -----------------
@app.get("/healthz", tags=["system"])
def healthz():
    model = get_model()
    return {
        "status": "healthy" if (DB_READY or not settings.DB_REQUIRED) else "degraded",
        "time": datetime.utcnow().isoformat(),
        "model_loaded": model.is_trained,
        "model_version": model.model_version,
    }

# ----------------- Training -----------------
@app.post("/model/train", tags=["model"], response_model=TrainResponse)
def train_model(request: TrainRequest, background_tasks: BackgroundTasks, user: str = Depends(verify_token)):
    start_time = time.time()
    version = request.version or f"api-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    model = get_model()
    model.model_version = version
    games = []  # ⚠️ à remplir depuis ta DB
    result = model.train(games)
    duration = time.time() - start_time
    background_tasks.add_task(model.save_model)
    return TrainResponse(status="success", version=version, duration=duration, result=result)

# ----------------- Recos ML -----------------
@app.post("/recommend/ml", tags=["recommend"])
def recommend_ml(req: PredictionRequest, user: dict = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()

    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    try:
        recommendations = model.predict(
            query=clean_query,
            k=req.k,
            min_confidence=req.min_confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # --------------------------
    # ✅ Appliquer les filtres
    # --------------------------
    if req.min_price is not None:
        recommendations = [
            r for r in recommendations
            if r.get("best_price_PC") is not None and r["best_price_PC"] >= req.min_price
        ]

    if req.platforms:
        recommendations = [
            r for r in recommendations
            if any(p in r.get("platforms", []) for p in req.platforms)
        ]

    if req.genres:
        recommendations = [
            r for r in recommendations
            if any(g.lower() in (r.get("genres", "").lower()) for g in req.genres)
        ]

    return {
        "query": clean_query,
        "recommendations": recommendations,
        "count": len(recommendations),
        "model_version": model.model_version
    }


# ----------------- Autres endpoints reco -----------------
@app.post("/recommend/similar-game", tags=["recommend"])
def recommend_similar_game(payload: SimilarGameRequest, user: str = Depends(verify_token)):
    model = get_model()
    return {"recommendations": model.predict_by_game_id(payload.game_id, k=payload.k)}

@app.get("/recommend/by-genre/{genre}", tags=["recommend"])
def recommend_by_genre(genre: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    model = get_model()
    return {"genre": genre, "recommendations": model.recommend_by_genre(genre, k=k)}

@app.get("/recommend/by-title/{title}", tags=["recommend"])
def recommend_by_title(title: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    model = get_model()
    return {"title": title, "recommendations": model.recommend_by_title_similarity(title, k=k)}

# ----------------- Auth -----------------
@app.post("/token", tags=["auth"])
def token(username: str = Form(...), password: str = Form(...)):
    access_token = create_access_token({"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

Instrumentator().instrument(app).expose(app, include_in_schema=True)
