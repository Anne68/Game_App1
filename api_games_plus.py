# api_games_plus.py
from __future__ import annotations

import os
import time
import math
import logging
import inspect
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple

import pymysql
from fastapi import FastAPI, HTTPException, Depends, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# FIX: Import correct des modules
from settings import get_settings
from model_manager import get_model
from monitoring_metrics import (
    prediction_latency,
    model_prediction_counter,
    get_monitor,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("games-api-ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# ---------------------------------------------------------------------
# Settings & Security
# ---------------------------------------------------------------------
settings = get_settings()

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

pwd_ctx = CryptContext(
    schemes=["bcrypt", "pbkdf2_sha256", "sha256_crypt"],
    deprecated="auto",
)

# ---------------------------------------------------------------------
# Flags DB
# ---------------------------------------------------------------------
DB_READY: bool = False
DB_LAST_ERROR: Optional[str] = None

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="Games API ML (C9→C13)",
    version="2.3.1",
    description="API avec modèle ML, monitoring avancé et MySQL pour les utilisateurs",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in settings.ALLOW_ORIGINS.split(",") if o] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Models (Pydantic)
# ---------------------------------------------------------------------
class TrainRequest(BaseModel):
    version: Optional[str] = None
    force_retrain: bool = False

class PredictionRequest(BaseModel):
    query: str
    k: int = 10
    min_confidence: float = 0.1

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

# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
def _ssl_kwargs() -> dict:
    if settings.DB_SSL_CA:
        return {"ssl": {"ca": settings.DB_SSL_CA}}
    return {"ssl": {}}

def get_db_conn():
    if not settings.db_configured:
        raise RuntimeError("Database not configured (missing DB_* env vars)")
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
        read_timeout=10,
        write_timeout=10,
        **_ssl_kwargs(),
    )

# ---------- USERS TABLE ----------
def ensure_users_table():
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'users';")
        if not cur.fetchone():
            cur.execute(
                """
                CREATE TABLE users (
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  username VARCHAR(190) UNIQUE NOT NULL,
                  hashed_password VARCHAR(255) NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            conn.commit()
            logger.info("Created users table with 'hashed_password'.")
            return

        cur.execute("SHOW COLUMNS FROM users LIKE 'hashed_password';")
        has_hashed = cur.fetchone() is not None
        if not has_hashed:
            cur.execute("SHOW COLUMNS FROM users LIKE 'password_hash';")
            has_legacy_hash = cur.fetchone() is not None
            if has_legacy_hash:
                cur.execute("ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255) NULL;")
                cur.execute("""
                    UPDATE users
                    SET hashed_password = password_hash
                    WHERE (hashed_password IS NULL OR hashed_password = '')
                      AND password_hash IS NOT NULL AND password_hash <> '';
                """)
                cur.execute("ALTER TABLE users MODIFY COLUMN hashed_password VARCHAR(255) NOT NULL;")
                conn.commit()
                logger.info("Added 'hashed_password' and migrated from 'password_hash'.")
            else:
                cur.execute("ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255) NOT NULL;")
                conn.commit()
                logger.info("Added 'hashed_password' column.")

def users_has_column(column: str) -> bool:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM users LIKE %s;", (column,))
        return cur.fetchone() is not None

def get_user(username: str) -> Optional[dict]:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        return cur.fetchone()

def create_user(username: str, password: str) -> None:
    hpwd = pwd_ctx.hash(password)
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO users(username, hashed_password) VALUES(%s,%s)", (username, hpwd))
        if users_has_column("password_hash"):
            cur.execute("UPDATE users SET password_hash=%s WHERE username=%s", (hpwd, username))
        conn.commit()

def extract_stored_password(row: dict) -> Tuple[Optional[str], str]:
    for col in ("hashed_password", "password_hash", "password"):
        if col in row:
            v = (row[col] or "").strip()
            if v:
                return col, v
    return None, ""

def update_user_password(username: str, new_hash: str) -> None:
    set_parts = ["hashed_password=%s"]
    params = [new_hash]
    if users_has_column("password_hash"):
        set_parts.append("password_hash=%s")
        params.append(new_hash)
    params.append(username)
    sql = f"UPDATE users SET {', '.join(set_parts)} WHERE username=%s"
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        conn.commit()

# ---------- GAMES ----------
def count_games_in_db() -> int:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM games WHERE COALESCE(title,'') <> ''")
        row = cur.fetchone()
        return int(row["n"]) if row else 0

def _parse_platforms(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]

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

def fetch_games_for_ml(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Renvoie une liste de jeux nettoyés pour le modèle."""
    sql = """
        SELECT
            game_id_rawg AS id,
            title,
            genres,
            rating,
            metacritic,
            platforms
        FROM games
        WHERE COALESCE(title,'') <> ''
    """
    if limit and limit > 0:
        sql += " LIMIT %s"

    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (limit,) if limit else None)
        rows = cur.fetchall() or []

    games: List[Dict[str, Any]] = []
    for r in rows:
        games.append({
            "id": int(r["id"]),
            "title": r.get("title") or "",
            "genres": r.get("genres") or "",
            "rating": _safe_float(r.get("rating"), 0.0),
            "metacritic": _safe_int(r.get("metacritic"), 0),
            "platforms": _parse_platforms(r.get("platforms")),
        })
    if not games:
        raise RuntimeError("Aucun jeu exploitable trouvé dans la table 'games'.")
    return games

def find_games_by_title(q: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Recherche case-insensitive par titre."""
    like = f"%{q.strip().lower()}%"
    sql = """
        SELECT
            g.game_id_rawg AS id,
            g.title,
            g.rating,
            g.metacritic,
            g.platforms,
            bp.best_price_PC AS best_price,
            bp.best_shop_PC  AS best_shop,
            bp.site_url_PC   AS site_url
        FROM games g
        LEFT JOIN best_price_pc bp
          ON bp.game_id_rawg = g.game_id_rawg OR bp.title = g.title
        WHERE LOWER(g.title) LIKE %s
        ORDER BY CHAR_LENGTH(g.title) ASC
        LIMIT %s
    """
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (like, max(1, limit)))
        return cur.fetchall() or []

# ---------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Decorator (latency)
# ---------------------------------------------------------------------
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
            async_wrapper.__name__ = func.__name__
            async_wrapper.__signature__ = inspect.signature(func)
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
            sync_wrapper.__name__ = func.__name__
            sync_wrapper.__signature__ = inspect.signature(func)
            return sync_wrapper
    return decorator

# ---------------------------------------------------------------------
# System & monitoring
# ---------------------------------------------------------------------
@app.get("/healthz", tags=["system"])
def healthz():
    model = get_model()
    monitor = get_monitor()
    return {
        "status": "healthy" if (DB_READY or not settings.DB_REQUIRED) else "degraded",
        "time": datetime.utcnow().isoformat(),
        "db_ready": DB_READY,
        "db_error": DB_LAST_ERROR,
        "model_loaded": model.is_trained,
        "model_version": model.model_version,
        "monitoring": monitor.get_metrics_summary(),
    }

@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "version": app.version, "status": "ok"}

Instrumentator().instrument(app).expose(app, include_in_schema=True)

# ---------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------
def _ensure_model_trained_with_db(force: bool = False):
    model = get_model()
    if (not model.is_trained) or force:
        games = fetch_games_for_ml()
        try:
            model.train(games)
        except ValueError as e:
            logger.warning("Training error (%s). Retrying with sanitized numeric fields.", e)
            for g in games:
                g["rating"] = _safe_float(g.get("rating"), 0.0)
                g["metacritic"] = _safe_int(g.get("metacritic"), 0)
            model.train(games)
        try:
            model.save_model()
        except Exception as e:
            logger.warning("Could not save model: %s", e)

# ---------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------
@app.post("/model/train", tags=["model"], response_model=TrainResponse)
def train_model(request: TrainRequest, background_tasks: BackgroundTasks, user: str = Depends(verify_token)):
    start_time = time.time()
    version = request.version or f"api-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    model = get_model()
    model.model_version = version

    games = fetch_games_for_ml()
    result = model.train(games)
    duration = time.time() - start_time

    monitor = get_monitor()
    monitor.record_training(
        model_version=version,
        games_count=len(games),
        feature_dim=model.game_features.shape[1],
        duration=duration,
        metrics=result,
    )
    background_tasks.add_task(model.save_model)
    return TrainResponse(status="success", version=version, duration=duration, result=result)

@app.get("/model/metrics", tags=["model"], response_model=ModelMetricsResponse)
def get_model_metrics(user: str = Depends(verify_token)):
    model = get_model()
    metrics = model.get_metrics()
    return ModelMetricsResponse(
        model_version=metrics.get("model_version", "unknown"),
        is_trained=metrics.get("is_trained", False),
        total_predictions=metrics.get("total_predictions", 0),
        avg_confidence=metrics.get("avg_confidence", 0.0),
        last_training=metrics.get("last_training"),
        games_count=metrics.get("games_count", 0),
        feature_dimension=metrics.get("feature_dim", 0),
    )

@app.post("/model/evaluate", tags=["model"])
def evaluate_model(test_queries: List[str] = Query(default=["RPG", "Action", "Indie", "Simulation"]),
                   user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    monitor = get_monitor()
    return monitor.evaluate_model(model, test_queries)

# ---------------------------------------------------------------------
# Recommendations (ML)
# ---------------------------------------------------------------------
@app.post("/recommend/ml", tags=["recommend"])
@measure_latency("recommend_ml")
def recommend_ml(request: PredictionRequest, user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    start_time = time.time()
    try:
        recommendations = model.predict(
            query=request.query.strip(),
            k=request.k,
            min_confidence=request.min_confidence
        )
    except ValueError as e:
        logger.warning("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=f"Bad input for model: {e}")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    latency = time.time() - start_time
    get_monitor().record_prediction("recommend_ml", request.query, recommendations, latency)
    return {
        "query": request.query,
        "recommendations": recommendations,
        "latency_ms": latency * 1000,
        "model_version": model.model_version,
    }

@app.get("/recommend/by-title/{title}", tags=["recommend"])
def recommend_by_title(title: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    matches = find_games_by_title(title, limit=1)
    if not matches:
        raise HTTPException(status_code=404, detail="Game not found")
    _ensure_model_trained_with_db()
    model = get_model()
    try:
        recos = model.predict_by_game_id(int(matches[0]["id"]), k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"title": title, "source_id": int(matches[0]["id"]), "recommendations": recos}

@app.get("/recommend/by-genre/{genre}", tags=["recommend"])
def recommend_by_genre(genre: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    recos = model.predict(query=genre, k=k, min_confidence=0.0)
    return {"genre": genre, "recommendations": recos}

# ---------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------
def _issue_token_core(username: str, password: str):
    if not DB_READY:
        if settings.DEMO_LOGIN_ENABLED and username == settings.DEMO_USERNAME and password == settings.DEMO_PASSWORD:
            access_token = create_access_token({"sub": username})
            return {"access_token": access_token, "token_type": "bearer", "mode": "demo"}
        raise HTTPException(status_code=503, detail="Database unavailable: cannot authenticate")

    try:
        u = get_user(username)
    except Exception as e:
        logger.error("DB error on login: %s", e)
        raise HTTPException(status_code=503, detail="Database error during login")

    if not u:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    col, stored = extract_stored_password(u)
    if not stored:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    try:
        if not pwd_ctx.verify(password, stored):
            raise HTTPException(status_code=401, detail="Incorrect username or password")
    except UnknownHashError:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    if col != "hashed_password" and users_has_column("hashed_password"):
        try:
            update_user_password(username, pwd_ctx.hash(password))
        except Exception as e:
            logger.warning("Cannot upgrade user hash for %s: %s", username, e)

    access_token = create_access_token({"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", tags=["auth"])
def token(username: str = Form(...), password: str = Form(...)):
    return _issue_token_core(username, password)

@app.post("/auth/token", tags=["auth"])
def token_alias(username: str = Form(...), password: str = Form(...)):
    return _issue_token_core(username, password)

@app.post("/register", tags=["auth"])
def register(username: str = Form(...), password: str = Form(...)):
    if not DB_READY:
        raise HTTPException(status_code=503, detail="Database unavailable: cannot register now")
    username = username.strip()
    if get_user(username):
        raise HTTPException(status_code=400, detail="User already exists")
    try:
        create_user(username, password)
    except Exception as e:
        logger.error("DB error on register: %s", e)
        raise HTTPException(status_code=503, detail="Database error during register")
    return {"ok": True}

# ---------------------------------------------------------------------
# Games search
# ---------------------------------------------------------------------
@app.get("/games/by-title/{title}", tags=["games"])
def games_by_title(title: str, user: str = Depends(verify_token)):
    rows = find_games_by_title(title, limit=25)
    results = []
    for r in rows:
        raw_plats = r.get("platforms") or ""
        platforms = [p.strip() for p in raw_plats.split(",") if p.strip()]
        results.append({
            "id": int(r["id"]),
            "title": r["title"],
            "rating": r.get("rating"),
            "metacritic": r.get("metacritic"),
            "best_price": r.get("best_price"),
            "best_shop": r.get("best_shop"),
            "site_url": r.get("site_url"),
            "platforms": platforms,
        })
    return {"results": results}

# ---------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------
@app.get("/monitoring/summary", tags=["monitoring"])
def monitoring_summary(user: str = Depends(verify_token)):
    monitor = get_monitor()
    model = get_model()
    return {"model": model.get_metrics(), "monitoring": monitor.get_metrics_summary(), "last_evaluation": monitor.last_evaluation}

@app.get("/monitoring/drift", tags=["monitoring"])
def check_drift(user: str = Depends(verify_token)):
    monitor = get_monitor()
    if len(monitor.confidence_history) < 100:
        return {"status": "insufficient_data", "message": "Need at least 100 predictions"}
    drift_score = monitor._detect_drift()
    status = "stable"
    if drift_score > 0.3:
        status = "high_drift"
    elif drift_score > 0.15:
        status = "moderate_drift"
    return {"drift_score": drift_score, "status": status,
            "recommendation": "Retrain model" if status == "high_drift"
            else "Monitor closely" if status == "moderate_drift" else "No action needed"}

# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global DB_READY, DB_LAST_ERROR

    logger.info("Starting Games API with ML...")

    try:
        if settings.db_configured:
            ensure_users_table()
            DB_READY = True
            DB_LAST_ERROR = None
            try:
                n = count_games_in_db()
                logger.info("Users table ready; games in DB: %s", n)
            except Exception:
                logger.info("Users table ready")
        else:
            DB_READY = False
            DB_LAST_ERROR = "Database not configured (set DB_* env vars)"
            logger.warning(DB_LAST_ERROR)
            if settings.DB_REQUIRED:
                raise RuntimeError(DB_LAST_ERROR)
    except Exception as e:
        DB_READY = False
        DB_LAST_ERROR = str(e)
        logger.error("Database initialization failed (will continue without DB): %s", e)
        if settings.DB_REQUIRED:
            raise

    model = get_model()
    model_path = os.path.join("model", "recommendation_model.pkl")
    try:
        if os.path.exists(model_path):
            if model.load_model():
                logger.info("Model %s loaded successfully", getattr(model, "model_version", "unknown"))
            else:
                logger.info("No saved model found; will train on first request")
        else:
            logger.info("No saved model found; will train on first request")
    except Exception as e:
        logger.warning("Error while loading model: %s. Will train on first request.", e)

    logger.info("API startup complete.")
