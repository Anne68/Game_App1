# api_games_plus.py
from __future__ import annotations

import os
import time
import logging
import inspect
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import pymysql
from fastapi import FastAPI, HTTPException, Depends, Form, Query, BackgroundTasks
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

# OAuth2 for Swagger
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Accepte plusieurs formats historiques et « upgrade » automatiquement
pwd_ctx = CryptContext(
    schemes=["bcrypt", "pbkdf2_sha256", "sha256_crypt"],
    deprecated="auto",
)

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="Games API ML (C9→C13)",
    version="2.1.0",
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
    # supprime l’avertissement pydantic “model_” comme espace réservé
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
# Demo data
# ---------------------------------------------------------------------
GAMES: List[Dict[str, Any]] = [
    {"id": 1, "title": "The Witcher 3: Wild Hunt", "genres": "RPG, Action", "rating": 4.9, "metacritic": 93, "platforms": ["PC", "PS4", "Xbox One"]},
    {"id": 2, "title": "Hades", "genres": "Action, Roguelike", "rating": 4.8, "metacritic": 93, "platforms": ["PC", "Switch", "PS4"]},
    {"id": 3, "title": "Stardew Valley", "genres": "Simulation, RPG", "rating": 4.7, "metacritic": 89, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 4, "title": "Celeste", "genres": "Platformer, Indie", "rating": 4.6, "metacritic": 94, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 5, "title": "Doom Eternal", "genres": "Action, FPS", "rating": 4.5, "metacritic": 88, "platforms": ["PC", "PS4", "Xbox One"]},
    {"id": 6, "title": "Hollow Knight", "genres": "Metroidvania, Indie", "rating": 4.8, "metacritic": 90, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 7, "title": "Disco Elysium", "genres": "RPG, Narrative", "rating": 4.7, "metacritic": 91, "platforms": ["PC", "PS4", "Xbox One"]},
]

# ---------------------------------------------------------------------
# DB helpers (MySQL users)
# ---------------------------------------------------------------------
def get_db_conn():
    ssl = {"ssl": {"ca": settings.DB_SSL_CA}} if settings.DB_SSL_CA else {}
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        **ssl,
    )

def ensure_users_table():
    """Crée la table si absente et migre 'password' -> 'password_hash' si besoin."""
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'users';")
        if not cur.fetchone():
            cur.execute(
                """
                CREATE TABLE users (
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  username VARCHAR(190) UNIQUE NOT NULL,
                  password_hash VARCHAR(255) NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            conn.commit()
            logger.info("Created users table (password_hash).")
            return

        # table présente → vérifier colonnes
        cur.execute("SHOW COLUMNS FROM users LIKE 'password_hash';")
        has_hash = cur.fetchone() is not None
        if not has_hash:
            cur.execute("SHOW COLUMNS FROM users LIKE 'password';")
            has_legacy = cur.fetchone() is not None
            if has_legacy:
                try:
                    cur.execute("ALTER TABLE users CHANGE COLUMN password password_hash VARCHAR(255) NOT NULL;")
                    conn.commit()
                    logger.info("Migrated users.password -> users.password_hash")
                except Exception as e:
                    logger.warning("Could not migrate users table, will use legacy 'password' column. %s", e)
            else:
                cur.execute("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255) NOT NULL;")
                conn.commit()
                logger.info("Added users.password_hash column.")

def users_has_column(column: str) -> bool:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM users LIKE %s;", (column,))
        return cur.fetchone() is not None

def users_password_column() -> str:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM users LIKE 'hashed_password';")
        if cur.fetchone():
            return "hashed_password"
        cur.execute("SHOW COLUMNS FROM users LIKE 'password_hash';")
        if cur.fetchone():
            return "password_hash"
        cur.execute("SHOW COLUMNS FROM users LIKE 'password';")
        if cur.fetchone():
            return "password"
    return "password_hash"

def get_user(username: str) -> Optional[dict]:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        return cur.fetchone()

def update_user_password(username: str, new_hash: str) -> None:
    """
    Écrit le hash moderne dans 'hashed_password'. Si 'password_hash' existe, on la met aussi.
    """
    set_parts = []
    params: list[Any] = []

    if users_has_column("hashed_password"):
        set_parts.append("hashed_password=%s")
        params.append(new_hash)
    if users_has_column("password_hash"):
        set_parts.append("password_hash=%s")
        params.append(new_hash)

    if not set_parts:
        # dernier recours : créer la colonne password_hash et écrire
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255) NOT NULL;")
            conn.commit()
        set_parts.append("password_hash=%s")
        params.append(new_hash)

    params.append(username)
    sql = f"UPDATE users SET {', '.join(set_parts)} WHERE username=%s"
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        conn.commit()

def migrate_legacy_passwords() -> int:
    """Copie password_hash -> hashed_password si besoin."""
    if not (users_has_column("hashed_password") and users_has_column("password_hash")):
        return 0
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE users
            SET hashed_password = password_hash
            WHERE (hashed_password IS NULL OR hashed_password = '')
              AND password_hash IS NOT NULL AND password_hash <> '';
            """
        )
        affected = cur.rowcount or 0
        conn.commit()
        return affected

def looks_like_hash(s: str) -> bool:
    return isinstance(s, str) and s.startswith("$")

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

# =======================
# AUTH — routes /token + /auth/token
# =======================
def _issue_token_core(username: str, password: str):
    u = get_user(username)
    if not u:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    stored = (u.get("hashed_password") or u.get("password_hash") or u.get("password") or "").strip()
    if not stored:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    ok = False
    if looks_like_hash(stored):
        try:
            ok = pwd_ctx.verify(password, stored)
        except UnknownHashError:
            ok = False
    else:
        # ancien stockage en clair
        ok = (stored == password)

    if not ok:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # Upgrade en bcrypt si nécessaire
    try:
        scheme = pwd_ctx.identify(stored) if looks_like_hash(stored) else None
    except Exception:
        scheme = None
    if scheme != "bcrypt":  # None, pbkdf2, sha256_crypt, plain…
        update_user_password(username, pwd_ctx.hash(password))

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
    username = username.strip()
    if get_user(username):
        raise HTTPException(status_code=400, detail="User already exists")

    hpwd = pwd_ctx.hash(password)
    col = users_password_column()
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(f"INSERT INTO users (username, {col}) VALUES (%s, %s)", (username, hpwd))
        # si une 2e colonne existe, synchronise
        if col != "hashed_password" and users_has_column("hashed_password"):
            cur.execute("UPDATE users SET hashed_password=%s WHERE username=%s", (hpwd, username))
        if col != "password_hash" and users_has_column("password_hash"):
            cur.execute("UPDATE users SET password_hash=%s WHERE username=%s", (hpwd, username))
        conn.commit()
    return {"ok": True}

# ---------------------------------------------------------------------
# Decorator (fix: preserve signature + support async)
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
        "status": "healthy",
        "time": datetime.utcnow().isoformat(),
        "model_loaded": model.is_trained,
        "model_version": model.model_version,
        "monitoring": monitor.get_metrics_summary(),
    }

@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "version": app.version, "status": "ok"}

# Prometheus
Instrumentator().instrument(app).expose(app, include_in_schema=True)

# ---------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------
@app.post("/model/train", tags=["model"], response_model=TrainResponse)
def train_model(request: TrainRequest, background_tasks: BackgroundTasks, user: str = Depends(verify_token)):
    model = get_model()
    monitor = get_monitor()

    if model.is_trained and not request.force_retrain:
        return TrainResponse(status="already_trained", version=model.model_version)

    start_time = time.time()
    version = request.version or f"api-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    model.model_version = version

    try:
        result = model.train(GAMES)
        duration = time.time() - start_time

        monitor.record_training(
            model_version=version,
            games_count=len(GAMES),
            feature_dim=model.game_features.shape[1],
            duration=duration,
            metrics=result,
        )

        background_tasks.add_task(model.save_model)

        return TrainResponse(status="success", version=version, duration=duration, result=result)
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))

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
    model = get_model()
    monitor = get_monitor()
    if not model.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    return monitor.evaluate_model(model, test_queries)

# ---------------------------------------------------------------------
# Recommendations (ML)
# ---------------------------------------------------------------------
@app.post("/recommend/ml", tags=["recommend"])
@measure_latency("recommend_ml")
def recommend_ml(request: PredictionRequest, user: str = Depends(verify_token)):
    model = get_model()
    monitor = get_monitor()

    if not model.is_trained:
        logger.info("Model not trained, auto-training...")
        model.train(GAMES)
        model.save_model()

    start_time = time.time()
    recommendations = model.predict(query=request.query, k=request.k, min_confidence=request.min_confidence)
    latency = time.time() - start_time

    monitor.record_prediction("recommend_ml", request.query, recommendations, latency)
    return {"query": request.query, "recommendations": recommendations, "latency_ms": latency * 1000, "model_version": model.model_version}

@app.get("/recommend/by-game/{game_id}", tags=["recommend"])
@measure_latency("recommend_by_game")
def recommend_by_game(game_id: int, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    model = get_model()
    monitor = get_monitor()
    if not model.is_trained:
        model.train(GAMES)

    start_time = time.time()
    try:
        recommendations = model.predict_by_game_id(game_id, k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    latency = time.time() - start_time
    monitor.record_prediction("recommend_by_game", f"game_id:{game_id}", recommendations, latency)
    return {"game_id": game_id, "recommendations": recommendations, "latency_ms": latency * 1000}

# --- Endpoints pour Streamlit UI ---
@app.get("/recommend/by-title/{title}", tags=["recommend"])
def recommend_by_title(title: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    matches = [g for g in GAMES if title.lower() in g["title"].lower()]
    if not matches:
        raise HTTPException(status_code=404, detail="Game not found")
    return recommend_by_game(matches[0]["id"], k, user)

@app.get("/recommend/by-genre/{genre}", tags=["recommend"])
def recommend_by_genre(genre: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    model = get_model()
    if not model.is_trained:
        model.train(GAMES)
    recos = model.predict(query=genre, k=k, min_confidence=0.0)
    return {"genre": genre, "recommendations": recos}

# ---------------------------------------------------------------------
# Simple games search
# ---------------------------------------------------------------------
@app.get("/games/by-title/{title}", tags=["games"])
def games_by_title(title: str, user: str = Depends(verify_token)):
    q = title.strip().lower()
    results = [{
        "id": g["id"], "title": g["title"], "genres": g["genres"],
        "rating": g["rating"], "metacritic": g["metacritic"],
        "platforms": ", ".join(g["platforms"]),
    } for g in GAMES if q in g["title"].lower()]
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
    logger.info("Starting Games API with ML...")

    # === DB init ===
    try:
        ensure_users_table()
        logger.info("Users table ready")

        # (optionnel) migre password_hash -> hashed_password si la colonne legacy existe
        try:
            migrated = migrate_legacy_passwords()
            if migrated:
                logger.info("Password migration: %d row(s) updated", migrated)
            else:
                logger.info("Password migration: nothing to do")
        except Exception as e:
            logger.warning("Password migration skipped (error): %s", e)

    except Exception:
        logger.exception("Database initialization failed")
        raise

    # === ML model ===
    model = get_model()
    model_path = os.path.join("model", "recommendation_model.pkl")
    try:
        if os.path.exists(model_path):
            if model.load_model():
                logger.info("Model %s loaded successfully", getattr(model, "model_version", "unknown"))
            else:
                logger.warning("Saved model found at %s but load failed; will train on first request", model_path)
        else:
            logger.info("No saved model found at %s; will train on first request", model_path)
    except Exception as e:
        logger.warning("Error while loading model: %s. Will train on first request.", e)

    logger.info("API startup complete. %d games available.", len(GAMES))
