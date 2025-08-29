# api_games_plus.py
from __future__ import annotations

import os
import time
import logging
import inspect
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Set

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

# Pour savoir si le modèle est entraîné sur la DB (et pas sur un fallback)
MODEL_ORIGIN: str = "unknown"

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="Games API ML (DB-backed)",
    version="3.0.0",
    description="API avec modèle ML, monitoring avancé et MySQL AlwaysData pour les utilisateurs et les jeux",
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
# DB helpers (users + games)
# ---------------------------------------------------------------------
def _ssl_kwargs() -> dict:
    if settings.DB_SSL_CA:
        return {"ssl": {"ca": settings.DB_SSL_CA}}
    return {"ssl": {}}

def get_db_conn():
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=6,
        read_timeout=12,
        write_timeout=12,
        **_ssl_kwargs(),
    )

def _table_exists(cur, name: str) -> bool:
    cur.execute("SHOW TABLES LIKE %s;", (name,))
    return cur.fetchone() is not None

def _columns(cur, table: str) -> Set[str]:
    cur.execute(f"SHOW COLUMNS FROM {table};")
    return {row["Field"] for row in cur.fetchall()}

# ---------- USERS ----------
def ensure_users_table():
    """Crée/upgrade la table users. Gère migration password -> password_hash/hashed_password."""
    with get_db_conn() as conn, conn.cursor() as cur:
        if not _table_exists(cur, "users"):
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

        cols = _columns(cur, "users")
        if "hashed_password" not in cols:
            # tenter migration depuis password_hash ou password
            if "password_hash" in cols:
                cur.execute("ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255) NULL;")
                cur.execute(
                    """
                    UPDATE users
                    SET hashed_password = password_hash
                    WHERE (hashed_password IS NULL OR hashed_password = '')
                      AND password_hash IS NOT NULL AND password_hash <> '';
                    """
                )
                cur.execute("ALTER TABLE users MODIFY COLUMN hashed_password VARCHAR(255) NOT NULL;")
                conn.commit()
                logger.info("Added 'hashed_password' and migrated from 'password_hash'.")
            elif "password" in cols:
                cur.execute("ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255) NULL;")
                cur.execute(
                    """
                    UPDATE users
                    SET hashed_password = password
                    WHERE (hashed_password IS NULL OR hashed_password = '')
                      AND password IS NOT NULL AND password <> '';
                    """
                )
                cur.execute("ALTER TABLE users MODIFY COLUMN hashed_password VARCHAR(255) NOT NULL;")
                conn.commit()
                logger.info("Added 'hashed_password' and migrated from legacy 'password'.")
            else:
                cur.execute("ALTER TABLE users ADD COLUMN hashed_password VARCHAR(255) NOT NULL;")
                conn.commit()
                logger.info("Added 'hashed_password' column.")

def users_has_column(column: str) -> bool:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW COLUMNS FROM users LIKE %s;", (column,))
        return cur.fetchone() is not None

def users_password_column() -> str:
    with get_db_conn() as conn, conn.cursor() as cur:
        for col in ("hashed_password", "password_hash", "password"):
            cur.execute("SHOW COLUMNS FROM users LIKE %s;", (col,))
            if cur.fetchone():
                return col
    return "hashed_password"

def get_user(username: str) -> Optional[dict]:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        return cur.fetchone()

def create_user(username: str, password: str) -> None:
    hpwd = pwd_ctx.hash(password)
    col = users_password_column()
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(f"INSERT INTO users(username, {col}) VALUES(%s,%s)", (username, hpwd))
        if col != "hashed_password" and users_has_column("hashed_password"):
            cur.execute("UPDATE users SET hashed_password=%s WHERE username=%s", (hpwd, username))
        if col != "password_hash" and users_has_column("password_hash"):
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

# ---------- GAMES (AlwaysData) ----------
def _fetch_platforms_map(cur) -> Dict[int, List[str]]:
    """Retourne {game_id: [platform names]} si les tables existent, sinon {}."""
    if not (_table_exists(cur, "platforms") and _table_exists(cur, "game_platforms")):
        return {}
    cur.execute(
        """
        SELECT gp.game_id, p.name AS platform
        FROM game_platforms gp
        JOIN platforms p ON p.id = gp.platform_id
        ORDER BY gp.game_id
        """
    )
    by_game: Dict[int, List[str]] = {}
    for row in cur.fetchall():
        by_game.setdefault(row["game_id"], []).append(row["platform"])
    return by_game

def fetch_games_for_ml(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Récupère les jeux depuis la DB et les met au format attendu par le modèle :
    {id, title, genres, rating, metacritic, platforms: [..]}
    """
    with get_db_conn() as conn, conn.cursor() as cur:
        if not _table_exists(cur, "games"):
            raise RuntimeError("Table 'games' absente dans la base.")

        cols = _columns(cur, "games")
        # Colonnes "douces" : toutes optionnelles sauf id/title
        must_have = {"id"}
        if "title" not in cols and "name" in cols:
            title_col = "name"
        else:
            title_col = "title"
            must_have.add("title")

        if not must_have.issubset(cols):
            raise RuntimeError("La table 'games' doit au minimum contenir 'id' et 'title' (ou 'name').")

        select_cols = [f"`{title_col}` AS title", "id"]
        # champs facultatifs
        if "genres" in cols:
            select_cols.append("genres")
        elif "genre" in cols:
            select_cols.append("genre AS genres")
        else:
            select_cols.append("'' AS genres")

        if "rating" in cols:
            select_cols.append("rating")
        else:
            select_cols.append("0.0 AS rating")

        if "metacritic" in cols:
            select_cols.append("metacritic")
        elif "mc" in cols:
            select_cols.append("mc AS metacritic")
        else:
            select_cols.append("0 AS metacritic")

        sql = f"SELECT {', '.join(select_cols)} FROM games ORDER BY id"
        if limit:
            sql += " LIMIT %s"
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)

        rows = cur.fetchall()
        platforms_map = _fetch_platforms_map(cur)

    games = []
    for r in rows:
        games.append(
            {
                "id": int(r["id"]),
                "title": r["title"],
                "genres": r.get("genres") or "",
                "rating": float(r.get("rating", 0.0) or 0.0),
                "metacritic": int(r.get("metacritic", 0) or 0),
                "platforms": platforms_map.get(int(r["id"]), []),
            }
        )
    return games

def find_games_by_title(q: str, limit: int = 20) -> List[Dict[str, Any]]:
    with get_db_conn() as conn, conn.cursor() as cur:
        if not _table_exists(cur, "games"):
            return []
        cols = _columns(cur, "games")
        title_col = "name" if ("name" in cols and "title" not in cols) else "title"
        cur.execute(
            f"""
            SELECT id, {title_col} AS title
            FROM games
            WHERE LOWER({title_col}) LIKE %s
            ORDER BY {title_col} ASC
            LIMIT %s
            """,
            (f"%{q.lower()}%", limit),
        )
        rows = cur.fetchall()

        # enrichir avec plateformes si possible
        platforms_map = _fetch_platforms_map(cur)
    out = []
    for r in rows:
        out.append(
            {
                "id": int(r["id"]),
                "title": r["title"],
                "platforms": ", ".join(_ for _ in platforms_map.get(int(r["id"]), [])),
            }
        )
    return out

def game_exists(game_id: int) -> bool:
    with get_db_conn() as conn, conn.cursor() as cur:
        if not _table_exists(cur, "games"):
            return False
        cur.execute("SELECT 1 FROM games WHERE id=%s LIMIT 1", (game_id,))
        return cur.fetchone() is not None

def count_games_in_db() -> int:
    with get_db_conn() as conn, conn.cursor() as cur:
        if not _table_exists(cur, "games"):
            return 0
        cur.execute("SELECT COUNT(*) AS c FROM games")
        return int(cur.fetchone()["c"])

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
        "model_origin": getattr(model, "origin", MODEL_ORIGIN),
        "model_version": model.model_version,
        "monitoring": monitor.get_metrics_summary(),
    }

@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "version": app.version, "status": "ok"}

Instrumentator().instrument(app).expose(app, include_in_schema=True)

# ---------------------------------------------------------------------
# Model management (sur base AlwaysData)
# ---------------------------------------------------------------------
def _ensure_model_trained_with_db(force: bool = False):
    global MODEL_ORIGIN
    model = get_model()
    if (not model.is_trained) or getattr(model, "origin", None) != "db" or force:
        games = fetch_games_for_ml()
        if not games:
            raise HTTPException(status_code=500, detail="Aucun jeu disponible en base pour entraîner le modèle")
        model.model_version = f"db-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        model.train(games)
        model.save_model()
        model.origin = "db"
        MODEL_ORIGIN = "db"
        logger.info("Model trained on DB with %d games", len(games))

@app.post("/model/train", tags=["model"], response_model=TrainResponse)
def train_model(request: TrainRequest, background_tasks: BackgroundTasks, user: str = Depends(verify_token)):
    start_time = time.time()
    _ensure_model_trained_with_db(force=request.force_retrain)
    model = get_model()
    duration = time.time() - start_time
    # on sauvegarde en tâche de fond (s'il y a eu entraînement)
    background_tasks.add_task(model.save_model)
    return TrainResponse(status="success", version=model.model_version, duration=duration, result={"origin": "db"})

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
# Recommendations (ML) – basées DB
# ---------------------------------------------------------------------
@app.post("/recommend/ml", tags=["recommend"])
@measure_latency("recommend_ml")
def recommend_ml(request: PredictionRequest, user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    monitor = get_monitor()
    start_time = time.time()
    recommendations = model.predict(query=request.query, k=request.k, min_confidence=request.min_confidence)
    latency = time.time() - start_time
    monitor.record_prediction("recommend_ml", request.query, recommendations, latency)
    return {
        "query": request.query,
        "recommendations": recommendations,
        "latency_ms": latency * 1000,
        "model_version": model.model_version,
    }

@app.get("/recommend/by-game/{game_id}", tags=["recommend"])
@measure_latency("recommend_by_game")
def recommend_by_game(game_id: int, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    if not game_exists(game_id):
        raise HTTPException(status_code=404, detail="Game not found")
    _ensure_model_trained_with_db()
    model = get_model()
    monitor = get_monitor()
    start_time = time.time()
    try:
        recommendations = model.predict_by_game_id(game_id, k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    latency = time.time() - start_time
    monitor.record_prediction("recommend_by_game", f"game_id:{game_id}", recommendations, latency)
    return {"game_id": game_id, "recommendations": recommendations, "latency_ms": latency * 1000}

@app.get("/recommend/by-title/{title}", tags=["recommend"])
def recommend_by_title(title: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    # cherche en DB
    matches = find_games_by_title(title, limit=1)
    if matches:
        return recommend_by_game(matches[0]["id"], k, user)
    # sinon : fallback ML basé sur texte
    _ensure_model_trained_with_db()
    model = get_model()
    recos = model.predict(query=title, k=k, min_confidence=0.0)
    return {
        "detail": "Game not found in DB; returning ML-based recommendations for your query.",
        "query": title,
        "recommendations": recos,
        "model_version": model.model_version,
    }

@app.get("/recommend/by-genre/{genre}", tags=["recommend"])
def recommend_by_genre(genre: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    recos = model.predict(query=genre, k=k, min_confidence=0.0)
    return {"genre": genre, "recommendations": recos}

# ---------------------------------------------------------------------
# Auth endpoints (MySQL users)
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
    try:
        if get_user(username):
            raise HTTPException(status_code=400, detail="User already exists")
        create_user(username, password)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("DB error on register: %s", e)
        raise HTTPException(status_code=503, detail="Database error during register")
    return {"ok": True}

# ---------------------------------------------------------------------
# Simple games search (DB)
# ---------------------------------------------------------------------
@app.get("/games/by-title/{title}", tags=["games"])
def games_by_title(title: str, user: str = Depends(verify_token)):
    results = find_games_by_title(title, limit=30)
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
    global DB_READY, DB_LAST_ERROR, MODEL_ORIGIN

    logger.info("Starting Games API with ML...")

    # === DB init ===
    try:
        ensure_users_table()
        DB_READY = True
        DB_LAST_ERROR = None
        gcount = count_games_in_db()
        logger.info("Users table ready; games in DB: %d", gcount)
    except Exception as e:
        DB_READY = False
        DB_LAST_ERROR = str(e)
        logger.error("Database initialization failed (will continue without DB): %s", e)
        if settings.DB_REQUIRED:
            raise

    # === ML model ===
    model = get_model()
    model_path = os.path.join("model", "recommendation_model.pkl")
    try:
        if os.path.exists(model_path) and model.load_model():
            MODEL_ORIGIN = getattr(model, "origin", MODEL_ORIGIN)
            logger.info("Model %s loaded successfully (origin=%s)", getattr(model, "model_version", "unknown"), MODEL_ORIGIN)
        else:
            logger.info("No saved model found; will train on first request")
    except Exception as e:
        logger.warning("Error while loading model: %s. Will train on first request.", e)

    logger.info("API startup complete.")
