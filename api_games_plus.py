# api_games_plus.p
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

# Compliance (optionnel)
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
    version="2.5.0",
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

# ----------------- DB helpers -----------------
def _ssl_kwargs() -> dict:
    ssl_dict: Dict[str, str] = {}
    if getattr(settings, "DB_SSL_CA", None):
        ssl_dict["ca"] = settings.DB_SSL_CA
    if getattr(settings, "DB_SSL_CERT", None) and getattr(settings, "DB_SSL_KEY", None):
        ssl_dict["cert"] = settings.DB_SSL_CERT
        ssl_dict["key"] = settings.DB_SSL_KEY
    return {"ssl": ssl_dict} if ssl_dict else {}

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
            logger.info("Created users table.")
            return
        cur.execute("SHOW COLUMNS FROM users LIKE 'hashed_password';")
        if not cur.fetchone():
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

def create_user(username: str, password: str) -> int:
    hpwd = pwd_ctx.hash(password)
    with get_db_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO users(username, hashed_password) VALUES(%s,%s)",
                (username, hpwd),
            )
            user_id = int(cur.lastrowid)
            if users_has_column("password_hash"):
                cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (hpwd, user_id))
            conn.commit()
            logger.info("User '%s' created (id=%s)", username, user_id)
            return user_id
        except pymysql.err.IntegrityError:
            cur.execute("SELECT id FROM users WHERE username=%s", (username,))
            row = cur.fetchone()
            return int(row["id"]) if row and "id" in row else 0

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

def count_games_in_db() -> int:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM games WHERE COALESCE(title,'') <> ''")
        row = cur.fetchone()
        return int(row["n"]) if row else 0

def _parse_platforms(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [p.strip() for p in str(value).split(",") if p.strip()]

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

# ---------- IMPORTANT: lire TOUJOURS la BDD (pas de fallback, sauf USE_DEMO_GAMES=true) ----------
def fetch_games_for_ml(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    use_demo = get_setting_bool("USE_DEMO_GAMES", False)
    if not DB_READY and not use_demo:
        raise RuntimeError("Database not ready and USE_DEMO_GAMES is false")

    if not DB_READY and use_demo:
        logger.warning("DB not ready. Falling back to demo games because USE_DEMO_GAMES=true")
        return [
            {"id": 1, "title": "The Witcher 3", "genres": "RPG Action", "rating": 4.9, "metacritic": 93, "platforms": ["PC", "PS4"]},
            {"id": 2, "title": "Hades", "genres": "Action Roguelike", "rating": 4.8, "metacritic": 93, "platforms": ["PC", "Switch"]},
            {"id": 3, "title": "Stardew Valley", "genres": "Simulation RPG", "rating": 4.7, "metacritic": 89, "platforms": ["PC", "Switch"]},
            {"id": 4, "title": "Celeste", "genres": "Platformer Indie", "rating": 4.6, "metacritic": 94, "platforms": ["PC", "Switch"]},
            {"id": 5, "title": "Doom Eternal", "genres": "Action FPS", "rating": 4.5, "metacritic": 88, "platforms": ["PC", "PS4"]},
            {"id": 6, "title": "Hollow Knight", "genres": "Metroidvania Indie", "rating": 4.7, "metacritic": 90, "platforms": ["PC", "Switch"]},
            {"id": 7, "title": "Cyberpunk 2077", "genres": "RPG Action", "rating": 4.0, "metacritic": 76, "platforms": ["PC", "PS4", "Xbox"]},
        ]

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

    logger.info("Loaded %s games from DB", len(games))
    if not games:
        raise RuntimeError("Aucun jeu exploitable trouvé dans la table 'games'.")
    return games

def find_games_by_title(q: str, limit: int = 25) -> List[Dict[str, Any]]:
    use_demo = get_setting_bool("USE_DEMO_GAMES", False)
    if not DB_READY and not use_demo:
        raise RuntimeError("Database not ready and USE_DEMO_GAMES is false")
    if not DB_READY and use_demo:
        # recherche simple dans la démo
        default_games = fetch_games_for_ml()
        q_lower = q.strip().lower()
        results = []
        for game in default_games:
            if q_lower in game["title"].lower():
                results.append({
                    "id": game["id"],
                    "title": game["title"],
                    "rating": game["rating"],
                    "metacritic": game["metacritic"],
                    "platforms": ",".join(game["platforms"]) if isinstance(game["platforms"], list) else game["platforms"]
                })
        return results[:limit]

    like = f"%{q.strip().lower()}%"
    sql = """
        SELECT
            g.game_id_rawg AS id,
            g.title,
            g.rating,
            g.metacritic,
            g.platforms
        FROM games g
        WHERE LOWER(g.title) LIKE %s
        ORDER BY CHAR_LENGTH(g.title) ASC
        LIMIT %s
    """
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (like, max(1, limit)))
        return cur.fetchall() or []

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

# ----------------- System & monitoring -----------------
@app.get("/healthz", tags=["system"])
def healthz():
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

    if COMPLIANCE_ENABLED:
        logger.info("E4 Compliance standards active")
        logger.info(f"Security validator: {get_security_validator() is not None}")
        logger.info(f"Accessibility validator: {get_accessibility_validator() is not None}")
    else:
        logger.warning("E4 Compliance standards not available")

    monitor = get_monitor()
    return {
        "status": "healthy" if (DB_READY or not settings.DB_REQUIRED) else "degraded",
        "time": datetime.utcnow().isoformat(),
        "db_ready": DB_READY,
        "db_error": DB_LAST_ERROR,
        "model_loaded": model.is_trained,
        "model_version": model.model_version,
        "monitoring": monitor.get_metrics_summary(),
        "compliance_enabled": COMPLIANCE_ENABLED,
    }

@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "version": app.version, "status": "ok"}

# Health probe Render HEAD /
@app.head("/", include_in_schema=False)
def head_root():
    return Response(status_code=200)

Instrumentator().instrument(app).expose(app, include_in_schema=True)

# ----------------- Training & metrics -----------------
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

# ----------------- Recos (ML) -----------------
@app.post("/recommend/ml", tags=["recommend"])
@measure_latency("recommend_ml")
def recommend_ml(request: PredictionRequest, user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    clean_query = request.query.strip()
    sec = get_security_validator()
    if sec:
        try:
            clean_query = sec.sanitize_input(clean_query)
        except Exception as e:
            logger.warning(f"Compliance sanitization failed: {e}")
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query is empty")
    start_time = time.time()
    try:
        recommendations = model.predict(query=clean_query, k=request.k, min_confidence=request.min_confidence)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad input for model: {e}")
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    latency = time.time() - start_time
    get_monitor().record_prediction("recommend_ml", clean_query, recommendations, latency)
    response = {
        "query": clean_query,
        "recommendations": recommendations,
        "latency_ms": latency * 1000,
        "model_version": model.model_version,
    }
    acc = get_accessibility_validator()
    if acc:
        try:
            response = acc.enhance_response_accessibility(response)
        except Exception as e:
            logger.warning(f"Accessibility enhancement failed: {e}")
    return response

@app.post("/recommend/similar-game", tags=["recommend"])
def recommend_similar_game(payload: SimilarGameRequest, user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    game_id = payload.game_id
    if not game_id and payload.title:
        matches = find_games_by_title(payload.title, limit=1)
        if not matches:
            raise HTTPException(status_code=404, detail="Game not found with given title")
        game_id = int(matches[0]["id"])
    if not game_id:
        raise HTTPException(status_code=400, detail="Provide either 'game_id' or 'title'")
    try:
        recs = model.predict_by_game_id(int(game_id), k=max(1, payload.k))
        return {"source_id": int(game_id), "recommendations": recs, "model_version": model.model_version}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/recommend/cluster/{cluster_id}", tags=["recommend"])
def recommend_cluster(cluster_id: int, sample: int = Query(50, ge=1, le=500), user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    try:
        games = model.get_cluster_games(cluster_id, sample=sample)
        return {"cluster": cluster_id, "count": len(games), "games": games}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recommend/cluster-explore", tags=["recommend"])
def recommend_cluster_explore(top_terms: int = Query(10, ge=3, le=30), user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    return model.cluster_explore(top_n_terms=top_terms)

@app.get("/recommend/random-cluster", tags=["recommend"])
def recommend_random_cluster(sample: int = Query(12, ge=1, le=200), user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    return model.random_cluster(sample=sample)

@app.get("/recommend/by-title/{title}", tags=["recommend"])
def recommend_by_title(title: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    try:
        recos = model.recommend_by_title_similarity(title, k=k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"title": title, "recommendations": recos}

@app.get("/recommend/by-genre/{genre}", tags=["recommend"])
def recommend_by_genre(genre: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    recos = model.recommend_by_genre(genre, k=k)
    return {"genre": genre, "recommendations": recos}

# ----------------- Auth (auto-create au login) -----------------
def _issue_token_core(username: str, password: str):
    if not DB_READY:
        if settings.DEMO_LOGIN_ENABLED and username == settings.DEMO_USERNAME and password == settings.DEMO_PASSWORD:
            access_token = create_access_token({"sub": username})
            return {"access_token": access_token, "token_type": "bearer", "mode": "demo"}
        raise HTTPException(status_code=503, detail="Database unavailable: cannot authenticate")

    auto_create = get_setting_bool("AUTO_CREATE_USER_ON_LOGIN", True)

    try:
        u = get_user(username)
    except Exception as e:
        logger.error("DB error on login lookup: %s", e)
        raise HTTPException(status_code=503, detail="Database error during login")

    if not u and auto_create:
        logger.info("Auto-creating user '%s' on first login", username)
        try:
            sec = get_security_validator()
            if sec:
                pv = sec.validate_password(password)
                if not pv.get("valid", True):
                    raise HTTPException(status_code=400, detail={
                        "message": "Password does not meet security requirements",
                        "issues": pv.get("issues", []),
                        "strength": pv.get("strength"),
                    })
                username = sec.sanitize_input(username)
            create_user(username, password)
            u = get_user(username)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Auto-create user failed: %s", e)
            raise HTTPException(status_code=503, detail="Database error creating user")

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
    try:
        ensure_users_table()
    except Exception as e:
        logger.error("DB error ensuring users table: %s", e)
        raise HTTPException(status_code=503, detail="Database error preparing table")
    username = username.strip()
    sec = get_security_validator()
    if sec:
        try:
            username = sec.sanitize_input(username)
            pv = sec.validate_password(password)
            if not pv["valid"]:
                raise HTTPException(status_code=400, detail={
                    "message": "Password does not meet security requirements",
                    "issues": pv["issues"],
                    "strength": pv["strength"]
                })
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Compliance validation failed: {e}")
    if get_user(username):
        raise HTTPException(status_code=400, detail="User already exists")
    try:
        user_id = create_user(username, password)
    except Exception as e:
        logger.error("DB error on register: %s", e)
        raise HTTPException(status_code=503, detail="Database error during register")
    return {"ok": True, "user_id": user_id}

# ----------------- Startup -----------------
@app.on_event("startup")
async def startup_event():
    global DB_READY, DB_LAST_ERROR
    logger.info("Starting Games API with ML and E4 compliance...")

    try:
        if settings.db_configured and settings.DB_REQUIRED:
            ensure_users_table()
            DB_READY = True
            DB_LAST_ERROR = None
            logger.info("Database connected successfully (required mode)")
        elif settings.db_configured:
            try:
                ensure_users_table()
                DB_READY = True
                DB_LAST_ERROR = None
                logger.info("Database connected successfully (optional mode)")
                try:
                    n = count_games_in_db()
                    logger.info("Games in DB: %s", n)
                except Exception:
                    logger.info("Users table ready")
            except Exception as e:
                DB_READY = False
                DB_LAST_ERROR = str(e)
                logger.warning(f"Database connection failed (continuing without DB): {e}")
        else:
            DB_READY = False
            DB_LAST_ERROR = "Database not configured (set DB_* env vars)"
            logger.info("Database not configured - using demo mode")
    except Exception as e:
        DB_READY = False
        DB_LAST_ERROR = str(e)
        logger.error(f"Database initialization failed: {e}")
        if settings.DB_REQUIRED:
            logger.error("DB_REQUIRED=true but database failed - stopping startup")
            raise

    # Charger modèle s'il existe
    model = get_model()
    model_path = os.path.join("model", "recommendation_model.pkl")
    try:
        if os.path.exists(model_path):
            if model.load_model():
                logger.info("Model loaded from disk: %s", model.model_version)
            else:
                logger.info("Model file exists but loading failed - will train on first request")
        else:
            logger.info("No saved model found - will train on first request")
    except Exception as e:
        logger.warning("Error loading model: %s - will train on first request", e)

    if COMPLIANCE_ENABLED:
        logger.info("E4 Compliance standards active")
