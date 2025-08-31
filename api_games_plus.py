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
# Logging SETUP (MUST be early)
# ---------------------------------------------------------------------
logger = logging.getLogger("games-api-ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# Compliance import avec fallback gracieux (AVANT de l'utiliser)
try:
    from compliance.standards_compliance import SecurityValidator, AccessibilityValidator
    COMPLIANCE_AVAILABLE = True
    logger.info("Compliance module loaded successfully")
except ImportError as e:
    COMPLIANCE_AVAILABLE = False
    logger.warning(f"Compliance module not available: {e}")

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
# FastAPI app - CRÉER L'APP D'ABORD
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

# Setup compliance APRÈS création de l'app
def setup_compliance():
    """Setup compliance validators si disponible"""
    if COMPLIANCE_AVAILABLE:
        try:
            app.state.security_validator = SecurityValidator()
            app.state.accessibility_validator = AccessibilityValidator()
            logger.info("Compliance validators attached to app state")
            return True
        except Exception as e:
            logger.error(f"Failed to setup compliance validators: {e}")
            return False
    else:
        logger.info("Compliance not available - continuing without")
        return False

# Initialiser compliance maintenant que app existe
COMPLIANCE_ENABLED = setup_compliance()

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
# Helper functions pour compliance
# ---------------------------------------------------------------------
def get_security_validator():
    """Récupère le security validator de manière sûre"""
    if COMPLIANCE_ENABLED and hasattr(app.state, 'security_validator'):
        return app.state.security_validator
    return None

def get_accessibility_validator():
    """Récupère l'accessibility validator de manière sûre"""
    if COMPLIANCE_ENABLED and hasattr(app.state, 'accessibility_validator'):
        return app.state.accessibility_validator
    return None

# ---------------------------------------------------------------------
# System & monitoring
# ---------------------------------------------------------------------
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

    # Log compliance status
    if COMPLIANCE_ENABLED:
        logger.info("E4 Compliance standards active")
        logger.info(f"Security validator: {get_security_validator() is not None}")
        logger.info(f"Accessibility validator: {get_accessibility_validator() is not None}")
    else:
        logger.warning("E4 Compliance standards not available")

    logger.info("API startup complete.")
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
# Recommendations (ML) avec compliance
# ---------------------------------------------------------------------
@app.post("/recommend/ml", tags=["recommend"])
@measure_latency("recommend_ml")
def recommend_ml(request: PredictionRequest, user: str = Depends(verify_token)):
    _ensure_model_trained_with_db()
    model = get_model()
    
    # Nettoyage de base
    clean_query = request.query.strip()
    
    # Nettoyage avec compliance si disponible
    security_validator = get_security_validator()
    if security_validator:
        try:
            clean_query = security_validator.sanitize_input(clean_query)
        except Exception as e:
            logger.warning(f"Compliance sanitization failed: {e}")
    
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query is empty")

    start_time = time.time()
    try:
        recommendations = model.predict(
            query=clean_query,
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
    get_monitor().record_prediction("recommend_ml", clean_query, recommendations, latency)
    
    response = {
        "query": clean_query,
        "recommendations": recommendations,
        "latency_ms": latency * 1000,
        "model_version": model.model_version,
    }
    
    # Enhancement accessibilité si disponible
    accessibility_validator = get_accessibility_validator()
    if accessibility_validator:
        try:
            response = accessibility_validator.enhance_response_accessibility(response)
        except Exception as e:
            logger.warning(f"Accessibility enhancement failed: {e}")
    
    return response

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
# Auth endpoints avec compliance
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
    
    # Nettoyage de base
    username = username.strip()
    
    # Validation avec compliance si disponible
    security_validator = get_security_validator()
    if security_validator:
        try:
            username = security_validator.sanitize_input(username)
            
            password_validation = security_validator.validate_password(password)
            if not password_validation["valid"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Password does not meet security requirements",
                        "issues": password_validation["issues"],
                        "strength": password_validation["strength"]
                    }
                )
        except Exception as e:
            logger.warning(f"Compliance validation failed: {e}")
    
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
    # Nettoyage du titre avec compliance si disponible
    clean_title = title.strip()
    security_validator = get_security_validator()
    if security_validator:
        try:
            clean_title = security_validator.sanitize_input(clean_title)
        except Exception as e:
            logger.warning(f"Title sanitization failed: {e}")
    
    rows = find_games_by_title(clean_title, limit=25)
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
    return {
        "model": model.get_metrics(), 
        "monitoring": monitor.get_metrics_summary(), 
        "last_evaluation": monitor.last_evaluation,
        "compliance_status": {
            "enabled": COMPLIANCE_ENABLED,
            "security_validator": get_security_validator() is not None,
            "accessibility_validator": get_accessibility_validator() is not None
        }
    }

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
    return {
        "drift_score": drift_score, 
        "status": status,
        "recommendation": "Retrain model" if status == "high_drift"
        else "Monitor closely" if status == "moderate_drift" else "No action needed"
    }

# ---------------------------------------------------------------------
# Compliance endpoints (nouveaux pour E4)
# ---------------------------------------------------------------------
@app.get("/compliance/status", tags=["compliance"])
def compliance_status(user: str = Depends(verify_token)):
    """Status de la conformité E4"""
    if not COMPLIANCE_ENABLED:
        return {
            "compliance_enabled": False,
            "message": "Compliance module not available",
            "standards": []
        }
    
    return {
        "compliance_enabled": True,
        "message": "E4 compliance standards active",
        "standards": [
            {
                "name": "Security Standards",
                "status": "active" if get_security_validator() else "inactive",
                "features": ["Input sanitization", "Password validation", "Injection protection"]
            },
            {
                "name": "Accessibility Standards", 
                "status": "active" if get_accessibility_validator() else "inactive",
                "features": ["ARIA labels", "Screen reader support", "WCAG 2.1 AA compliance"]
            }
        ]
    }

@app.post("/compliance/validate-password", tags=["compliance"])
def validate_password_endpoint(password: str = Form(...), user: str = Depends(verify_token)):
    """Validation de mot de passe selon standards E4"""
    security_validator = get_security_validator()
    if not security_validator:
        return {"error": "Compliance module not available"}
    
    try:
        result = security_validator.validate_password(password)
        return {
            "validation_result": result,
            "standards": "E4 Security Requirements"
        }
    except Exception as e:
        logger.error(f"Password validation failed: {e}")
        return {"error": "Validation failed"}

# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global DB_READY, DB_LAST_ERROR

    logger.info("Starting Games API with ML and E4 compliance...")

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

    model = get_model
