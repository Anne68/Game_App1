# api_games_plus.py — API FastAPI avec ML + Auth MySQL + Monitoring
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# ML + monitoring
from model_manager import get_model, RecommendationModel
from monitoring_metrics import get_monitor, measure_latency

# Auth via MySQL
import pymysql
from settings import get_settings

logger = logging.getLogger("games-api-ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# ========= Config via settings.py =========
settings = get_settings()
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(
    title="Games API ML (C9→C13)",
    version="2.0.0",
    description="API avec modèle ML, monitoring avancé et pipeline MLOps",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOW_ORIGINS.split(",") if o.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Pydantic Models =========
class TrainRequest(BaseModel):
    version: Optional[str] = None
    force_retrain: bool = False

class PredictionRequest(BaseModel):
    query: str
    k: int = 10
    min_confidence: float = 0.1

class ModelMetricsResponse(BaseModel):
    # Evite le warning Pydantic: "model_" namespace protégé
    model_config = {"protected_namespaces": ()}
    model_version: str
    is_trained: bool
    total_predictions: int
    avg_confidence: float
    last_training: Optional[str]
    games_count: int
    feature_dimension: int

# ========= Demo Data =========
GAMES: List[Dict] = [
    {"id": 1, "title": "The Witcher 3: Wild Hunt", "genres": "RPG, Action", "rating": 4.9, "metacritic": 93, "platforms": ["PC", "PS4", "Xbox One"]},
    {"id": 2, "title": "Hades", "genres": "Action, Roguelike", "rating": 4.8, "metacritic": 93, "platforms": ["PC", "Switch", "PS4"]},
    {"id": 3, "title": "Stardew Valley", "genres": "Simulation, RPG", "rating": 4.7, "metacritic": 89, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 4, "title": "Celeste", "genres": "Platformer, Indie", "rating": 4.6, "metacritic": 94, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 5, "title": "Doom Eternal", "genres": "Action, FPS", "rating": 4.5, "metacritic": 88, "platforms": ["PC", "PS4", "Xbox One"]},
    {"id": 6, "title": "Hollow Knight", "genres": "Metroidvania, Indie", "rating": 4.8, "metacritic": 90, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 7, "title": "Disco Elysium", "genres": "RPG, Narrative", "rating": 4.7, "metacritic": 91, "platforms": ["PC", "PS4", "Xbox One"]},
]

# ========= MySQL helpers =========
def _get_db_conn():
    """Connexion PyMySQL basée sur settings.py (SSL optionnel)."""
    ssl_params = {"ca": settings.DB_SSL_CA} if getattr(settings, "DB_SSL_CA", None) else None
    kwargs = dict(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )
    if ssl_params:
        kwargs["ssl"] = ssl_params
    return pymysql.connect(**kwargs)

def _init_users_table():
    ddl = (
        "CREATE TABLE IF NOT EXISTS users ("
        " id INT AUTO_INCREMENT PRIMARY KEY,"
        " username VARCHAR(255) UNIQUE NOT NULL,"
        " hashed_password VARCHAR(255) NOT NULL,"
        " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
    )
    with _get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)

def _get_user(username: str) -> Optional[Dict]:
    with _get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, hashed_password FROM users WHERE username=%s", (username,))
            return cur.fetchone()

def _create_user(username: str, hashed_password: str) -> None:
    with _get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (username, hashed_password) VALUES (%s, %s)", (username, hashed_password))

# ========= Utils auth =========
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
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

# ========= System & Monitoring =========
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

# Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, include_in_schema=True)

# ========= ML Model Management =========
@app.post("/model/train", tags=["model"], response_model=Dict)
def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(verify_token),
):
    """Entraîne ou ré-entraîne le modèle de recommandation"""
    model = get_model()
    monitor = get_monitor()

    if model.is_trained and not request.force_retrain:
        return {"status": "already_trained", "version": model.model_version}

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

        return {"status": "success", "version": version, "duration": duration, "result": result}
    except Exception as e:
        logger.error(f"Training failed: {e}")
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
def evaluate_model(
    test_queries: List[str] = Query(default=["RPG", "Action", "Indie", "Simulation"]),
    user: str = Depends(verify_token),
):
    model = get_model()
    monitor = get_monitor()
    if not model.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    return monitor.evaluate_model(model, test_queries)

# ========= Recommendations =========
@app.post("/recommend/ml", tags=["recommend"])
@measure_latency("recommend_ml")
def recommend_ml(
    request: PredictionRequest,
    user: str = Depends(verify_token),
):
    model = get_model()
    monitor = get_monitor()
    if not model.is_trained:
        logger.info("Model not trained, auto-training...")
        model.train(GAMES)
        model.save_model()

    t0 = time.time()
    try:
        recommendations = model.predict(query=request.query, k=request.k, min_confidence=request.min_confidence)
        latency = time.time() - t0
        monitor.record_prediction("recommend_ml", request.query, recommendations, latency)
        return {
            "query": request.query,
            "recommendations": recommendations,
            "latency_ms": latency * 1000,
            "model_version": model.model_version,
        }
    except Exception as e:
        monitor.record_error("recommend_ml", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/by-game/{game_id}", tags=["recommend"])
@measure_latency("recommend_by_game")
def recommend_by_game(
    game_id: int,
    k: int = Query(10, ge=1, le=50),
    user: str = Depends(verify_token),
):
    model = get_model()
    monitor = get_monitor()
    if not model.is_trained:
        model.train(GAMES)

    t0 = time.time()
    try:
        recommendations = model.predict_by_game_id(game_id, k)
        latency = time.time() - t0
        monitor.record_prediction("recommend_by_game", f"game_id:{game_id}", recommendations, latency)
        return {"game_id": game_id, "recommendations": recommendations, "latency_ms": latency * 1000}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        monitor.record_error("recommend_by_game", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Compat Streamlit: par titre / par genre
@app.get("/recommend/by-title/{title}", tags=["recommend"])
@measure_latency("recommend_by_title")
def recommend_by_title(
    title: str,
    k: int = Query(10, ge=1, le=50),
    user: str = Depends(verify_token),
):
    model = get_model()
    monitor = get_monitor()
    if not model.is_trained:
        model.train(GAMES)

    df = model.games_df
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No games loaded")

    # matching simple: similarité TF-IDF sur le titre
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(stop_words="english").fit(df["title"])
    qv = vec.transform([title])
    sims = cosine_similarity(qv, vec.transform(df["title"]))[0]
    best_idx = int(sims.argmax())
    best_id = int(df.iloc[best_idx]["id"])
    recs = model.predict_by_game_id(best_id, k=k)
    monitor.record_prediction("recommend_by_title", f"title:{title}", recs, 0.0)
    return {"title": title, "base_match_id": best_id, "recommendations": recs}

@app.get("/recommend/by-genre/{genre}", tags=["recommend"])
@measure_latency("recommend_by_genre")
def recommend_by_genre(
    genre: str,
    k: int = Query(10, ge=1, le=50),
    min_confidence: float = Query(0.1, ge=0.0, le=1.0),
    user: str = Depends(verify_token),
):
    model = get_model()
    monitor = get_monitor()
    if not model.is_trained:
        model.train(GAMES)
    recs = model.predict(genre, k=k, min_confidence=min_confidence)
    monitor.record_prediction("recommend_by_genre", f"genre:{genre}", recs, 0.0)
    return {"genre": genre, "recommendations": recs}

# ========= Auth (MySQL) & Games =========
@app.post("/register", tags=["auth"])
def register(username: str = Form(...), password: str = Form(...)):
    """Crée un utilisateur dans la base MySQL (table `users`)."""
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        raise HTTPException(status_code=400, detail=f"Password too short (min {settings.PASSWORD_MIN_LENGTH})")
    # (optionnel) regex de complexité
    # import re
    # if not re.match(settings.PASSWORD_REGEX, password):
    #     raise HTTPException(status_code=400, detail="Password does not meet complexity requirements")
    if _get_user(username):
        raise HTTPException(status_code=400, detail="User already exists")
    hashed = pwd_ctx.hash(password)
    try:
        _create_user(username, hashed)
        return {"ok": True}
    except Exception as e:
        logger.error(f"/register failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/token", tags=["auth"])
def token(username: str = Form(...), password: str = Form(...)):
    """Vérifie identifiants depuis MySQL et génère un JWT."""
    row = _get_user(username)
    if not row:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    hpwd = row["hashed_password"]
    if not pwd_ctx.verify(password, hpwd):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token({"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/games/by-title/{title}", tags=["games"])
def games_by_title(title: str, user: str = Depends(verify_token)):
    q = title.strip().lower()
    results = [
        {
            "id": g["id"],
            "title": g["title"],
            "genres": g["genres"],
            "rating": g["rating"],
            "metacritic": g["metacritic"],
            "platforms": ", ".join(g["platforms"]),
        }
        for g in GAMES
        if q in g["title"].lower()
    ]
    return {"results": results}

# ========= Monitoring extra =========
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
    return {
        "drift_score": drift_score,
        "status": status,
        "recommendation": "Retrain model" if status == "high_drift" else ("Monitor closely" if status == "moderate_drift" else "No action needed"),
    }

# ========= Startup =========
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Games API with ML...")

    # Init table users
    try:
        _init_users_table()
        logger.info("Users table ready")
    except Exception as e:
        logger.error(f"Failed to init users table: {e}")

    # Charger modèle si dispo
    model = get_model()
    if os.path.exists("model/recommendation_model.pkl"):
        if model.load_model():
            logger.info(f"Model {model.model_version} loaded successfully")
        else:
            logger.warning("Failed to load saved model, will train on first request")
    else:
        logger.info("No saved model found, will train on first request")

    logger.info(f"API startup complete. {len(GAMES)} games available.")
