# api_games_enhanced.py - Version améliorée de l'API avec ML complet
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Form, Request, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from passlib.context import CryptContext
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# Import des modules ML
from model_manager import get_model, RecommendationModel
from monitoring_metrics import get_monitor, measure_latency


logger = logging.getLogger("games-api-ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# ========= Config =========
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(
    title="Games API ML (C9→C13)", 
    version="2.0.0", 
    description="API avec modèle ML, monitoring avancé et pipeline MLOps"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Data Models =========
class TrainRequest(BaseModel):
    version: Optional[str] = None
    force_retrain: bool = False

class PredictionRequest(BaseModel):
    query: str
    k: int = 10
    min_confidence: float = 0.1

class ModelMetricsResponse(BaseModel):
    model_version: str
    is_trained: bool
    total_predictions: int
    avg_confidence: float
    last_training: Optional[str]
    games_count: int
    feature_dimension: int

# ========= Demo Data (identique à l'original) =========
GAMES: List[Dict] = [
    {"id": 1, "title": "The Witcher 3: Wild Hunt", "genres": "RPG, Action", "rating": 4.9, "metacritic": 93, "platforms": ["PC", "PS4", "Xbox One"]},
    {"id": 2, "title": "Hades", "genres": "Action, Roguelike", "rating": 4.8, "metacritic": 93, "platforms": ["PC", "Switch", "PS4"]},
    {"id": 3, "title": "Stardew Valley", "genres": "Simulation, RPG", "rating": 4.7, "metacritic": 89, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 4, "title": "Celeste", "genres": "Platformer, Indie", "rating": 4.6, "metacritic": 94, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 5, "title": "Doom Eternal", "genres": "Action, FPS", "rating": 4.5, "metacritic": 88, "platforms": ["PC", "PS4", "Xbox One"]},
    {"id": 6, "title": "Hollow Knight", "genres": "Metroidvania, Indie", "rating": 4.8, "metacritic": 90, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 7, "title": "Disco Elysium", "genres": "RPG, Narrative", "rating": 4.7, "metacritic": 91, "platforms": ["PC", "PS4", "Xbox One"]},
]

USERS: Dict[str, str] = {}

# ========= Utils (identiques) =========
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
        "monitoring": monitor.get_metrics_summary()
    }

@app.get("/", include_in_schema=False)
def root():
    return {
        "name": app.title,
        "version": app.version,
        "status": "ok"
    }



# Prometheus metrics avec Instrumentator
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, include_in_schema=True)

# ========= ML Model Management =========
@app.post("/model/train", tags=["model"], response_model=Dict)
def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(verify_token)
):
    """Entraîne ou ré-entraîne le modèle de recommandation"""
    model = get_model()
    monitor = get_monitor()
    
    if model.is_trained and not request.force_retrain:
        return {"status": "already_trained", "version": model.model_version}
    
    # Entraînement
    start_time = time.time()
    version = request.version or f"api-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    model.model_version = version
    
    try:
        result = model.train(GAMES)
        duration = time.time() - start_time
        
        # Enregistrer les métriques
        monitor.record_training(
            model_version=version,
            games_count=len(GAMES),
            feature_dim=model.game_features.shape[1],
            duration=duration,
            metrics=result
        )
        
        # Sauvegarder le modèle en arrière-plan
        background_tasks.add_task(model.save_model)
        
        return {
            "status": "success",
            "version": version,
            "duration": duration,
            "result": result
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/metrics", tags=["model"], response_model=ModelMetricsResponse)
def get_model_metrics(user: str = Depends(verify_token)):
    """Retourne les métriques du modèle"""
    model = get_model()
    metrics = model.get_metrics()
    
    return ModelMetricsResponse(
        model_version=metrics.get("model_version", "unknown"),
        is_trained=metrics.get("is_trained", False),
        total_predictions=metrics.get("total_predictions", 0),
        avg_confidence=metrics.get("avg_confidence", 0.0),
        last_training=metrics.get("last_training"),
        games_count=metrics.get("games_count", 0),
        feature_dimension=metrics.get("feature_dim", 0)
    )

@app.post("/model/evaluate", tags=["model"])
def evaluate_model(
    test_queries: List[str] = Query(default=["RPG", "Action", "Indie", "Simulation"]),
    user: str = Depends(verify_token)
):
    """Évalue le modèle avec des requêtes de test"""
    model = get_model()
    monitor = get_monitor()
    
    if not model.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    results = monitor.evaluate_model(model, test_queries)
    return results

# ========= Enhanced Recommendations with ML =========
@app.post("/recommend/ml", tags=["recommend"])
@measure_latency("recommend_ml")
def recommend_ml(
    request: PredictionRequest,
    user: str = Depends(verify_token)
):
    """Recommandations basées sur le modèle ML"""
    model = get_model()
    monitor = get_monitor()
    
    if not model.is_trained:
        # Auto-train si pas encore fait
        logger.info("Model not trained, auto-training...")
        model.train(GAMES)
        model.save_model()
    
    start_time = time.time()
    
    try:
        recommendations = model.predict(
            query=request.query,
            k=request.k,
            min_confidence=request.min_confidence
        )
        
        latency = time.time() - start_time
        
        # Enregistrer la prédiction
        monitor.record_prediction(
            endpoint="recommend_ml",
            query=request.query,
            recommendations=recommendations,
            latency=latency
        )
        
        return {
            "query": request.query,
            "recommendations": recommendations,
            "latency_ms": latency * 1000,
            "model_version": model.model_version
        }
        
    except Exception as e:
        monitor.record_error("recommend_ml", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/by-game/{game_id}", tags=["recommend"])
@measure_latency("recommend_by_game")
def recommend_by_game(
    game_id: int,
    k: int = Query(10, ge=1, le=50),
    user: str = Depends(verify_token)
):
    """Recommandations basées sur un jeu spécifique"""
    model = get_model()
    monitor = get_monitor()
    
    if not model.is_trained:
        model.train(GAMES)
    
    start_time = time.time()
    
    try:
        recommendations = model.predict_by_game_id(game_id, k)
        latency = time.time() - start_time
        
        monitor.record_prediction(
            endpoint="recommend_by_game",
            query=f"game_id:{game_id}",
            recommendations=recommendations,
            latency=latency
        )
        
        return {
            "game_id": game_id,
            "recommendations": recommendations,
            "latency_ms": latency * 1000
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        monitor.record_error("recommend_by_game", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ========= Original endpoints (preserved) =========
@app.post("/register", tags=["auth"])
def register(username: str = Form(...), password: str = Form(...)):
    if username in USERS:
        raise HTTPException(status_code=400, detail="User already exists")
    USERS[username] = pwd_ctx.hash(password)
    return {"ok": True}

@app.post("/token", tags=["auth"])
def token(username: str = Form(...), password: str = Form(...)):
    hpwd = USERS.get(username)
    if not hpwd or not pwd_ctx.verify(password, hpwd):
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

# ========= Enhanced Monitoring Endpoints =========
@app.get("/monitoring/summary", tags=["monitoring"])
def monitoring_summary(user: str = Depends(verify_token)):
    """Résumé complet du monitoring"""
    monitor = get_monitor()
    model = get_model()
    
    return {
        "model": model.get_metrics(),
        "monitoring": monitor.get_metrics_summary(),
        "last_evaluation": monitor.last_evaluation
    }

@app.get("/monitoring/drift", tags=["monitoring"])
def check_drift(user: str = Depends(verify_token)):
    """Vérifie le drift du modèle"""
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
        "recommendation": "Retrain model" if status == "high_drift" else "Monitor closely" if status == "moderate_drift" else "No action needed"
    }

# ========= Startup =========
@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    logger.info("Starting Games API with ML...")
    
    # Charger le modèle si disponible
    model = get_model()
    if os.path.exists("model/recommendation_model.pkl"):
        if model.load_model():
            logger.info(f"Model {model.model_version} loaded successfully")
        else:
            logger.warning("Failed to load saved model, will train on first request")
    else:
        logger.info("No saved model found, will train on first request")
    
    logger.info(f"API startup complete. {len(GAMES)} games available.")

@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "
