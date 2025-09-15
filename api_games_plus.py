# api_games_plus.py - Enhanced ML/AI API with multiple models
from __future__ import annotations

import os
import time
import math
import logging
import inspect
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union

import pymysql
from fastapi import FastAPI, HTTPException, Depends, Form, Query, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Import des modules locaux
from settings import get_settings
from model_manager import (
    get_model, get_model_manager, MLModelManager, ModelType,
    RecommendationModel, GameClassificationModel, GameClusteringModel
)
from monitoring_metrics import (
    prediction_latency, model_prediction_counter, get_monitor,
)

# Logging setup
logger = logging.getLogger("games-api-ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# Compliance import avec fallback gracieux
try:
    from compliance.standards_compliance import SecurityValidator, AccessibilityValidator
    COMPLIANCE_AVAILABLE = True
    logger.info("Compliance module loaded successfully")
except ImportError as e:
    COMPLIANCE_AVAILABLE = False
    logger.warning(f"Compliance module not available: {e}")

# Settings & Security
settings = get_settings()

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

pwd_ctx = CryptContext(
    schemes=["bcrypt", "pbkdf2_sha256", "sha256_crypt"],
    deprecated="auto",
)

# Flags DB
DB_READY: bool = False
DB_LAST_ERROR: Optional[str] = None

# FastAPI app
app = FastAPI(
    title="Games AI/ML API (Enhanced)",
    version="3.0.0",
    description="API avancée avec modèles IA multiples pour recommandations, classification et clustering de jeux",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in settings.ALLOW_ORIGINS.split(",") if o] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup compliance
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
    return False

COMPLIANCE_ENABLED = setup_compliance()

# =====================================================================
# MODÈLES PYDANTIC ÉTENDUS
# =====================================================================

class TrainRequest(BaseModel):
    version: Optional[str] = None
    force_retrain: bool = False
    algorithm: Optional[str] = "hybrid"  # Pour recommendation model

class PredictionRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=50)
    min_confidence: float = Field(default=0.1, ge=0.0, le=1.0)
    user_profile: Optional[Dict] = None

class ClassificationRequest(BaseModel):
    game_description: str
    title: str = ""
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    metacritic: int = Field(default=0, ge=0, le=100)

class ClusteringRequest(BaseModel):
    game_description: str
    title: str = ""
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    metacritic: int = Field(default=0, ge=0, le=100)

class BatchPredictionRequest(BaseModel):
    queries: List[str]
    k: int = Field(default=5, ge=1, le=20)
    min_confidence: float = Field(default=0.1, ge=0.0, le=1.0)

class ModelComparisonRequest(BaseModel):
    query: str
    algorithms: List[str] = ["tfidf", "nmf", "hybrid"]
    k: int = Field(default=5, ge=1, le=20)

class UserProfileRequest(BaseModel):
    user_id: str
    preferred_genres: List[str] = []
    preferred_platforms: List[str] = []
    seen_games: List[int] = []
    rating_preference: Optional[str] = None  # "high", "medium", "any"

class ModelMetricsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_version: str
    model_type: str
    is_trained: bool
    total_predictions: int
    avg_confidence: float
    last_training: Optional[str]
    algorithm: Optional[str] = None
    accuracy: Optional[float] = None

# =====================================================================
# HELPERS DB ET AUTH (identiques)
# =====================================================================

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

# [Functions DB et Auth identiques à l'original...]
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

def get_user(username: str) -> Optional[dict]:
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        return cur.fetchone()

def create_user(username: str, password: str) -> None:
    hpwd = pwd_ctx.hash(password)
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO users(username, hashed_password) VALUES(%s,%s)", (username, hpwd))
        conn.commit()

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
            "rating": float(r.get("rating") or 0),
            "metacritic": int(r.get("metacritic") or 0),
            "platforms": r.get("platforms", "").split(",") if r.get("platforms") else [],
        })
    return games

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

# =====================================================================
# HELPERS COMPLIANCE
# =====================================================================

def get_security_validator():
    if COMPLIANCE_ENABLED and hasattr(app.state, 'security_validator'):
        return app.state.security_validator
    return None

def get_accessibility_validator():
    if COMPLIANCE_ENABLED and hasattr(app.state, 'accessibility_validator'):
        return app.state.accessibility_validator
    return None

# =====================================================================
# DECORATEUR LATENCE
# =====================================================================

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

# =====================================================================
# ENDPOINTS SYSTÈME
# =====================================================================

@app.get("/healthz", tags=["system"])
def healthz():
    model_manager = get_model_manager()
    models_info = model_manager.list_models()
    
    return {
        "status": "healthy" if (DB_READY or not settings.DB_REQUIRED) else "degraded",
        "time": datetime.utcnow().isoformat(),
        "db_ready": DB_READY,
        "db_error": DB_LAST_ERROR,
        "available_models": len(models_info),
        "models": models_info,
        "compliance_enabled": COMPLIANCE_ENABLED,
        "api_version": "3.0.0"
    }

@app.get("/", include_in_schema=False)
def root():
    return {
        "name": app.title, 
        "version": "3.0.0", 
        "status": "ok",
        "features": ["recommendations", "classification", "clustering", "multi-model"]
    }

Instrumentator().instrument(app).expose(app, include_in_schema=True)

# =====================================================================
# ENDPOINTS GESTION MODÈLES
# =====================================================================

@app.get("/models", tags=["models"])
def list_models(user: str = Depends(verify_token)):
    """Liste tous les modèles disponibles"""
    model_manager = get_model_manager()
    return {
        "models": model_manager.list_models(),
        "supported_types": [ModelType.RECOMMENDATION, ModelType.CLASSIFICATION, ModelType.CLUSTERING]
    }

@app.post("/models/{model_type}/create", tags=["models"])
def create_model(
    model_type: str,
    model_id: str = Query(..., description="Identifiant unique du modèle"),
    algorithm: Optional[str] = Query(None, description="Algorithme (pour recommendation)"),
    n_clusters: Optional[int] = Query(8, description="Nombre de clusters (pour clustering)"),
    user: str = Depends(verify_token)
):
    """Crée un nouveau modèle"""
    model_manager = get_model_manager()
    
    try:
        kwargs = {}
        if model_type == ModelType.RECOMMENDATION and algorithm:
            kwargs["algorithm"] = algorithm
        elif model_type == ModelType.CLUSTERING and n_clusters:
            kwargs["n_clusters"] = n_clusters
        
        model = model_manager.create_model(model_type, model_id, **kwargs)
        
        return {
            "status": "created",
            "model_id": model_id,
            "model_type": model_type,
            "model_info": model.get_info()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/models/{model_id}/train", tags=["models"])
def train_model(
    model_id: str,
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    user: str = Depends(verify_token)
):
    """Entraîne un modèle spécifique"""
    model_manager = get_model_manager()
    model = model_manager.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not request.force_retrain and model.is_trained:
        return {
            "status": "already_trained",
            "model_id": model_id,
            "model_info": model.get_info()
        }
    
    try:
        games = fetch_games_for_ml()
        start_time = time.time()
        
        # Configuration spécifique pour recommendation model
        if hasattr(model, 'algorithm') and request.algorithm:
            model.algorithm = request.algorithm
        
        result = model.train(games)
        duration = time.time() - start_time
        
        # Sauvegarde en arrière-plan
        background_tasks.add_task(
            model_manager.save_model, 
            model_id, 
            f"model/{model_id}_{model.model_type}.pkl"
        )
        
        return {
            "status": "success",
            "model_id": model_id,
            "duration": duration,
            "training_result": result
        }
        
    except Exception as e:
        logger.error(f"Training failed for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}/metrics", tags=["models"], response_model=ModelMetricsResponse)
def get_model_metrics(model_id: str, user: str = Depends(verify_token)):
    """Récupère les métriques d'un modèle"""
    model_manager = get_model_manager()
    model = model_manager.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    metrics = model.get_info()
    
    # Métriques spécifiques selon le type
    if hasattr(model, 'metrics'):
        metrics.update(model.metrics)
    
    return ModelMetricsResponse(
        model_version=metrics.get("version", "unknown"),
        model_type=metrics.get("type", "unknown"),
        is_trained=metrics.get("is_trained", False),
        total_predictions=metrics.get("total_predictions", 0),
        avg_confidence=metrics.get("avg_confidence", 0.0),
        last_training=metrics.get("last_updated"),
        algorithm=metrics.get("algorithm"),
        accuracy=metrics.get("accuracy")
    )

# =====================================================================
# ENDPOINTS RECOMMANDATIONS AVANCÉES
# =====================================================================

@app.post("/recommend/ml", tags=["recommendations"])
@measure_latency("recommend_ml")
def recommend_ml(request: PredictionRequest, user: str = Depends(verify_token)):
    """Recommandations ML avec le modèle principal"""
    model = get_model()  # Modèle de recommandation principal
    
    if not model.is_trained:
        # Auto-entraînement si nécessaire
        games = fetch_games_for_ml()
        model.train(games)
    
    # Nettoyage avec compliance
    clean_query = request.query.strip()
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
            min_confidence=request.min_confidence,
            user_profile=request.user_profile
        )
    except Exception as e:
        logger.warning("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    latency = time.time() - start_time
    get_monitor().record_prediction("recommend_ml", clean_query, recommendations, latency)
    
    response = {
        "query": clean_query,
        "recommendations": recommendations,
        "latency_ms": latency * 1000,
        "model_version": model.model_version,
        "algorithm": model.algorithm,
        "user_profile_applied": request.user_profile is not None
    }
    
    # Enhancement accessibilité
    accessibility_validator = get_accessibility_validator()
    if accessibility_validator:
        try:
            response = accessibility_validator.enhance_response_accessibility(response)
        except Exception as e:
            logger.warning(f"Accessibility enhancement failed: {e}")
    
    return response

@app.post("/recommend/model/{model_id}", tags=["recommendations"])
@measure_latency("recommend_model")
def recommend_with_model(
    model_id: str, 
    request: PredictionRequest, 
    user: str = Depends(verify_token)
):
    """Recommandations avec un modèle spécifique"""
    model_manager = get_model_manager()
    model = model_manager.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.model_type != ModelType.RECOMMENDATION:
        raise HTTPException(status_code=400, detail="Model is not a recommendation model")
    
    if not model.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")
    
    try:
        recommendations = model.predict(
            query=request.query,
            k=request.k,
            min_confidence=request.min_confidence,
            user_profile=request.user_profile
        )
        
        return {
            "model_id": model_id,
            "query": request.query,
            "recommendations": recommendations,
            "algorithm": getattr(model, 'algorithm', 'unknown')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/batch", tags=["recommendations"])
@measure_latency("recommend_batch")
def recommend_batch(request: BatchPredictionRequest, user: str = Depends(verify_token)):
    """Recommandations par lot pour plusieurs requêtes"""
    model = get_model()
    
    if not model.is_trained:
        games = fetch_games_for_ml()
        model.train(games)
    
    if len(request.queries) > 20:
        raise HTTPException(status_code=400, detail="Too many queries (max 20)")
    
    results = []
    start_time = time.time()
    
    for i, query in enumerate(request.queries):
        try:
            recommendations = model.predict(
                query=query,
                k=request.k,
                min_confidence=request.min_confidence
            )
            results.append({
                "query_index": i,
                "query": query,
                "recommendations": recommendations,
                "count": len(recommendations)
            })
        except Exception as e:
            results.append({
                "query_index": i,
                "query": query,
                "error": str(e),
                "recommendations": []
            })
    
    total_latency = time.time() - start_time
    
    return {
        "batch_size": len(request.queries),
        "results": results,
        "total_latency_ms": total_latency * 1000,
        "avg_latency_per_query_ms": (total_latency * 1000) / len(request.queries)
    }

@app.post("/recommend/compare", tags=["recommendations"])
@measure_latency("recommend_compare")
def compare_algorithms(request: ModelComparisonRequest, user: str = Depends(verify_token)):
    """Compare plusieurs algorithmes de recommandation"""
    model_manager = get_model_manager()
    results = []
    
    for algorithm in request.algorithms:
        try:
            # Créer ou récupérer le modèle pour cet algorithme
            model_id = f"recommendation_{algorithm}"
            model = model_manager.get_model(model_id)
            
            if not model:
                model = model_manager.create_model(
                    ModelType.RECOMMENDATION, 
                    model_id, 
                    algorithm=algorithm
                )
                # Entraîner si nécessaire
                if not model.is_trained:
                    games = fetch_games_for_ml()
                    model.train(games)
            
            start_time = time.time()
            recommendations = model.predict(
                query=request.query,
                k=request.k
            )
            latency = time.time() - start_time
            
            results.append({
                "algorithm": algorithm,
                "recommendations": recommendations,
                "latency_ms": latency * 1000,
                "count": len(recommendations),
                "avg_confidence": np.mean([r['confidence'] for r in recommendations]) if recommendations else 0
            })
            
        except Exception as e:
            results.append({
                "algorithm": algorithm,
                "error": str(e),
                "recommendations": []
            })
    
    return {
        "query": request.query,
        "comparison_results": results,
        "best_algorithm": max(results, key=lambda x: x.get('avg_confidence', 0))['algorithm'] if results else None
    }

# =====================================================================
# ENDPOINTS CLASSIFICATION
# =====================================================================

@app.post("/classify/game", tags=["classification"])
@measure_latency("classify_game")
def classify_game(request: ClassificationRequest, user: str = Depends(verify_token)):
    """Classifie un jeu par genre/catégorie"""
    model_manager = get_model_manager()
    model_id = "classification_default"
    model = model_manager.get_model(model_id)
    
    if not model:
        # Créer et entraîner le modèle de classification
        model = model_manager.create_model(ModelType.CLASSIFICATION, model_id)
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        prediction = model.predict(
            game_description=request.game_description,
            title=request.title,
            rating=request.rating,
            metacritic=request.metacritic
        )
        
        return {
            "classification_result": prediction,
            "model_id": model_id,
            "input": {
                "title": request.title,
                "description": request.game_description,
                "rating": request.rating,
                "metacritic": request.metacritic
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch", tags=["classification"])
@measure_latency("classify_batch")
def classify_games_batch(
    games_data: List[ClassificationRequest], 
    user: str = Depends(verify_token)
):
    """Classification par lot de plusieurs jeux"""
    if len(games_data) > 50:
        raise HTTPException(status_code=400, detail="Too many games (max 50)")
    
    model_manager = get_model_manager()
    model = model_manager.get_model("classification_default")
    
    if not model:
        model = model_manager.create_model(ModelType.CLASSIFICATION, "classification_default")
        games = fetch_games_for_ml()
        model.train(games)
    
    results = []
    
    for i, game_request in enumerate(games_data):
        try:
            prediction = model.predict(
                game_description=game_request.game_description,
                title=game_request.title,
                rating=game_request.rating,
                metacritic=game_request.metacritic
            )
            results.append({
                "index": i,
                "title": game_request.title,
                "classification": prediction
            })
        except Exception as e:
            results.append({
                "index": i,
                "title": game_request.title,
                "error": str(e)
            })
    
    return {
        "batch_size": len(games_data),
        "results": results
    }

# =====================================================================
# ENDPOINTS CLUSTERING
# =====================================================================

@app.post("/cluster/game", tags=["clustering"])
@measure_latency("cluster_game")
def cluster_game(request: ClusteringRequest, user: str = Depends(verify_token)):
    """Détermine le cluster d'appartenance d'un jeu"""
    model_manager = get_model_manager()
    model_id = "clustering_default"
    model = model_manager.get_model(model_id)
    
    if not model:
        model = model_manager.create_model(ModelType.CLUSTERING, model_id)
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        prediction = model.predict(
            game_description=request.game_description,
            title=request.title,
            rating=request.rating,
            metacritic=request.metacritic
        )
        
        return {
            "clustering_result": prediction,
            "model_id": model_id,
            "input": {
                "title": request.title,
                "description": request.game_description,
                "rating": request.rating,
                "metacritic": request.metacritic
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cluster/analysis", tags=["clustering"])
def get_cluster_analysis(user: str = Depends(verify_token)):
    """Analyse des clusters existants"""
    model_manager = get_model_manager()
    model = model_manager.get_model("clustering_default")
    
    if not model or not model.is_trained:
        raise HTTPException(status_code=400, detail="Clustering model not available")
    
    return {
        "cluster_descriptions": getattr(model, 'cluster_descriptions', {}),
        "n_clusters": getattr(model, 'n_clusters', 0),
        "model_info": model.get_info()
    }

# =====================================================================
# ENDPOINTS GESTION PROFILS UTILISATEUR
# =====================================================================

@app.post("/user/profile", tags=["user"])
def create_user_profile(request: UserProfileRequest, user: str = Depends(verify_token)):
    """Crée ou met à jour un profil utilisateur pour personnalisation"""
    # Stockage simple en mémoire pour la démo
    # En production, utiliser la base de données
    
    try:
        # Validation du profil
        profile_data = {
            "user_id": request.user_id,
            "preferred_genres": request.preferred_genres,
            "preferred_platforms": request.preferred_platforms,
            "seen_games": request.seen_games,
            "rating_preference": request.rating_preference,
            "created_at": datetime.utcnow().isoformat(),
            "updated_by": user
        }
        
        # En production: sauvegarder en DB
        # save_user_profile(profile_data)
        
        return {
            "status": "success",
            "profile": profile_data,
            "recommendations_personalized": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/recommendations/{user_id}", tags=["user"])
@measure_latency("user_recommendations")
def get_personalized_recommendations(
    user_id: str,
    query: str = Query(..., description="Requête de recherche"),
    k: int = Query(10, ge=1, le=50),
    user: str = Depends(verify_token)
):
    """Recommandations personnalisées basées sur le profil utilisateur"""
    # Récupérer le profil utilisateur (simulé)
    user_profile = {
        "preferred_genres": ["RPG", "Action"],
        "preferred_platforms": ["PC", "PS4"],
        "seen_games": [1, 2, 3],
        "rating_preference": "high"
    }
    
    model = get_model()
    if not model.is_trained:
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        recommendations = model.predict(
            query=query,
            k=k,
            user_profile=user_profile
        )
        
        return {
            "user_id": user_id,
            "query": query,
            "recommendations": recommendations,
            "personalization_applied": True,
            "user_profile_summary": {
                "preferred_genres": user_profile["preferred_genres"],
                "preferred_platforms": user_profile["preferred_platforms"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# ENDPOINTS AUTH (identiques à l'original)
# =====================================================================

@app.post("/token", tags=["auth"])
def token(username: str = Form(...), password: str = Form(...)):
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

    stored_password = u.get('hashed_password', '')
    if not stored_password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    try:
        if not pwd_ctx.verify(password, stored_password):
            raise HTTPException(status_code=401, detail="Incorrect username or password")
    except UnknownHashError:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token({"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", tags=["auth"])
def register(username: str = Form(...), password: str = Form(...)):
    if not DB_READY:
        raise HTTPException(status_code=503, detail="Database unavailable: cannot register now")
    
    username = username.strip()
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

# =====================================================================
# ENDPOINTS ANALYSE ET ÉVALUATION
# =====================================================================

@app.post("/analyze/model-performance", tags=["analysis"])
def analyze_model_performance(
    test_queries: List[str] = Query(default=["RPG", "Action", "Indie", "Strategy"]),
    user: str = Depends(verify_token)
):
    """Analyse comparative de performance des modèles"""
    model_manager = get_model_manager()
    algorithms = ["tfidf", "nmf", "hybrid"]
    results = {}
    
    for algorithm in algorithms:
        model_id = f"recommendation_{algorithm}"
        model = model_manager.get_model(model_id)
        
        if not model:
            model = model_manager.create_model(ModelType.RECOMMENDATION, model_id, algorithm=algorithm)
            games = fetch_games_for_ml()
            model.train(games)
        
        # Tests de performance
        algorithm_results = {
            "latencies": [],
            "confidences": [],
            "recommendation_counts": [],
            "diversity_scores": []
        }
        
        for query in test_queries:
            start_time = time.time()
            recommendations = model.predict(query, k=10)
            latency = time.time() - start_time
            
            algorithm_results["latencies"].append(latency * 1000)
            
            if recommendations:
                confidences = [r['confidence'] for r in recommendations]
                algorithm_results["confidences"].extend(confidences)
                algorithm_results["recommendation_counts"].append(len(recommendations))
                
                # Diversité (nombre de genres uniques)
                genres = set()
                for rec in recommendations:
                    genres.update(rec.get('genres', '').split(','))
                algorithm_results["diversity_scores"].append(len(genres))
        
        # Statistiques agrégées
        results[algorithm] = {
            "avg_latency_ms": np.mean(algorithm_results["latencies"]),
            "avg_confidence": np.mean(algorithm_results["confidences"]) if algorithm_results["confidences"] else 0,
            "avg_recommendations": np.mean(algorithm_results["recommendation_counts"]) if algorithm_results["recommendation_counts"] else 0,
            "avg_diversity": np.mean(algorithm_results["diversity_scores"]) if algorithm_results["diversity_scores"] else 0,
            "model_info": model.get_info()
        }
    
    # Analyse comparative
    best_algorithm = max(results.keys(), key=lambda alg: results[alg]["avg_confidence"])
    fastest_algorithm = min(results.keys(), key=lambda alg: results[alg]["avg_latency_ms"])
    most_diverse = max(results.keys(), key=lambda alg: results[alg]["avg_diversity"])
    
    return {
        "performance_analysis": results,
        "test_queries": test_queries,
        "summary": {
            "best_quality": best_algorithm,
            "fastest": fastest_algorithm,
            "most_diverse": most_diverse
        },
        "recommendations": {
            "production": "hybrid" if "hybrid" in results else best_algorithm,
            "speed_critical": fastest_algorithm,
            "diversity_focused": most_diverse
        }
    }

@app.post("/analyze/data-insights", tags=["analysis"])
def analyze_data_insights(user: str = Depends(verify_token)):
    """Analyse des insights sur les données de jeux"""
    try:
        games = fetch_games_for_ml()
        df = pd.DataFrame(games)
        
        # Statistiques générales
        stats = {
            "total_games": len(df),
            "avg_rating": float(df['rating'].mean()),
            "avg_metacritic": float(df['metacritic'].mean()),
            "rating_distribution": df['rating'].describe().to_dict(),
            "metacritic_distribution": df['metacritic'].describe().to_dict()
        }
        
        # Analyse des genres
        all_genres = []
        for genres_str in df['genres']:
            if pd.notna(genres_str):
                all_genres.extend([g.strip() for g in str(genres_str).split(',')])
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        stats["top_genres"] = genre_counts.to_dict()
        
        # Analyse des plateformes
        all_platforms = []
        for platforms in df['platforms']:
            if isinstance(platforms, list):
                all_platforms.extend(platforms)
        
        platform_counts = pd.Series(all_platforms).value_counts().head(10)
        stats["top_platforms"] = platform_counts.to_dict()
        
        # Corrélations
        numeric_df = df[['rating', 'metacritic']].dropna()
        if len(numeric_df) > 1:
            correlation = float(numeric_df.corr().iloc[0, 1])
            stats["rating_metacritic_correlation"] = correlation
        
        # Recommandations d'amélioration
        recommendations = []
        
        if stats["total_games"] < 100:
            recommendations.append("Consider expanding the game dataset for better ML performance")
        
        if len(stats["top_genres"]) < 5:
            recommendations.append("Dataset could benefit from more genre diversity")
        
        if stats["rating_metacritic_correlation"] < 0.3:
            recommendations.append("Low correlation between rating and metacritic scores - investigate data quality")
        
        stats["improvement_recommendations"] = recommendations
        
        return {
            "data_insights": stats,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# ENDPOINTS UPLOAD ET TRAITEMENT FICHIERS
# =====================================================================

@app.post("/upload/games-dataset", tags=["data"])
async def upload_games_dataset(
    file: UploadFile = File(...),
    user: str = Depends(verify_token)
):
    """Upload d'un dataset de jeux au format CSV"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Lecture du fichier
        content = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        
        # Validation des colonnes requises
        required_columns = ['title', 'genres']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Nettoyage des données
        df = df.dropna(subset=['title'])
        df['rating'] = pd.to_numeric(df.get('rating', 0), errors='coerce').fillna(0)
        df['metacritic'] = pd.to_numeric(df.get('metacritic', 0), errors='coerce').fillna(0)
        
        # Conversion en format compatible
        games_data = []
        for _, row in df.iterrows():
            game = {
                "id": len(games_data) + 1000,  # IDs temporaires
                "title": str(row['title']),
                "genres": str(row.get('genres', '')),
                "rating": float(row.get('rating', 0)),
                "metacritic": int(row.get('metacritic', 0)),
                "platforms": str(row.get('platforms', '')).split(',') if row.get('platforms') else []
            }
            games_data.append(game)
        
        # Entraîner automatiquement les modèles avec les nouvelles données
        model_manager = get_model_manager()
        training_results = {}
        
        for model_type in [ModelType.RECOMMENDATION, ModelType.CLASSIFICATION, ModelType.CLUSTERING]:
            try:
                model_id = f"{model_type}_from_upload"
                
                if model_type == ModelType.RECOMMENDATION:
                    model = model_manager.create_model(model_type, model_id, algorithm="hybrid")
                elif model_type == ModelType.CLUSTERING:
                    model = model_manager.create_model(model_type, model_id, n_clusters=min(8, len(games_data)//10))
                else:
                    model = model_manager.create_model(model_type, model_id)
                
                result = model.train(games_data)
                training_results[model_type] = {
                    "status": "success",
                    "model_id": model_id,
                    "result": result
                }
                
            except Exception as e:
                training_results[model_type] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return {
            "upload_status": "success",
            "file_info": {
                "filename": file.filename,
                "size_bytes": len(content),
                "rows_processed": len(games_data),
                "columns": list(df.columns)
            },
            "data_summary": {
                "total_games": len(games_data),
                "avg_rating": float(df['rating'].mean()) if 'rating' in df.columns else 0,
                "unique_genres": len(set(str(row.get('genres', '')).split(',')[0] for _, row in df.iterrows()))
            },
            "training_results": training_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

# =====================================================================
# ENDPOINTS MONITORING AVANCÉ
# =====================================================================

@app.get("/monitoring/models-status", tags=["monitoring"])
def get_models_status(user: str = Depends(verify_token)):
    """Status détaillé de tous les modèles"""
    model_manager = get_model_manager()
    models_info = model_manager.list_models()
    
    status_summary = {
        "total_models": len(models_info),
        "trained_models": 0,
        "models_by_type": {},
        "models_details": []
    }
    
    for model_data in models_info:
        model_id = model_data["id"]
        model_info = model_data["info"]
        model_type = model_info["type"]
        
        if model_info["is_trained"]:
            status_summary["trained_models"] += 1
        
        if model_type not in status_summary["models_by_type"]:
            status_summary["models_by_type"][model_type] = 0
        status_summary["models_by_type"][model_type] += 1
        
        # Informations détaillées
        model = model_manager.get_model(model_id)
        detailed_info = {
            "model_id": model_id,
            "type": model_type,
            "is_trained": model_info["is_trained"],
            "version": model_info["version"],
            "created_at": model_info["created_at"],
            "last_updated": model_info["last_updated"]
        }
        
        # Métriques spécifiques
        if hasattr(model, 'metrics'):
            detailed_info["metrics"] = model.metrics
        
        status_summary["models_details"].append(detailed_info)
    
    return status_summary

@app.get("/monitoring/system-health", tags=["monitoring"])
def get_system_health(user: str = Depends(verify_token)):
    """Health check système complet"""
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "components": {},
        "performance_metrics": {},
        "recommendations": []
    }
    
    # Vérification base de données
    try:
        if DB_READY:
            games = fetch_games_for_ml()
            health_status["components"]["database"] = {
                "status": "healthy",
                "games_count": len(games)
            }
        else:
            health_status["components"]["database"] = {
                "status": "unavailable",
                "error": DB_LAST_ERROR
            }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "error",
            "error": str(e)
        }
        health_status["overall_status"] = "degraded"
    
    # Vérification modèles
    model_manager = get_model_manager()
    models = model_manager.list_models()
    trained_models = sum(1 for m in models if m["info"]["is_trained"])
    
    health_status["components"]["ml_models"] = {
        "status": "healthy" if trained_models > 0 else "warning",
        "total_models": len(models),
        "trained_models": trained_models
    }
    
    # Vérification compliance
    health_status["components"]["compliance"] = {
        "status": "enabled" if COMPLIANCE_ENABLED else "disabled",
        "security_validator": get_security_validator() is not None,
        "accessibility_validator": get_accessibility_validator() is not None
    }
    
    # Métriques de performance (simulées)
    try:
        import psutil
        health_status["performance_metrics"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        # Recommandations basées sur les métriques
        if health_status["performance_metrics"]["cpu_percent"] > 80:
            health_status["recommendations"].append("High CPU usage detected")
        if health_status["performance_metrics"]["memory_percent"] > 80:
            health_status["recommendations"].append("High memory usage detected")
            
    except ImportError:
        health_status["performance_metrics"] = {"note": "psutil not available"}
    
    return health_status

# =====================================================================
# STARTUP EVENT
# =====================================================================

@app.on_event("startup")
async def startup_event():
    global DB_READY, DB_LAST_ERROR

    logger.info("Starting Enhanced Games AI/ML API v3.0.0...")

    try:
        if settings.db_configured:
            ensure_users_table()
            DB_READY = True
            DB_LAST_ERROR = None
            logger.info("Database ready")
        else:
            DB_READY = False
            DB_LAST_ERROR = "Database not configured"
            logger.warning(DB_LAST_ERROR)
    except Exception as e:
        DB_READY = False
        DB_LAST_ERROR = str(e)
        logger.error(f"Database initialization failed: {e}")

    # Initialisation des modèles par défaut
    model_manager = get_model_manager()
    
    # Créer modèle de recommandation par défaut
    try:
        if not model_manager.get_model("recommendation_default"):
            model_manager.create_model(
                ModelType.RECOMMENDATION, 
                "recommendation_default", 
                algorithm="hybrid"
            )
            logger.info("Default recommendation model created")
    except Exception as e:
        logger.warning(f"Could not create default recommendation model: {e}")

    logger.info("Enhanced Games AI/ML API startup complete!")
    logger.info("Available features: recommendations, classification, clustering, multi-model comparison")
    logger.info(f"Compliance enabled: {COMPLIANCE_ENABLED}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
