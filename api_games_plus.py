# api_enhanced_games.py - API intégrée avec modèle hybride et wishlist
from __future__ import annotations

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Imports Prometheus
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

# Imports existants
from settings import get_settings
from monitoring_metrics import get_monitor, prediction_latency, model_prediction_counter

# Nouveaux imports
from model_manager import get_hybrid_model, reset_hybrid_model, HybridRecommendationModel
from wishlist_manager import (
    WishlistManager, WishlistAddRequest, WishlistUpdateRequest, 
    WishlistResponse, NotificationResponse
)

logger = logging.getLogger("enhanced-games-api")
settings = get_settings()

# =========================
# NOUVEAUX MODELS PYDANTIC
# =========================

class HybridPredictionRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=50)
    min_confidence: float = Field(default=0.1, ge=0.0, le=1.0)
    algorithm: str = Field(default="hybrid_ensemble", 
                          pattern="^(hybrid_ensemble|content_only|collaborative_only|gradient_boosting_only)$")

class HybridTrainRequest(BaseModel):
    version: Optional[str] = None
    force_retrain: bool = False
    ensemble_weights: Optional[Dict[str, float]] = None

class PriceCheckRequest(BaseModel):
    check_all: bool = False
    user_id: Optional[int] = None

class ModelInfoResponse(BaseModel):
    model_version: str
    is_trained: bool
    components: Dict[str, Any]
    ensemble_weights: Dict[str, float]
    metrics: Dict[str, Any]

# =========================
# FASTAPI APP SETUP
# =========================

app = FastAPI(
    title="Enhanced Games API with Hybrid ML & Wishlist",
    version="3.0.0",
    description="API avec modèle hybride (Content+Collaborative+GradientBoosting) et système de wishlist avec notifications",
)

# Configuration Prometheus/metrics
Instrumentator().instrument(app).expose(app, include_in_schema=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in settings.ALLOW_ORIGINS.split(",") if o] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
DB_READY: bool = False
HYBRID_MODEL_READY: bool = False
PRICE_MONITORING_TASK: Optional[asyncio.Task] = None
wishlist_manager: Optional[WishlistManager] = None

# =========================
# ENDPOINTS RACINE ET HEALTH
# =========================

@app.get("/", include_in_schema=False)
def root():
    """Endpoint racine"""
    return {
        "name": app.title, 
        "version": app.version,
        "message": "Enhanced Games API with Hybrid ML & Wishlist - Ready",
        "status": "ok",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics"
    }

@app.head("/", include_in_schema=False)
def head_root():
    """Health probe HEAD pour Render"""
    return Response(status_code=200)

@app.get("/metrics", include_in_schema=False)
def get_metrics():
    """Endpoint Prometheus pour les métriques"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# =========================
# UTILITAIRES BASE DE DONNÉES
# =========================

def get_db_conn():
    """Fonction de connexion DB (reprise de l'API existante)"""
    import pymysql
    
    if not settings.db_configured:
        raise RuntimeError("Database not configured")
    
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

def fetch_games_for_ml(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Récupère les jeux pour l'entraînement ML"""
    
    sql = """
        SELECT game_id_rawg AS id, title, genres, rating, metacritic, platforms
        FROM games
        WHERE COALESCE(title,'') <> ''
        ORDER BY rating DESC
    """
    
    if limit and limit > 0:
        sql += " LIMIT %s"
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,) if limit else None)
                rows = cur.fetchall() or []
    except Exception as e:
        logger.warning(f"Database error, using fallback data: {e}")
        # Données de fallback
        return [
            {"id": 1, "title": "The Witcher 3", "genres": "RPG Action", "rating": 4.9, "metacritic": 93, "platforms": ["PC", "PS4"]},
            {"id": 2, "title": "Hades", "genres": "Action Roguelike", "rating": 4.8, "metacritic": 93, "platforms": ["PC", "Switch"]},
            {"id": 3, "title": "Stardew Valley", "genres": "Simulation RPG", "rating": 4.7, "metacritic": 89, "platforms": ["PC", "Switch"]},
            {"id": 4, "title": "Celeste", "genres": "Platformer Indie", "rating": 4.6, "metacritic": 94, "platforms": ["PC", "Switch"]},
            {"id": 5, "title": "Doom Eternal", "genres": "Action FPS", "rating": 4.5, "metacritic": 88, "platforms": ["PC", "PS4"]},
            {"id": 6, "title": "Cyberpunk 2077", "genres": "RPG Action", "rating": 4.0, "metacritic": 76, "platforms": ["PC", "PS4", "Xbox"]},
            {"id": 7, "title": "Hollow Knight", "genres": "Metroidvania Indie", "rating": 4.7, "metacritic": 90, "platforms": ["PC", "Switch"]},
        ]
    
    games = []
    for r in rows:
        # Parse platforms si c'est une chaîne
        platforms = r.get("platforms", "")
        if isinstance(platforms, str):
            platforms = [p.strip() for p in platforms.split(",") if p.strip()]
        elif not isinstance(platforms, list):
            platforms = ["PC"]
        
        games.append({
            "id": int(r["id"]),
            "title": r.get("title") or "",
            "genres": r.get("genres") or "",
            "rating": float(r.get("rating") or 0.0),
            "metacritic": int(r.get("metacritic") or 0),
            "platforms": platforms,
        })
    
    logger.info(f"Loaded {len(games)} games from database")
    return games

# =========================
# AUTH HELPERS (repris de l'API existante)
# =========================

from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    """Vérifie le token JWT"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_user_id_from_username(username: str) -> int:
    """Récupère l'ID utilisateur à partir du nom d'utilisateur"""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE username = %s", (username,))
                row = cur.fetchone()
                if row:
                    return int(row["id"])
    except Exception as e:
        logger.error(f"Error getting user ID: {e}")
    
    # Fallback: utiliser un hash du username
    return abs(hash(username)) % 1000000

# =========================
# ENDPOINTS MODÈLE HYBRIDE
# =========================

@app.post("/recommend/hybrid", tags=["hybrid-model"])
def recommend_hybrid(request: HybridPredictionRequest, username: str = Depends(verify_token)):
    """Recommandations avec le modèle hybride"""
    
    if not HYBRID_MODEL_READY:
        raise HTTPException(status_code=503, detail="Hybrid model not ready")
    
    hybrid_model = get_hybrid_model()
    
    if not hybrid_model.is_trained:
        # Auto-entraînement si nécessaire
        logger.info("Auto-training hybrid model...")
        games_data = fetch_games_for_ml()
        hybrid_model.train(games_data)
    
    start_time = datetime.utcnow()
    
    try:
        # Selon l'algorithme demandé
        if request.algorithm == "content_only":
            recommendations = hybrid_model._predict_content_based_only(request.query, request.k)
        elif request.algorithm == "collaborative_only":
            recommendations = hybrid_model._predict_collaborative_only(request.k)
        elif request.algorithm == "gradient_boosting_only":
            recommendations = hybrid_model._predict_gradient_boosting_only(request.k)
        else:  # hybrid_ensemble
            recommendations = hybrid_model.predict(request.query, request.k, request.min_confidence)
        
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Enregistrer les métriques
        get_monitor().record_prediction("recommend_hybrid", request.query, recommendations, latency_ms / 1000)
        
        return {
            "query": request.query,
            "algorithm": request.algorithm,
            "recommendations": recommendations,
            "latency_ms": latency_ms,
            "model_version": hybrid_model.model_version,
            "total_results": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Hybrid prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/model/train-hybrid", tags=["hybrid-model"])
def train_hybrid_model(request: HybridTrainRequest, username: str = Depends(verify_token)):
    """Entraîne le modèle hybride"""
    
    hybrid_model = get_hybrid_model()
    
    # Mettre à jour les poids d'ensemble si fournis
    if request.ensemble_weights:
        # Validation des poids
        total_weight = sum(request.ensemble_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Ensemble weights must sum to 1.0")
        
        hybrid_model.ensemble_weights.update(request.ensemble_weights)
    
    # Définir la version
    if request.version:
        hybrid_model.model_version = request.version
    
    # Récupérer les données d'entraînement
    games_data = fetch_games_for_ml()
    
    if len(games_data) < 5:
        raise HTTPException(status_code=400, detail="Insufficient training data")
    
    try:
        result = hybrid_model.train(games_data)
        
        # Sauvegarder le modèle
        try:
            model_path = f"model/hybrid_model_{hybrid_model.model_version}.pkl"
            hybrid_model.save_model(model_path)
            result["model_saved"] = True
            result["model_path"] = model_path
        except Exception as e:
            logger.warning(f"Could not save hybrid model: {e}")
            result["model_saved"] = False
        
        global HYBRID_MODEL_READY
        HYBRID_MODEL_READY = True
        
        return result
        
    except Exception as e:
        logger.error(f"Hybrid training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/model/hybrid/info", tags=["hybrid-model"], response_model=ModelInfoResponse)
def get_hybrid_model_info(username: str = Depends(verify_token)):
    """Informations détaillées sur le modèle hybride"""
    
    hybrid_model = get_hybrid_model()
    info = hybrid_model.get_model_info()
    
    return ModelInfoResponse(**info)

# Ajouter des méthodes simplifiées au modèle hybride pour les algorithmes séparés
def add_single_algorithm_methods():
    """Ajoute des méthodes pour les algorithmes individuels"""
    
    def _predict_content_based_only(self, query: str, k: int) -> List[Dict]:
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        content_scores = self._predict_content_based(query, k * 2)
        
        results = []
        for game_id, score in sorted(content_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            game_row = self.games_df[self.games_df["id"] == game_id].iloc[0]
            results.append({
                "id": int(game_id),
                "title": game_row["title"],
                "genres": game_row["genres"],
                "confidence": float(score),
                "rating": float(game_row["rating"]),
                "metacritic": int(game_row["metacritic"]),
                "algorithm": "content_only"
            })
        
        return results
    
    def _predict_collaborative_only(self, k: int) -> List[Dict]:
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        collab_scores = self._predict_collaborative(k * 2)
        
        results = []
        for game_id, score in sorted(collab_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            game_row = self.games_df[self.games_df["id"] == game_id].iloc[0]
            results.append({
                "id": int(game_id),
                "title": game_row["title"],
                "genres": game_row["genres"],
                "confidence": float(score),
                "rating": float(game_row["rating"]),
                "metacritic": int(game_row["metacritic"]),
                "algorithm": "collaborative_only"
            })
        
        return results
    
    def _predict_gradient_boosting_only(self, k: int) -> List[Dict]:
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        gb_scores = self._predict_gradient_boosting()
        
        results = []
        for game_id, score in sorted(gb_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            game_row = self.games_df[self.games_df["id"] == game_id].iloc[0]
            results.append({
                "id": int(game_id),
                "title": game_row["title"],
                "genres": game_row["genres"],
                "confidence": float(score),
                "rating": float(game_row["rating"]),
                "metacritic": int(game_row["metacritic"]),
                "algorithm": "gradient_boosting_only"
            })
        
        return results
    
    # Ajouter les méthodes à la classe
    HybridRecommendationModel._predict_content_based_only = _predict_content_based_only
    HybridRecommendationModel._predict_collaborative_only = _predict_collaborative_only
    HybridRecommendationModel._predict_gradient_boosting_only = _predict_gradient_boosting_only

# =========================
# ENDPOINTS WISHLIST
# =========================

@app.post("/wishlist", tags=["wishlist"], status_code=201)
def add_to_wishlist(request: WishlistAddRequest, username: str = Depends(verify_token)):
    """Ajoute un jeu à la wishlist"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    user_id = get_user_id_from_username(username)
    
    try:
        wishlist_item = wishlist_manager.add_to_wishlist(user_id, request)
        
        return {
            "id": wishlist_item.id,
            "game_title": wishlist_item.game_title,
            "target_price": wishlist_item.target_price,
            "current_price": wishlist_item.current_price,
            "price_currency": wishlist_item.price_currency,
            "created_at": wishlist_item.created_at.isoformat() if wishlist_item.created_at else None,
            "is_active": wishlist_item.is_active,
            "message": "Game added to wishlist successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding to wishlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to add to wishlist")

@app.get("/wishlist", tags=["wishlist"])
def get_user_wishlist(username: str = Depends(verify_token), active_only: bool = True) -> List[WishlistResponse]:
    """Récupère la wishlist de l'utilisateur"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    user_id = get_user_id_from_username(username)
    
    try:
        wishlist_items = wishlist_manager.get_user_wishlist(user_id, active_only)
        
        response_items = []
        for item in wishlist_items:
            # Calculer si une alerte est déclenchée
            alert_triggered = (
                item.current_price is not None and 
                item.current_price <= item.target_price
            )
            
            # Calculer la différence de prix
            price_difference = None
            if item.current_price:
                price_difference = item.current_price - item.target_price
            
            response_items.append(WishlistResponse(
                id=item.id,
                game_title=item.game_title,
                target_price=item.target_price,
                current_price=item.current_price,
                price_currency=item.price_currency,
                created_at=item.created_at.isoformat() if item.created_at else "",
                is_active=item.is_active,
                notification_sent=item.notification_sent,
                price_difference=price_difference,
                alert_triggered=alert_triggered
            ))
        
        return response_items
        
    except Exception as e:
        logger.error(f"Error getting wishlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to get wishlist")

@app.put("/wishlist/{wishlist_id}", tags=["wishlist"])
def update_wishlist_item(wishlist_id: int, request: WishlistUpdateRequest, username: str = Depends(verify_token)):
    """Met à jour un item de wishlist"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    user_id = get_user_id_from_username(username)
    
    try:
        updated_item = wishlist_manager.update_wishlist_item(wishlist_id, user_id, request)
        
        return {
            "id": updated_item.id,
            "game_title": updated_item.game_title,
            "target_price": updated_item.target_price,
            "current_price": updated_item.current_price,
            "is_active": updated_item.is_active,
            "updated_at": updated_item.updated_at.isoformat() if updated_item.updated_at else None,
            "message": "Wishlist item updated successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating wishlist item: {e}")
        raise HTTPException(status_code=500, detail="Failed to update wishlist item")

@app.delete("/wishlist/{wishlist_id}", tags=["wishlist"])
def remove_from_wishlist(wishlist_id: int, username: str = Depends(verify_token)):
    """Supprime un item de la wishlist"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    user_id = get_user_id_from_username(username)
    
    try:
        success = wishlist_manager.remove_from_wishlist(wishlist_id, user_id)
        
        if success:
            return {"message": "Item removed from wishlist successfully"}
        else:
            raise HTTPException(status_code=404, detail="Wishlist item not found")
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error removing from wishlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove from wishlist")

@app.post("/wishlist/check-prices", tags=["wishlist"])
def check_wishlist_prices(username: str = Depends(verify_token)):
    """Vérifie manuellement les prix de la wishlist de l'utilisateur"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    try:
        alerts_created = wishlist_manager.check_price_alerts()
        
        return {
            "message": "Price check completed",
            "alerts_created": alerts_created,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking prices: {e}")
        raise HTTPException(status_code=500, detail="Failed to check prices")

# =========================
# ENDPOINTS NOTIFICATIONS
# =========================

@app.get("/wishlist/notifications", tags=["notifications"])
def get_user_notifications(username: str = Depends(verify_token), limit: int = 20) -> List[NotificationResponse]:
    """Récupère les notifications de l'utilisateur"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    user_id = get_user_id_from_username(username)
    
    try:
        notifications = wishlist_manager.get_user_notifications(user_id, limit)
        
        response_notifications = []
        for notif in notifications:
            response_notifications.append(NotificationResponse(
                id=notif["id"],
                wishlist_item_id=notif["wishlist_item_id"],
                message=notif["message"],
                notification_type=notif["notification_type"],
                created_at=notif["created_at"],
                is_read=notif["is_read"]
            ))
        
        return response_notifications
        
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to get notifications")

@app.put("/wishlist/notifications/{notification_id}/read", tags=["notifications"])
def mark_notification_read(notification_id: int, username: str = Depends(verify_token)):
    """Marque une notification comme lue"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    user_id = get_user_id_from_username(username)
    
    try:
        success = wishlist_manager.mark_notification_read(notification_id, user_id)
        
        if success:
            return {"message": "Notification marked as read"}
        else:
            raise HTTPException(status_code=404, detail="Notification not found")
        
    except Exception as e:
        logger.error(f"Error marking notification as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")

@app.get("/wishlist/notifications/count", tags=["notifications"])
def get_notification_count(username: str = Depends(verify_token), unread_only: bool = True):
    """Compte les notifications de l'utilisateur"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    user_id = get_user_id_from_username(username)
    
    try:
        count = wishlist_manager.get_notification_count(user_id, unread_only)
        
        return {
            "count": count,
            "unread_only": unread_only,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error getting notification count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get notification count")

# =========================
# ENDPOINTS ADMIN
# =========================

@app.post("/admin/check-all-prices", tags=["admin"])
def admin_check_all_prices(username: str = Depends(verify_token)):
    """Vérifie tous les prix de toutes les wishlists (admin)"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    try:
        alerts_created = wishlist_manager.check_price_alerts()
        
        return {
            "message": "Mass price check completed",
            "alerts_created": alerts_created,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in mass price check: {e}")
        raise HTTPException(status_code=500, detail="Failed to check all prices")

@app.post("/admin/cleanup-notifications", tags=["admin"])
def admin_cleanup_notifications(days_old: int = 30, username: str = Depends(verify_token)):
    """Nettoie les anciennes notifications (admin)"""
    
    if not wishlist_manager:
        raise HTTPException(status_code=503, detail="Wishlist service not available")
    
    try:
        deleted_count = wishlist_manager.cleanup_old_notifications(days_old)
        
        return {
            "message": "Notification cleanup completed",
            "deleted": deleted_count,
            "days_old": days_old,
            "cleaned_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up notifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup notifications")

# =========================
# HEALTH & MONITORING
# =========================

@app.get("/healthz", tags=["system"])
def enhanced_healthz():
    """Health check amélioré"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": DB_READY,
            "hybrid_model": HYBRID_MODEL_READY,
            "wishlist_service": wishlist_manager is not None,
            "price_monitoring": PRICE_MONITORING_TASK is not None and not PRICE_MONITORING_TASK.done()
        }
    }
    
    # Vérifier le statut du modèle hybride
    if HYBRID_MODEL_READY:
        hybrid_model = get_hybrid_model()
        health_status["model_info"] = {
            "version": hybrid_model.model_version,
            "is_trained": hybrid_model.is_trained,
            "total_predictions": hybrid_model.prediction_count
        }
    
    # Vérifier le statut global
    if not all(health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return health_status

# =========================
# STARTUP EVENT
# =========================

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global DB_READY, HYBRID_MODEL_READY, PRICE_MONITORING_TASK, wishlist_manager
    
    logger.info("Starting Enhanced Games API with Hybrid ML & Wishlist...")
    
    # 1. Initialiser la base de données
    try:
        if settings.db_configured:
            with get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT VERSION()")
                    version = cur.fetchone()
                    logger.info(f"Database connected: {version['VERSION()']}")
            
            DB_READY = True
            
            # Initialiser le gestionnaire de wishlist
            wishlist_manager = WishlistManager(get_db_conn)
            wishlist_manager.ensure_wishlist_tables()
            logger.info("Wishlist service initialized")
            
        else:
            logger.warning("Database not configured - running in demo mode")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        if settings.DB_REQUIRED:
            raise
    
    # 2. Ajouter les méthodes au modèle hybride
    add_single_algorithm_methods()
    
    # 3. Charger le modèle hybride existant
    try:
        hybrid_model = get_hybrid_model()
        model_path = "model/hybrid_model_latest.pkl"
        
        if os.path.exists(model_path):
            if hybrid_model.load_model(model_path):
                logger.info(f"Hybrid model loaded: {hybrid_model.model_version}")
                HYBRID_MODEL_READY = True
            else:
                logger.info("Failed to load hybrid model - will train on first request")
        else:
            logger.info("No hybrid model found - will train on first request")
            
    except Exception as e:
        logger.warning(f"Hybrid model initialization error: {e}")
    
    # 4. Démarrer la surveillance des prix en arrière-plan
    if wishlist_manager:
        try:
            PRICE_MONITORING_TASK = asyncio.create_task(wishlist_manager.start_price_monitoring())
            logger.info("Price monitoring task started")
        except Exception as e:
            logger.error(f"Failed to start price monitoring: {e}")
    
    logger.info("Enhanced Games API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt"""
    global PRICE_MONITORING_TASK
    
    logger.info("Shutting down Enhanced Games API...")
    
    # Arrêter la surveillance des prix
    if PRICE_MONITORING_TASK and not PRICE_MONITORING_TASK.done():
        PRICE_MONITORING_TASK.cancel()
        try:
            await PRICE_MONITORING_TASK
        except asyncio.CancelledError:
            pass
        logger.info("Price monitoring task stopped")
    
    # Sauvegarder le modèle hybride
    try:
        if HYBRID_MODEL_READY:
            hybrid_model = get_hybrid_model()
            hybrid_model.save_model("model/hybrid_model_latest.pkl")
            logger.info("Hybrid model saved")
    except Exception as e:
        logger.warning(f"Failed to save hybrid model: {e}")
    
    logger.info("Enhanced Games API shutdown completed")

# Import des endpoints existants (auth, etc.)
# Vous devrez ajouter ici les endpoints d'authentification de votre API existante

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
