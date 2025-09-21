# api_games_complete_adapted.py - API adaptée à votre structure BDD
from __future__ import annotations

import os
import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from passlib.context import CryptContext

# Imports Prometheus
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

# Imports existants (adaptez selon vos modules)
try:
    from settings import get_settings
    from monitoring_metrics import get_monitor
except ImportError:
    # Fallback si les modules n'existent pas
    class MockSettings:
        db_configured = True
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = int(os.getenv("DB_PORT", "3306"))
        DB_USER = os.getenv("DB_USER", "root")
        DB_PASSWORD = os.getenv("DB_PASSWORD", "")
        DB_NAME = os.getenv("DB_NAME", "anne_games_db")
        ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
        SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        ALGORITHM = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES = 1440
        DB_REQUIRED = True
    
    def get_settings():
        return MockSettings()
    
    class MockMonitor:
        def record_prediction(self, *args, **kwargs):
            pass
    
    def get_monitor():
        return MockMonitor()

logger = logging.getLogger("games-api")
settings = get_settings()

# =========================
# CONFIGURATION AUTH
# =========================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =========================
# MODELS PYDANTIC
# =========================

# Models d'authentification
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class UserResponse(BaseModel):
    id: int
    username: str
    created_at: Optional[str] = None
    is_active: bool = True

# Models pour les jeux
class GameResponse(BaseModel):
    game_id_rawg: int
    title: str
    release_date: Optional[str] = None
    genres: Optional[str] = None
    platforms: Optional[str] = None
    rating: Optional[float] = None
    metacritic: Optional[int] = None
    last_update: Optional[str] = None

class GameWithPriceResponse(GameResponse):
    best_price_PC: Optional[str] = None
    best_shop_PC: Optional[str] = None
    site_url_PC: Optional[str] = None
    similarity_score: Optional[float] = None

# Models pour les prix
class PriceInfo(BaseModel):
    title: str
    best_price_PC: Optional[str] = None
    best_shop_PC: Optional[str] = None
    site_url_PC: Optional[str] = None
    game_id_rawg: Optional[int] = None
    similarity_score: Optional[float] = None
    last_update: Optional[str] = None

# Models pour les recommandations
class RecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=50)
    min_rating: float = Field(default=0.0, ge=0.0, le=5.0)
    genres: Optional[List[str]] = None
    platforms: Optional[List[str]] = None

class RecommendationResponse(BaseModel):
    games: List[GameWithPriceResponse]
    query: str
    total_results: int
    filters_applied: Dict[str, Any]

# Models pour les interactions utilisateur
class UserInteraction(BaseModel):
    user_id: int
    game_id_rawg: int
    interaction_type: str
    rating: Optional[float] = None

# Models pour la wishlist
class WishlistItem(BaseModel):
    game_id_rawg: int
    target_price: Optional[float] = None
    currency: str = Field(default="EUR")
    created_at: Optional[str] = None

class WishlistResponse(BaseModel):
    id: int
    game_id_rawg: int
    title: str
    target_price: Optional[float] = None
    current_price: Optional[str] = None
    price_difference: Optional[float] = None
    alert_triggered: bool = False
    created_at: Optional[str] = None

# Models pour les statistiques
class StatsResponse(BaseModel):
    total_games: int
    total_games_with_prices: int
    avg_similarity_score: Optional[float] = None
    high_quality_matches: int
    last_extraction: Optional[str] = None

# =========================
# FASTAPI APP SETUP
# =========================

app = FastAPI(
    title="Anne's Games API - Complete Edition",
    version="4.0.0",
    description="API complète pour les jeux avec authentification, recommandations, prix et wishlist",
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

# =========================
# ENDPOINTS RACINE ET HEALTH
# =========================

@app.get("/", include_in_schema=False)
def root():
    """Endpoint racine"""
    return {
        "name": app.title, 
        "version": app.version,
        "message": "Anne's Games API - Complete Edition - Ready",
        "status": "ok",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics",
        "endpoints": {
            "auth": ["/register", "/token", "/me"],
            "games": ["/games", "/games/search", "/games/{game_id}"],
            "prices": ["/prices", "/prices/search"],
            "recommendations": ["/recommend"],
            "wishlist": ["/wishlist"],
            "stats": ["/stats"]
        }
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
    """Fonction de connexion DB"""
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

# =========================
# FONCTIONS D'AUTHENTIFICATION
# =========================

def hash_password(password: str) -> str:
    """Hache un mot de passe"""
    return pwd_ctx.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe"""
    return pwd_ctx.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crée un token JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def get_user_from_db(username: str) -> Optional[Dict]:
    """Récupère un utilisateur de la base de données"""
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, username, 
                       COALESCE(hashed_password, password_hash) as password_hash
                       FROM users WHERE username = %s""",
                    (username,)
                )
                return cur.fetchone()
    except Exception as e:
        logger.error(f"Error getting user from database: {e}")
        return None

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authentifie un utilisateur"""
    user = get_user_from_db(username)
    if not user:
        return None
    
    if not verify_password(password, user["password_hash"]):
        return None
    
    return user

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
# ENDPOINTS D'AUTHENTIFICATION
# =========================

@app.post("/register", tags=["auth"], response_model=UserResponse, status_code=201)
def register_user(user_data: UserCreate):
    """Inscription d'un nouvel utilisateur"""
    
    try:
        # Vérifier si l'utilisateur existe déjà
        existing_user = get_user_from_db(user_data.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Hacher le mot de passe
        hashed_password = hash_password(user_data.password)
        
        # Créer l'utilisateur
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO users (username, hashed_password) VALUES (%s, %s)""",
                    (user_data.username, hashed_password)
                )
                
                # Récupérer l'utilisateur créé
                new_user = get_user_from_db(user_data.username)
        
        return UserResponse(
            id=new_user["id"],
            username=new_user["username"],
            is_active=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/token", tags=["auth"], response_model=Token)
def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """Connexion utilisateur (obtenir un token JWT)"""
    
    # Authentifier l'utilisateur
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Créer le token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.get("/me", tags=["auth"], response_model=UserResponse)
def get_current_user(username: str = Depends(verify_token)):
    """Récupère les informations de l'utilisateur connecté"""
    
    user = get_user_from_db(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user["id"],
        username=user["username"],
        is_active=True
    )

# =========================
# ENDPOINTS JEUX
# =========================

@app.get("/games", tags=["games"], response_model=List[GameResponse])
def get_games(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_rating: Optional[float] = Query(default=None, ge=0.0, le=5.0),
    genre: Optional[str] = Query(default=None),
    platform: Optional[str] = Query(default=None)
):
    """Récupère la liste des jeux avec filtres optionnels"""
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Construire la requête avec filtres
                sql = "SELECT * FROM games WHERE 1=1"
                params = []
                
                if min_rating is not None:
                    sql += " AND rating >= %s"
                    params.append(min_rating)
                
                if genre:
                    sql += " AND genres LIKE %s"
                    params.append(f"%{genre}%")
                
                if platform:
                    sql += " AND platforms LIKE %s"
                    params.append(f"%{platform}%")
                
                sql += " ORDER BY rating DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cur.execute(sql, params)
                games = cur.fetchall()
        
        return [
            GameResponse(
                game_id_rawg=game["game_id_rawg"],
                title=game["title"],
                release_date=game["release_date"].isoformat() if game["release_date"] else None,
                genres=game.get("genres"),
                platforms=game.get("platforms"),
                rating=game.get("rating"),
                metacritic=game.get("metacritic"),
                last_update=game["last_update"].isoformat() if game.get("last_update") else None
            )
            for game in games
        ]
        
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch games")

@app.get("/games/{game_id}", tags=["games"], response_model=GameWithPriceResponse)
def get_game_by_id(game_id: int):
    """Récupère un jeu spécifique avec son prix"""
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Récupérer le jeu avec son prix
                cur.execute("""
                    SELECT g.*, 
                           p.best_price_PC, p.best_shop_PC, p.site_url_PC,
                           p.similarity_score
                    FROM games g
                    LEFT JOIN best_price_pc p ON g.title = p.title
                    WHERE g.game_id_rawg = %s
                """, (game_id,))
                
                game = cur.fetchone()
                
                if not game:
                    raise HTTPException(status_code=404, detail="Game not found")
        
        return GameWithPriceResponse(
            game_id_rawg=game["game_id_rawg"],
            title=game["title"],
            release_date=game["release_date"].isoformat() if game["release_date"] else None,
            genres=game.get("genres"),
            platforms=game.get("platforms"),
            rating=game.get("rating"),
            metacritic=game.get("metacritic"),
            last_update=game["last_update"].isoformat() if game.get("last_update") else None,
            best_price_PC=game.get("best_price_PC"),
            best_shop_PC=game.get("best_shop_PC"),
            site_url_PC=game.get("site_url_PC"),
            similarity_score=float(game["similarity_score"]) if game.get("similarity_score") else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching game {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch game")

@app.get("/games/search", tags=["games"], response_model=List[GameWithPriceResponse])
def search_games(
    q: str = Query(..., min_length=2),
    limit: int = Query(default=20, ge=1, le=100)
):
    """Recherche de jeux par titre"""
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT g.*, 
                           p.best_price_PC, p.best_shop_PC, p.site_url_PC,
                           p.similarity_score
                    FROM games g
                    LEFT JOIN best_price_pc p ON g.title = p.title
                    WHERE g.title LIKE %s
                    ORDER BY g.rating DESC
                    LIMIT %s
                """, (f"%{q}%", limit))
                
                games = cur.fetchall()
        
        return [
            GameWithPriceResponse(
                game_id_rawg=game["game_id_rawg"],
                title=game["title"],
                release_date=game["release_date"].isoformat() if game["release_date"] else None,
                genres=game.get("genres"),
                platforms=game.get("platforms"),
                rating=game.get("rating"),
                metacritic=game.get("metacritic"),
                last_update=game["last_update"].isoformat() if game.get("last_update") else None,
                best_price_PC=game.get("best_price_PC"),
                best_shop_PC=game.get("best_shop_PC"),
                site_url_PC=game.get("site_url_PC"),
                similarity_score=float(game["similarity_score"]) if game.get("similarity_score") else None
            )
            for game in games
        ]
        
    except Exception as e:
        logger.error(f"Error searching games: {e}")
        raise HTTPException(status_code=500, detail="Failed to search games")

# =========================
# ENDPOINTS PRIX
# =========================

@app.get("/prices", tags=["prices"], response_model=List[PriceInfo])
def get_prices(
    limit: int = Query(default=50, ge=1, le=500),
    min_similarity: Optional[float] = Query(default=0.8, ge=0.0, le=1.0),
    shop: Optional[str] = Query(default=None)
):
    """Récupère les informations de prix"""
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                sql = "SELECT * FROM best_price_pc WHERE 1=1"
                params = []
                
                if min_similarity is not None:
                    sql += " AND similarity_score >= %s"
                    params.append(min_similarity)
                
                if shop:
                    sql += " AND best_shop_PC LIKE %s"
                    params.append(f"%{shop}%")
                
                sql += " ORDER BY similarity_score DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(sql, params)
                prices = cur.fetchall()
        
        return [
            PriceInfo(
                title=price["title"],
                best_price_PC=price.get("best_price_PC"),
                best_shop_PC=price.get("best_shop_PC"),
                site_url_PC=price.get("site_url_PC"),
                game_id_rawg=price.get("game_id_rawg"),
                similarity_score=float(price["similarity_score"]) if price.get("similarity_score") else None,
                last_update=price["last_update"].isoformat() if price.get("last_update") else None
            )
            for price in prices
        ]
        
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch prices")

@app.get("/prices/search", tags=["prices"], response_model=List[PriceInfo])
def search_prices(
    q: str = Query(..., min_length=2),
    limit: int = Query(default=20, ge=1, le=100)
):
    """Recherche de prix par titre de jeu"""
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM best_price_pc 
                    WHERE title LIKE %s 
                    ORDER BY similarity_score DESC 
                    LIMIT %s
                """, (f"%{q}%", limit))
                
                prices = cur.fetchall()
        
        return [
            PriceInfo(
                title=price["title"],
                best_price_PC=price.get("best_price_PC"),
                best_shop_PC=price.get("best_shop_PC"),
                site_url_PC=price.get("site_url_PC"),
                game_id_rawg=price.get("game_id_rawg"),
                similarity_score=float(price["similarity_score"]) if price.get("similarity_score") else None,
                last_update=price["last_update"].isoformat() if price.get("last_update") else None
            )
            for price in prices
        ]
        
    except Exception as e:
        logger.error(f"Error searching prices: {e}")
        raise HTTPException(status_code=500, detail="Failed to search prices")

# =========================
# ENDPOINTS RECOMMANDATIONS
# =========================

from fastapi import HTTPException, Depends
from typing import List

@app.post(
    "/recommend",
    tags=["recommendations"],
    response_model=RecommendationResponse
)
def get_recommendations(
    request: RecommendationRequest,
    username: str = Depends(verify_token)
):
    """Système de recommandation basé sur les données utilisateur"""
    try:
        user_id = get_user_id_from_username(username)
    except Exception as e:
        logger.error(f"Error resolving user id from username: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve user")

    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Récupérer les interactions de l'utilisateur pour personnaliser
                cur.execute(
                    """
                    SELECT game_id_rawg, interaction_type, rating
                    FROM user_interactions
                    WHERE user_id = %s
                    """,
                    (user_id,)
                )
                user_interactions = cur.fetchall()

                # Construire la requête de recommandation
                sql = """
                    SELECT
                        g.*,
                        p.best_price_PC,
                        p.best_shop_PC,
                        p.site_url_PC,
                        p.similarity_score
                    FROM games g
                    LEFT JOIN best_price_pc p ON g.title = p.title
                    WHERE 1=1
                """
                params: List[object] = []

                # Filtrer par rating minimum
                if getattr(request, "min_rating", 0) and request.min_rating > 0:
                    sql += " AND g.rating >= %s"
                    params.append(request.min_rating)

                # Filtrer par genres (champ texte contenant une liste/chaîne de genres)
                if getattr(request, "genres", None):
                    genre_conditions = []
                    for genre in request.genres:
                        genre_conditions.append("g.genres ILIKE %s")
                        params.append(f"%{genre}%")
                    # Regrouper les conditions par genre avec OR
                    sql += f" AND ({' OR '.join(genre_conditions)})"

                # (Optionnel) tri par score de similarité si présent
                sql += " ORDER BY COALESCE(p.similarity_score, 0) DESC, g.rating DESC"

                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

                # Mapper rows -> objets du response_model (à adapter à ton schéma)
                recommendations = map_rows_to_recommendations(rows, user_interactions)

                return RecommendationResponse(
                    user_id=user_id,
                    count=len(recommendations),
                    items=recommendations
                )

    except HTTPException:
        # déjà levée plus haut
        raise
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recommendations")

# =========================
# ENDPOINTS WISHLIST
# =========================

@app.post("/wishlist", tags=["wishlist"], status_code=201)
def add_to_wishlist(item: WishlistItem, username: str = Depends(verify_token)):
    """Ajoute un jeu à la wishlist"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Vérifier si le jeu existe
                cur.execute("SELECT title FROM games WHERE game_id_rawg = %s", (item.game_id_rawg,))
                game = cur.fetchone()
                if not game:
                    raise HTTPException(status_code=404, detail="Game not found")
                
                # Créer la table wishlist si elle n'existe pas
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS wishlist (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT NOT NULL,
                        game_id_rawg INT NOT NULL,
                        target_price DECIMAL(10,2),
                        currency VARCHAR(3) DEFAULT 'EUR',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE,
                        UNIQUE KEY unique_user_game (user_id, game_id_rawg),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        FOREIGN KEY (game_id_rawg) REFERENCES games(game_id_rawg) ON DELETE CASCADE
                    )
                """)
                
                # Ajouter à la wishlist
                cur.execute("""
                    INSERT INTO wishlist (user_id, game_id_rawg, target_price, currency)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    target_price = VALUES(target_price),
                    currency = VALUES(currency),
                    is_active = TRUE
                """, (user_id, item.game_id_rawg, item.target_price, item.currency))
                
                wishlist_id = cur.lastrowid or cur.execute("SELECT id FROM wishlist WHERE user_id = %s AND game_id_rawg = %s", (user_id, item.game_id_rawg)) or cur.fetchone()["id"]
        
        return {
            "id": wishlist_id,
            "message": "Game added to wishlist successfully",
            "game_title": game["title"],
            "target_price": item.target_price,
            "currency": item.currency
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding to wishlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to add to wishlist")

@app.get("/wishlist", tags=["wishlist"], response_model=List[WishlistResponse])
def get_user_wishlist(username: str = Depends(verify_token), active_only: bool = True):
    """Récupère la wishlist de l'utilisateur"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    SELECT w.*, g.title, p.best_price_PC
                    FROM wishlist w
                    JOIN games g ON w.game_id_rawg = g.game_id_rawg
                    LEFT JOIN best_price_pc p ON g.title = p.title
                    WHERE w.user_id = %s
                """
                params = [user_id]
                
                if active_only:
                    sql += " AND w.is_active = TRUE"
                
                sql += " ORDER BY w.created_at DESC"
                
                cur.execute(sql, params)
                wishlist_items = cur.fetchall()
        
        def parse_price(price_str):
            """Parse price string to float"""
            if not price_str:
                return None
            # Remove currency symbols and convert to float
            price_clean = re.sub(r'[€$£]', '', price_str).replace(',', '.')
            try:
                return float(price_clean)
            except:
                return None
        
        results = []
        for item in wishlist_items:
            current_price_str = item.get("best_price_PC")
            current_price = parse_price(current_price_str)
            
            # Calculate price difference and alert
            price_difference = None
            alert_triggered = False
            
            if current_price and item.get("target_price"):
                price_difference = current_price - float(item["target_price"])
                alert_triggered = current_price <= float(item["target_price"])
            
            results.append(WishlistResponse(
                id=item["id"],
                game_id_rawg=item["game_id_rawg"],
                title=item["title"],
                target_price=float(item["target_price"]) if item.get("target_price") else None,
                current_price=current_price_str,
                price_difference=price_difference,
                alert_triggered=alert_triggered,
                created_at=item["created_at"].isoformat() if item.get("created_at") else None
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching wishlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch wishlist")

@app.delete("/wishlist/{wishlist_id}", tags=["wishlist"])
def remove_from_wishlist(wishlist_id: int, username: str = Depends(verify_token)):
    """Supprime un item de la wishlist"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Vérifier que l'item appartient à l'utilisateur
                cur.execute("""
                    DELETE FROM wishlist 
                    WHERE id = %s AND user_id = %s
                """, (wishlist_id, user_id))
                
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Wishlist item not found")
        
        return {"message": "Item removed from wishlist successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing from wishlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove from wishlist")

@app.get("/wishlist/alerts", tags=["wishlist"])
def get_price_alerts(username: str = Depends(verify_token)):
    """Récupère les alertes de prix pour l'utilisateur"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT w.*, g.title, p.best_price_PC, p.best_shop_PC, p.site_url_PC
                    FROM wishlist w
                    JOIN games g ON w.game_id_rawg = g.game_id_rawg
                    JOIN best_price_pc p ON g.title = p.title
                    WHERE w.user_id = %s AND w.is_active = TRUE
                    AND w.target_price IS NOT NULL
                """, (user_id,))
                
                wishlist_items = cur.fetchall()
        
        alerts = []
        for item in wishlist_items:
            if item.get("best_price_PC"):
                # Parse current price
                current_price_str = item["best_price_PC"]
                price_clean = re.sub(r'[€$£]', '', current_price_str).replace(',', '.')
                try:
                    current_price = float(price_clean)
                    target_price = float(item["target_price"])
                    
                    if current_price <= target_price:
                        alerts.append({
                            "wishlist_id": item["id"],
                            "game_id_rawg": item["game_id_rawg"],
                            "title": item["title"],
                            "current_price": current_price_str,
                            "target_price": target_price,
                            "savings": target_price - current_price,
                            "shop": item.get("best_shop_PC"),
                            "url": item.get("site_url_PC"),
                            "alert_triggered_at": datetime.utcnow().isoformat()
                        })
                except ValueError:
                    continue
        
        return {
            "user_id": user_id,
            "alerts": alerts,
            "total_alerts": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Error fetching price alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch price alerts")

# =========================
# ENDPOINTS STATISTIQUES
# =========================

@app.get("/stats", tags=["statistics"], response_model=StatsResponse)
def get_stats():
    """Récupère les statistiques générales de l'API"""
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Total des jeux
                cur.execute("SELECT COUNT(*) as total FROM games")
                total_games = cur.fetchone()["total"]
                
                # Jeux avec prix
                cur.execute("SELECT COUNT(*) as total FROM best_price_pc")
                total_with_prices = cur.fetchone()["total"]
                
                # Statistiques de similarité
                cur.execute("""
                    SELECT 
                        AVG(similarity_score) as avg_similarity,
                        COUNT(CASE WHEN similarity_score >= 0.8 THEN 1 END) as high_quality
                    FROM best_price_pc 
                    WHERE similarity_score IS NOT NULL
                """)
                similarity_stats = cur.fetchone()
                
                # Dernière extraction
                cur.execute("SELECT last_extraction FROM api_state ORDER BY id DESC LIMIT 1")
                last_extraction_row = cur.fetchone()
                last_extraction = last_extraction_row["last_extraction"] if last_extraction_row else None
        
        return StatsResponse(
            total_games=total_games,
            total_games_with_prices=total_with_prices,
            avg_similarity_score=float(similarity_stats["avg_similarity"]) if similarity_stats["avg_similarity"] else None,
            high_quality_matches=similarity_stats["high_quality"],
            last_extraction=last_extraction.isoformat() if last_extraction else None
        )
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")

@app.get("/stats/user", tags=["statistics"])
def get_user_stats(username: str = Depends(verify_token)):
    """Récupère les statistiques personnalisées de l'utilisateur"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Interactions utilisateur
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT interaction_type) as unique_interaction_types,
                        AVG(rating) as avg_user_rating
                    FROM user_interactions 
                    WHERE user_id = %s
                """, (user_id,))
                interaction_stats = cur.fetchone()
                
                # Wishlist stats
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_wishlist_items,
                        COUNT(CASE WHEN is_active THEN 1 END) as active_items
                    FROM wishlist 
                    WHERE user_id = %s
                """, (user_id,))
                wishlist_stats = cur.fetchone()
                
                # Genres préférés basés sur les interactions
                cur.execute("""
                    SELECT g.genres, COUNT(*) as interaction_count
                    FROM user_interactions ui
                    JOIN games g ON ui.game_id_rawg = g.game_id_rawg
                    WHERE ui.user_id = %s AND g.genres IS NOT NULL
                    GROUP BY g.genres
                    ORDER BY interaction_count DESC
                    LIMIT 5
                """, (user_id,))
                favorite_genres = cur.fetchall()
        
        return {
            "user_id": user_id,
            "username": username,
            "interactions": {
                "total": interaction_stats["total_interactions"],
                "unique_types": interaction_stats["unique_interaction_types"],
                "avg_rating": float(interaction_stats["avg_user_rating"]) if interaction_stats["avg_user_rating"] else None
            },
            "wishlist": {
                "total_items": wishlist_stats["total_wishlist_items"],
                "active_items": wishlist_stats["active_items"]
            },
            "favorite_genres": [
                {
                    "genres": genre["genres"],
                    "interaction_count": genre["interaction_count"]
                }
                for genre in favorite_genres
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching user stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user statistics")

# =========================
# ENDPOINTS ADMIN
# =========================

@app.get("/admin/health-detailed", tags=["admin"])
def detailed_health_check(username: str = Depends(verify_token)):
    """Health check détaillé pour les administrateurs"""
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Vérifier toutes les tables
                tables_status = {}
                tables = ["games", "users", "best_price_pc", "api_state", "user_interactions"]
                
                for table in tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                        count = cur.fetchone()["count"]
                        tables_status[table] = {"status": "ok", "count": count}
                    except Exception as e:
                        tables_status[table] = {"status": "error", "error": str(e)}
                
                # État de l'API
                cur.execute("SELECT * FROM api_state ORDER BY id DESC LIMIT 1")
                api_state = cur.fetchone()
        
        return {
            "database": "connected",
            "tables": tables_status,
            "api_state": {
                "last_page": api_state["last_page"] if api_state else None,
                "last_extraction": api_state["last_extraction"].isoformat() if api_state and api_state["last_extraction"] else None,
                "total_games_extracted": api_state["total_games_extracted"] if api_state else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# =========================
# HEALTH & MONITORING
# =========================

@app.get("/healthz", tags=["system"])
def healthz():
    """Health check simple"""
    
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "disconnected",
            "error": str(e)
        }

# =========================
# STARTUP EVENT
# =========================

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage"""
    global DB_READY
    
    logger.info("Starting Anne's Games API...")
    
    try:
        if settings.db_configured:
            with get_db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT VERSION()")
                    version = cur.fetchone()
                    logger.info(f"Database connected: {version['VERSION()']}")
            
            DB_READY = True
            logger.info("Database connection established")
            
        else:
            logger.warning("Database not configured")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        if settings.DB_REQUIRED:
            raise
    
    logger.info("Anne's Games API startup completed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) genre in request.genres:
                        genre_conditions.append("g.genres LIKE %s")
                        params.append(f"%{genre}%")
                    sql += " AND (" + " OR ".join(genre_conditions) + ")"
                
                # Filtrer par plateformes
                if request.platforms:
                    platform_conditions = []
                    for platform in request.platforms:
                        platform_conditions.append("g.platforms LIKE %s")
                        params.append(f"%{platform}%")
                    sql += " AND (" + " OR ".join(platform_conditions) + ")"
                
                # Recherche textuelle si fournie
                if request.query.strip():
                    sql += " AND (g.title LIKE %s OR g.genres LIKE %s)"
                    query_param = f"%{request.query}%"
                    params.extend([query_param, query_param])
                
                # Exclure les jeux déjà dans les interactions de l'utilisateur
                if user_interactions:
                    interacted_games = [str(interaction["game_id_rawg"]) for interaction in user_interactions]
                    sql += f" AND g.game_id_rawg NOT IN ({','.join(interacted_games)})"
                
                sql += " ORDER BY g.rating DESC, g.metacritic DESC LIMIT %s"
                params.append(request.k)
                
                cur.execute(sql, params)
                recommended_games = cur.fetchall()
        
        games = [
            GameWithPriceResponse(
                game_id_rawg=game["game_id_rawg"],
                title=game["title"],
                release_date=game["release_date"].isoformat() if game["release_date"] else None,
                genres=game.get("genres"),
                platforms=game.get("platforms"),
                rating=game.get("rating"),
                metacritic=game.get("metacritic"),
                last_update=game["last_update"].isoformat() if game.get("last_update") else None,
                best_price_PC=game.get("best_price_PC"),
                best_shop_PC=game.get("best_shop_PC"),
                site_url_PC=game.get("site_url_PC"),
                similarity_score=float(game["similarity_score"]) if game.get("similarity_score") else None
            )
            for game in recommended_games
        ]
        
        return RecommendationResponse(
            games=games,
            query=request.query,
            total_results=len(games),
            filters_applied={
                "min_rating": request.min_rating,
                "genres": request.genres,
                "platforms": request.platforms,
                "user_interactions_excluded": len(user_interactions)
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

# =========================
# ENDPOINTS INTERACTIONS UTILISATEUR
# =========================

@app.post("/interactions", tags=["user-data"], status_code=201)
def add_user_interaction(
    game_id_rawg: int,
    interaction_type: str,
    rating: Optional[float] = None,
    username: str = Depends(verify_token)
):
    """Ajoute une interaction utilisateur"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Vérifier si le jeu existe
                cur.execute("SELECT game_id_rawg FROM games WHERE game_id_rawg = %s", (game_id_rawg,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Game not found")
                
                # Ajouter ou mettre à jour l'interaction
                cur.execute("""
                    INSERT INTO user_interactions (user_id, game_id_rawg, interaction_type, rating, timestamp)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                    interaction_type = VALUES(interaction_type),
                    rating = VALUES(rating),
                    timestamp = NOW()
                """, (user_id, game_id_rawg, interaction_type, rating))
        
        return {
            "message": "Interaction added successfully",
            "user_id": user_id,
            "game_id_rawg": game_id_rawg,
            "interaction_type": interaction_type,
            "rating": rating
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding user interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to add interaction")

@app.get("/interactions", tags=["user-data"])
def get_user_interactions(username: str = Depends(verify_token)):
    """Récupère les interactions de l'utilisateur"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ui.*, g.title, g.rating, g.genres
                    FROM user_interactions ui
                    JOIN games g ON ui.game_id_rawg = g.game_id_rawg
                    WHERE ui.user_id = %s
                    ORDER BY ui.timestamp DESC
                """, (user_id,))
                
                interactions = cur.fetchall()
        
        return {
            "user_id": user_id,
            "interactions": [
                {
                    "game_id_rawg": interaction["game_id_rawg"],
                    "title": interaction["title"],
                    "interaction_type": interaction["interaction_type"],
                    "rating": interaction.get("rating"),
                    "timestamp": interaction["timestamp"].isoformat() if interaction.get("timestamp") else None,
                    "game_rating": interaction.get("rating"),
                    "genres": interaction.get("genres")
                }
                for
