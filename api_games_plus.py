# api_games_plus.py — version corrigée
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from passlib.context import CryptContext

# Prometheus
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator


# ---------------------------
# Fallback settings (si import échoue)
# ---------------------------
try:
    from settings import get_settings  # type: ignore
except Exception:
    class _MockSettings:
        db_configured: bool = True
        DB_HOST: str = os.getenv("DB_HOST", "localhost")
        DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
        DB_USER: str = os.getenv("DB_USER", "root")
        DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
        DB_NAME: str = os.getenv("DB_NAME", "anne_games_db")
        ALLOW_ORIGINS: str = os.getenv("ALLOW_ORIGINS", "*")
        SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
        ALGORITHM: str = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
        DB_REQUIRED: bool = False

    def get_settings() -> _MockSettings:  # type: ignore
        return _MockSettings()

# Monitoring mock (optionnel)
try:
    from monitoring_metrics import get_monitor  # type: ignore
except Exception:
    class _MockMonitor:
        def record_prediction(self, *args, **kwargs) -> None:
            pass

    def get_monitor() -> _MockMonitor:  # type: ignore
        return _MockMonitor()


logger = logging.getLogger("games-api")
logging.basicConfig(level=logging.INFO)

settings = get_settings()

# ---------------------------
# Auth config
# ---------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------------------
# Pydantic models
# ---------------------------
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


class PriceInfo(BaseModel):
    title: str
    best_price_PC: Optional[str] = None
    best_shop_PC: Optional[str] = None
    site_url_PC: Optional[str] = None
    game_id_rawg: Optional[int] = None
    similarity_score: Optional[float] = None
    last_update: Optional[str] = None


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


class UserInteraction(BaseModel):
    user_id: int
    game_id_rawg: int
    interaction_type: str
    rating: Optional[float] = None


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


class StatsResponse(BaseModel):
    total_games: int
    total_games_with_prices: int
    avg_similarity_score: Optional[float] = None
    high_quality_matches: int
    last_extraction: Optional[str] = None


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Anne's Games API - Complete Edition",
    version="4.0.0",
    description="API complète pour les jeux avec authentification, recommandations, prix et wishlist",
)

Instrumentator().instrument(app).expose(app, include_in_schema=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in settings.ALLOW_ORIGINS.split(",") if o] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_READY: bool = False


# ---------------------------
# Root & metrics
# ---------------------------
@app.get("/", include_in_schema=False)
def root():
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
            "stats": ["/stats"],
        },
    }


@app.head("/", include_in_schema=False)
def head_root():
    return Response(status_code=200)


@app.get("/metrics", include_in_schema=False)
def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------
# DB util
# ---------------------------
def get_db_conn():
    import pymysql  # lazy import

    if not getattr(settings, "db_configured", True):
        raise RuntimeError("Database not configured")

    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


# ---------------------------
# Auth helpers
# ---------------------------
def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_ctx.verify(plain_password, hashed_password)
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_user_from_db(username: str) -> Optional[Dict[str, Any]]:
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, username,
                       COALESCE(hashed_password, password_hash) AS password_hash
                FROM users
                WHERE username = %s
                """,
                (username,),
            )
            return cur.fetchone()
    except Exception as e:
        logger.error(f"Error getting user from database: {e}")
        return None


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_user_from_db(username)
    if not user:
        return None
    if not verify_password(password, user.get("password_hash") or ""):
        return None
    return user


def verify_token(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_user_id_from_username(username: str) -> int:
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            if row:
                return int(row["id"])
    except Exception as e:
        logger.error(f"Error getting user ID: {e}")
    # Fallback stable
    return abs(hash(username)) % 1_000_000


# ---------------------------
# Auth endpoints
# ---------------------------
@app.post("/register", tags=["auth"], response_model=UserResponse, status_code=201)
def register_user(user_data: UserCreate):
    try:
        existing_user = get_user_from_db(user_data.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")

        hashed_password = hash_password(user_data.password)
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, hashed_password) VALUES (%s, %s)",
                (user_data.username, hashed_password),
            )

        new_user = get_user_from_db(user_data.username)
        assert new_user is not None
        return UserResponse(
            id=int(new_user["id"]),
            username=str(new_user["username"]),
            is_active=True,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/token", tags=["auth"], response_model=Token)
def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.get("/me", tags=["auth"], response_model=UserResponse)
def get_current_user(username: str = Depends(verify_token)):
    user = get_user_from_db(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        id=int(user["id"]),
        username=str(user["username"]),
        is_active=True,
    )


# ---------------------------
# Games endpoints
# ---------------------------
@app.get("/games", tags=["games"], response_model=List[GameResponse])
def get_games(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_rating: Optional[float] = Query(default=None, ge=0.0, le=5.0),
    genre: Optional[str] = Query(default=None),
    platform: Optional[str] = Query(default=None),
):
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            sql = "SELECT * FROM games WHERE 1=1"
            params: List[Any] = []

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
            rows = cur.fetchall()

        return [
            GameResponse(
                game_id_rawg=row["game_id_rawg"],
                title=row["title"],
                release_date=row["release_date"].isoformat() if row.get("release_date") else None,
                genres=row.get("genres"),
                platforms=row.get("platforms"),
                rating=row.get("rating"),
                metacritic=row.get("metacritic"),
                last_update=row["last_update"].isoformat() if row.get("last_update") else None,
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch games")


@app.get("/games/{game_id}", tags=["games"], response_model=GameWithPriceResponse)
def get_game_by_id(game_id: int):
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.*, p.best_price_PC, p.best_shop_PC, p.site_url_PC, p.similarity_score
                FROM games g
                LEFT JOIN best_price_pc p ON g.title = p.title
                WHERE g.game_id_rawg = %s
                """,
                (game_id,),
            )
            game = cur.fetchone()

        if not game:
            raise HTTPException(status_code=404, detail="Game not found")

        return GameWithPriceResponse(
            game_id_rawg=game["game_id_rawg"],
            title=game["title"],
            release_date=game["release_date"].isoformat() if game.get("release_date") else None,
            genres=game.get("genres"),
            platforms=game.get("platforms"),
            rating=game.get("rating"),
            metacritic=game.get("metacritic"),
            last_update=game["last_update"].isoformat() if game.get("last_update") else None,
            best_price_PC=game.get("best_price_PC"),
            best_shop_PC=game.get("best_shop_PC"),
            site_url_PC=game.get("site_url_PC"),
            similarity_score=float(game["similarity_score"]) if game.get("similarity_score") is not None else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching game {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch game")


@app.get("/games/search", tags=["games"], response_model=List[GameWithPriceResponse])
def search_games(q: str = Query(..., min_length=2), limit: int = Query(default=20, ge=1, le=100)):
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.*, p.best_price_PC, p.best_shop_PC, p.site_url_PC, p.similarity_score
                FROM games g
                LEFT JOIN best_price_pc p ON g.title = p.title
                WHERE g.title LIKE %s
                ORDER BY g.rating DESC
                LIMIT %s
                """,
                (f"%{q}%", limit),
            )
            rows = cur.fetchall()

        return [
            GameWithPriceResponse(
                game_id_rawg=row["game_id_rawg"],
                title=row["title"],
                release_date=row["release_date"].isoformat() if row.get("release_date") else None,
                genres=row.get("genres"),
                platforms=row.get("platforms"),
                rating=row.get("rating"),
                metacritic=row.get("metacritic"),
                last_update=row["last_update"].isoformat() if row.get("last_update") else None,
                best_price_PC=row.get("best_price_PC"),
                best_shop_PC=row.get("best_shop_PC"),
                site_url_PC=row.get("site_url_PC"),
                similarity_score=float(row["similarity_score"]) if row.get("similarity_score") is not None else None,
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error searching games: {e}")
        raise HTTPException(status_code=500, detail="Failed to search games")


# ---------------------------
# Prices endpoints
# ---------------------------
@app.get("/prices", tags=["prices"], response_model=List[PriceInfo])
def get_prices(
    limit: int = Query(default=50, ge=1, le=500),
    min_similarity: Optional[float] = Query(default=0.8, ge=0.0, le=1.0),
    shop: Optional[str] = Query(default=None),
):
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            sql = "SELECT * FROM best_price_pc WHERE 1=1"
            params: List[Any] = []

            if min_similarity is not None:
                sql += " AND similarity_score >= %s"
                params.append(min_similarity)
            if shop:
                sql += " AND best_shop_PC LIKE %s"
                params.append(f"%{shop}%")

            sql += " ORDER BY similarity_score DESC LIMIT %s"
            params.append(limit)

            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            PriceInfo(
                title=row["title"],
                best_price_PC=row.get("best_price_PC"),
                best_shop_PC=row.get("best_shop_PC"),
                site_url_PC=row.get("site_url_PC"),
                game_id_rawg=row.get("game_id_rawg"),
                similarity_score=float(row["similarity_score"]) if row.get("similarity_score") is not None else None,
                last_update=row["last_update"].isoformat() if row.get("last_update") else None,
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch prices")


@app.get("/prices/search", tags=["prices"], response_model=List[PriceInfo])
def search_prices(q: str = Query(..., min_length=2), limit: int = Query(default=20, ge=1, le=100)):
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM best_price_pc
                WHERE title LIKE %s
                ORDER BY similarity_score DESC
                LIMIT %s
                """,
                (f"%{q}%", limit),
            )
            rows = cur.fetchall()

        return [
            PriceInfo(
                title=row["title"],
                best_price_PC=row.get("best_price_PC"),
                best_shop_PC=row.get("best_shop_PC"),
                site_url_PC=row.get("site_url_PC"),
                game_id_rawg=row.get("game_id_rawg"),
                similarity_score=float(row["similarity_score"]) if row.get("similarity_score") is not None else None,
                last_update=row["last_update"].isoformat() if row.get("last_update") else None,
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error searching prices: {e}")
        raise HTTPException(status_code=500, detail="Failed to search prices")


# ---------------------------
# Recommendations
# ---------------------------
@app.post("/recommend", tags=["recommendations"], response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest, username: str = Depends(verify_token)):
    try:
        user_id = get_user_id_from_username(username)

        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT game_id_rawg, interaction_type, rating
                FROM user_interactions
                WHERE user_id = %s
                """,
                (user_id,),
            )
            user_interactions = cur.fetchall() or []

            sql = """
                SELECT g.*, p.best_price_PC, p.best_shop_PC, p.site_url_PC, p.similarity_score
                FROM games g
                LEFT JOIN best_price_pc p ON g.title = p.title
                WHERE 1=1
            """
            params: List[Any] = []

            if request.min_rating > 0:
                sql += " AND g.rating >= %s"
                params.append(request.min_rating)

            if request.genres:
                sql += " AND (" + " OR ".join(["g.genres LIKE %s"] * len(request.genres)) + ")"
                params.extend([f"%{genre}%" for genre in request.genres])

            if request.platforms:
                sql += " AND (" + " OR ".join(["g.platforms LIKE %s"] * len(request.platforms)) + ")"
                params.extend([f"%{p}%" for p in request.platforms])

            if request.query.strip():
                sql += " AND (g.title LIKE %s OR g.genres LIKE %s)"
                qp = f"%{request.query}%"
                params.extend([qp, qp])

            if user_interactions:
                ids = [str(i["game_id_rawg"]) for i in user_interactions]
                placeholders = ",".join(["%s"] * len(ids))
                sql += f" AND g.game_id_rawg NOT IN ({placeholders})"
                params.extend(ids)

            sql += " ORDER BY g.rating DESC, g.metacritic DESC LIMIT %s"
            params.append(request.k)

            cur.execute(sql, params)
            rows = cur.fetchall()

        games = [
            GameWithPriceResponse(
                game_id_rawg=row["game_id_rawg"],
                title=row["title"],
                release_date=row["release_date"].isoformat() if row.get("release_date") else None,
                genres=row.get("genres"),
                platforms=row.get("platforms"),
                rating=row.get("rating"),
                metacritic=row.get("metacritic"),
                last_update=row["last_update"].isoformat() if row.get("last_update") else None,
                best_price_PC=row.get("best_price_PC"),
                best_shop_PC=row.get("best_shop_PC"),
                site_url_PC=row.get("site_url_PC"),
                similarity_score=float(row["similarity_score"]) if row.get("similarity_score") is not None else None,
            )
            for row in rows
        ]

        return RecommendationResponse(
            games=games,
            query=request.query,
            total_results=len(games),
            filters_applied={
                "min_rating": request.min_rating,
                "genres": request.genres,
                "platforms": request.platforms,
                "user_interactions_excluded": len(user_interactions),
            },
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")


# ---------------------------
# Interactions
# ---------------------------
@app.post("/interactions", tags=["user-data"], status_code=201)
def add_user_interaction(
    game_id_rawg: int,
    interaction_type: str,
    rating: Optional[float] = None,
    username: str = Depends(verify_token),
):
    try:
        user_id = get_user_id_from_username(username)
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT game_id_rawg FROM games WHERE game_id_rawg = %s", (game_id_rawg,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Game not found")

            cur.execute(
                """
                INSERT INTO user_interactions (user_id, game_id_rawg, interaction_type, rating, timestamp)
                VALUES (%s, %s, %s, %s, NOW())
                ON DUPLICATE KEY UPDATE
                    interaction_type = VALUES(interaction_type),
                    rating = VALUES(rating),
                    timestamp = NOW()
                """,
                (user_id, game_id_rawg, interaction_type, rating),
            )

        return {
            "message": "Interaction added successfully",
            "user_id": user_id,
            "game_id_rawg": game_id_rawg,
            "interaction_type": interaction_type,
            "rating": rating,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding user interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to add interaction")


@app.get("/interactions", tags=["user-data"])
def get_user_interactions(username: str = Depends(verify_token)):
    try:
        user_id = get_user_id_from_username(username)
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT ui.*, g.title, g.rating AS game_rating, g.genres
                FROM user_interactions ui
                JOIN games g ON ui.game_id_rawg = g.game_id_rawg
                WHERE ui.user_id = %s
                ORDER BY ui.timestamp DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()

        return {
            "user_id": user_id,
            "interactions": [
                {
                    "game_id_rawg": r["game_id_rawg"],
                    "title": r["title"],
                    "interaction_type": r["interaction_type"],
                    "rating": r.get("rating"),
                    "timestamp": r["timestamp"].isoformat() if r.get("timestamp") else None,
                    "game_rating": r.get("game_rating"),
                    "genres": r.get("genres"),
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.error(f"Error fetching user interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch interactions")


# ---------------------------
# Stats & Health
# ---------------------------
@app.get("/stats", tags=["statistics"], response_model=StatsResponse)
def get_stats():
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM games")
            total_games = int(cur.fetchone()["total"])

            cur.execute("SELECT COUNT(*) AS total FROM best_price_pc")
            total_with_prices = int(cur.fetchone()["total"])

            cur.execute(
                """
                SELECT
                    AVG(similarity_score) AS avg_similarity,
                    COUNT(CASE WHEN similarity_score >= 0.8 THEN 1 END) AS high_quality
                FROM best_price_pc
                WHERE similarity_score IS NOT NULL
                """
            )
            sim = cur.fetchone() or {}

            cur.execute("SELECT last_extraction FROM api_state ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            last_extraction = row["last_extraction"] if row else None

        return StatsResponse(
            total_games=total_games,
            total_games_with_prices=total_with_prices,
            avg_similarity_score=float(sim["avg_similarity"]) if sim.get("avg_similarity") is not None else None,
            high_quality_matches=int(sim.get("high_quality") or 0),
            last_extraction=last_extraction.isoformat() if last_extraction else None,
        )
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")


@app.get("/healthz", tags=["system"])
def healthz():
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "disconnected",
            "error": str(e),
        }


# ---------------------------
# Startup
# ---------------------------
@app.on_event("startup")
async def startup_event():
    global DB_READY
    logger.info("Starting Anne's Games API...")
    try:
        if getattr(settings, "db_configured", True):
            with get_db_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT VERSION()")
                version = cur.fetchone()
                logger.info(f"Database connected: {version.get('VERSION()')}")
            DB_READY = True
            logger.info("Database connection established")
        else:
            logger.warning("Database not configured")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        if getattr(settings, "DB_REQUIRED", False):
            raise
    logger.info("Anne's Games API startup completed")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
