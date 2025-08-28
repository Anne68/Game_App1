# api_games_plus.py
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Literal

import logging
import pymysql
from fastapi import FastAPI, HTTPException, Depends, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# Reco
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from settings import get_settings


# ========= App & config =========
settings = get_settings()
app = FastAPI(
    title="Games API (C9→C13)",
    description="API sécurisée avec monitoring, modèle de reco, tests et CI/CD",
    version="1.2.1",
)

# CORS
origins = [o.strip() for o in settings.ALLOW_ORIGINS.split(",") if o]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Logs
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("games-api")
DEBUG_MODE = settings.LOG_LEVEL.upper() == "DEBUG"

# Sécurité / JWT
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
ALGORITHM = settings.ALGORITHM
SECRET_KEY = settings.SECRET_KEY
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


def _normalize_db_user(u: str) -> str:
    """Si l'env fournit 'anne@2a00:...' on ne garde que 'anne' (MySQL attend juste le login)."""
    if "@" in u:
        right = u.split("@", 1)[1]
        if "." in right or ":" in right:
            return u.split("@", 1)[0]
    return u


def connect_to_db():
    ssl_args = {}
    if getattr(settings, "DB_SSL_CA", None):
        ssl_args = {"ssl": {"ca": settings.DB_SSL_CA}}  # TLS Alwaysdata

    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=_normalize_db_user(settings.DB_USER),
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        **ssl_args,
    )


# ========= Schema autodetect =========
TABLES: Dict[str, Optional[str]] = {"games": None, "prices": None, "game_platforms": None, "platforms": None}
COLS: Dict[str, Dict[str, Optional[str]]] = {
    "games": {"id": None, "title": None, "genres": None, "rating": None, "metacritic": None, "rawg_id": None},
    "prices": {"id": None, "title": None, "game_id": None, "platform_id": None, "best_price": None, "best_shop": None, "site_url": None, "last_update": None},
    "game_platforms": {"game_id": None, "platform_id": None},
    "platforms": {"id": None, "name": None},
}
CANDIDATE_TABLES = {
    "games": ["games", "jeux"],
    "prices": ["best_price_pc", "prix"],
    "game_platforms": ["game_platforms", "jeux_plateformes"],
    "platforms": ["platforms", "plateformes"],
}
CANDIDATE_COLS = {
    "games": {
        "id": ["id", "id_jeu", "game_id", "id_game"],
        "title": ["title", "titre", "name"],
        "genres": ["genres", "genre", "tags"],
        "rating": ["rating", "note"],
        "metacritic": ["metacritic", "metascore"],
        "rawg_id": ["game_id_rawg", "rawg_id", "id_rawg"],
    },
    "prices": {
        "id": ["id", "id_prix"],
        "title": ["title", "game_title", "titre"],
        "game_id": ["game_id", "game_id_rawg", "id_jeu", "id_game"],
        "platform_id": ["platform_id", "id_plateforme", "id_platform"],
        "best_price": ["best_price", "best_price_pc", "price"],
        "best_shop": ["best_shop", "best_shop_pc", "shop", "store"],
        "site_url": ["site_url", "site_url_pc", "url", "link"],
        "last_update": ["last_update", "updated_at"],
    },
    "game_platforms": {
        "game_id": ["game_id", "game_id_rawg", "id_jeu", "id_game"],
        "platform_id": ["platform_id", "id_plateforme", "id_platform"],
    },
    "platforms": {
        "id": ["platform_id", "id_platform", "id_plateforme"],
        "name": ["platform_name", "name", "nom"],
    },
}


def _tables_map(cur) -> dict:
    cur.execute(
        "SELECT TABLE_NAME AS name FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s",
        (settings.DB_NAME,),
    )
    return {r["name"].lower(): r["name"] for r in cur.fetchall() if r.get("name")}


def _columns_map(cur, tname: str) -> dict:
    cur.execute(
        "SELECT COLUMN_NAME AS name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
        (settings.DB_NAME, tname),
    )
    return {r["name"].lower(): r["name"] for r in cur.fetchall() if r.get("name")}


def _infer_col_by_alias(cur, real_table_name: str, candidates: List[str]) -> Optional[str]:
    cmap = _columns_map(cur, real_table_name)
    for c in candidates:
        if c.lower() in cmap:
            return cmap[c.lower()]
    return None


def autodetect_schema():
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            tmap = _tables_map(cur)
            for key, cands in CANDIDATE_TABLES.items():
                TABLES[key] = next((tmap[c] for c in cands if c in tmap), None)
            if not TABLES["games"]:
                raise RuntimeError("Aucune table 'games/jeux' trouvée.")

            for tkey, real_name in TABLES.items():
                if not real_name:
                    continue
                cmap = _columns_map(cur, real_name)
                for ckey, cands in CANDIDATE_COLS[tkey].items():
                    COLS[tkey][ckey] = next((cmap[c] for c in cands if c in cmap), None)

    logger.info("TABLES=%s", TABLES)
    logger.info("COLS=%s", COLS)


# ========= Auth =========
def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def get_user_from_db(username: str):
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users WHERE username=%s", (username,))
                return cur.fetchone()
    except Exception:
        logger.exception("get_user_from_db failed")
        return None


def create_access_token(data: dict, exp: timedelta | None = None):
    payload = data.copy()
    payload.update({"exp": datetime.utcnow() + (exp or timedelta(minutes=15))})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Token invalide")
        return sub
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")


# ========= Monitoring =========
if settings.PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app)


# ========= Utils =========
def _http500(msg: str, exc: Exception):
    logger.exception("%s: %s", msg, exc)
    if DEBUG_MODE:
        raise HTTPException(status_code=500, detail=f"{msg}: {exc}")
    raise HTTPException(status_code=500, detail="Erreur interne")


def _iso(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return v


# ========= Petit debug pour voir les routes exposées =========
@app.get("/__paths", tags=["system"])
def list_paths():
    return {"paths": sorted({getattr(r, "path", "") for r in app.routes if getattr(r, "path", "")})}


# ========= Routes =========
@app.get("/healthz", tags=["system"])
def healthz():
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=500, detail="DB not reachable")


@app.post("/register", tags=["auth"])
@limiter.limit("5/minute")
def register(request: Request, username: str = Form(...), password: str = Form(...)):
    if get_user_from_db(username):
        raise HTTPException(status_code=409, detail="Nom d'utilisateur déjà pris")
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (username, hashed_password) VALUES (%s,%s)",
                    (username, pwd_context.hash(password)),
                )
        return {"message": f"Utilisateur '{username}' créé"}
    except Exception:
        logger.exception("register failed")
        raise HTTPException(status_code=500, detail="Erreur lors de la création")


# ======= TOKEN (OAuth2) + alias =======
@app.post("/token", tags=["auth"])
@limiter.limit("20/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_from_db(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    tok = create_access_token({"sub": user["username"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": tok, "token_type": "bearer"}


@app.post("/auth/token", tags=["auth"])
@app.post("/login", tags=["auth"])
@app.post("/login/access-token", tags=["auth"])
@app.post("/api/token", tags=["auth"])
def login_alias(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    return login(request, form_data)


@app.get("/", tags=["public"])
def home():
    return {"message": "Bienvenue sur l'API des jeux vidéo !"}


# ---- (le reste : games / prices / platforms / recommend) ----
# [!!!] Rien à changer ci-dessous : conserve exactement tes fonctions existantes
#       pour /games, /games/by-title, /prices, /platforms, /recommend, etc.
#       (je ne les répète pas ici pour ne pas dépasser la taille du message).
#       Garde ton code tel qu'il était après autodétection et TF-IDF.
