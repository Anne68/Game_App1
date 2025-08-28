# api_games_plus.py
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Literal

import logging
import pymysql
from fastapi import FastAPI, HTTPException, Depends, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
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
    version="1.2.0",
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


def connect_to_db():
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
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


def _candidate_game_ids(cur, input_id: int) -> List[int]:
    """Accepte un id local OU un RAWG id et renvoie les deux s’ils existent."""
    vals = [input_id]
    t_games = TABLES["games"]
    c_id = COLS["games"]["id"]
    c_rawg = COLS["games"]["rawg_id"]
    if not t_games or not (c_id or c_rawg):
        return vals

    where, params = [], []
    if c_id:
        where.append(f"{c_id}=%s")
        params.append(input_id)
    if c_rawg:
        where.append(f"{c_rawg}=%s")
        params.append(input_id)
    if not where:
        return vals

    select = []
    if c_id:
        select.append(f"{c_id} AS lid")
    if c_rawg:
        select.append(f"{c_rawg} AS rid")
    cur.execute(f"SELECT {', '.join(select)} FROM {t_games} WHERE " + " OR ".join(where), tuple(params))
    row = cur.fetchone()
    if row:
        for k in ("lid", "rid"):
            v = row.get(k)
            if v is not None and v not in vals:
                vals.append(v)
    return vals


def _first_game_ids_by_title(cur, title: str) -> List[int]:
    """Résout un titre -> [id_local?, id_rawg?] (première correspondance)."""
    t_games = TABLES["games"]
    c_title = COLS["games"]["title"] or "title"
    c_id = COLS["games"]["id"]
    c_rawg = COLS["games"]["rawg_id"]
    cur.execute(
        f"SELECT {c_id or 'NULL'} AS lid, {c_rawg or 'NULL'} AS rid FROM {t_games} WHERE {c_title} LIKE %s LIMIT 1",
        (f"%{title}%",),
    )
    row = cur.fetchone()
    if not row:
        return []
    out = []
    for k in ("lid", "rid"):
        if row.get(k) is not None:
            out.append(row[k])
    return out


# ========= Reco (TF-IDF) =========
vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix = None
index_to_title: List[str] = []
index_to_genres: List[str] = []


def _load_corpus_from_db() -> List[str]:
    with connect_to_db() as conn:
        with conn.cursor() as cur:
            c_title = COLS["games"]["title"] or "title"
            c_genres = COLS["games"]["genres"]
            t = TABLES["games"]
            cur.execute(f"SELECT {c_title} AS title, {c_genres or 'NULL'} AS genres FROM {t}")
            rows = cur.fetchall()
    global index_to_title, index_to_genres
    index_to_title = [r["title"] for r in rows]
    index_to_genres = [r.get("genres") or "" for r in rows]
    return [(r["title"] + " " + (r.get("genres") or "")).strip() for r in rows]


def _ensure_reco_model():
    global vectorizer, tfidf_matrix
    if vectorizer is None or tfidf_matrix is None:
        try:
            corpus = _load_corpus_from_db()
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            logger.info("Reco TF-IDF initialisée (%d jeux)", len(corpus))
        except Exception:
            logger.exception("Impossible de construire le modèle de reco")
            vectorizer = None
            tfidf_matrix = None


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
    except Exception as e:
        logger.exception("register failed")
        raise HTTPException(status_code=500, detail="Erreur lors de la création")


@app.post("/token", tags=["auth"])
@limiter.limit("20/minute")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user_from_db(username)
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    tok = create_access_token({"sub": user["username"]}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": tok, "token_type": "bearer"}


@app.get("/", tags=["public"])
def home():
    return {"message": "Bienvenue sur l'API des jeux vidéo !"}


# ---- LISTE/SEARCH JEUX ----
@app.get("/games", tags=["games"], dependencies=[Depends(verify_token)])
@limiter.limit("120/minute")
def get_all_games(
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    sort: Literal["title", "rating", "metacritic", "-title", "-rating", "-metacritic"] = "title",
):
    try:
        t = TABLES["games"]
        c_title = COLS["games"]["title"] or "title"
        c_genres = COLS["games"]["genres"]
        c_meta = COLS["games"]["metacritic"]
        c_rating = COLS["games"]["rating"]
        c_rawg = COLS["games"]["rawg_id"]

        sort_map = {
            "title": c_title,
            "-title": f"{c_title} DESC",
            "rating": c_rating or "NULL",
            "-rating": f"{(c_rating or 'NULL')} DESC",
            "metacritic": c_meta or "NULL",
            "-metacritic": f"{(c_meta or 'NULL')} DESC",
        }
        order_by = sort_map.get(sort, c_title)

        offset = (page - 1) * size
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT
                        {c_title} AS title,
                        {c_genres or "NULL"} AS genres,
                        {c_rating or "NULL"} AS rating,
                        {c_meta or "NULL"} AS metacritic,
                        {c_rawg or "NULL"} AS game_id_rawg
                       FROM {t}
                       ORDER BY {order_by}
                       LIMIT %s OFFSET %s""",
                    (size, offset),
                )
                rows = cur.fetchall()
        if not rows:
            return {"games": []}
        return {"games": rows, "page": page, "size": size}
    except Exception as e:
        _http500("get_all_games failed", e)


@app.get("/games/by-title/{title}", tags=["games"], dependencies=[Depends(verify_token)])
@app.get("/games/title/{title}", tags=["games"], dependencies=[Depends(verify_token)])
@limiter.limit("120/minute")
def get_games_by_title(request: Request, title: str):
    try:
        t = TABLES["games"]
        c_title = COLS["games"]["title"] or "title"
        c_genres = COLS["games"]["genres"]
        c_meta = COLS["games"]["metacritic"]
        c_rating = COLS["games"]["rating"]
        c_rawg = COLS["games"]["rawg_id"]
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT
                        {c_title} AS title,
                        {c_genres or "NULL"} AS genres,
                        {c_rating or "NULL"} AS rating,
                        {c_meta or "NULL"} AS metacritic,
                        {c_rawg or "NULL"} AS game_id_rawg
                       FROM {t}
                       WHERE {c_title} LIKE %s""",
                    (f"%{title}%",),
                )
                rows = cur.fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Aucun jeu trouvé")
        return {"games": rows}
    except Exception as e:
        _http500("get_games_by_title failed", e)


# ---- PRIX ----
def _query_prices_joined_by_gid(cur, gid: int) -> List[Dict[str, Any]]:
    t_prices = TABLES["prices"]
    t_plat = TABLES["platforms"]
    t_gp = TABLES["game_platforms"]

    c_gid = COLS["prices"]["game_id"] or _infer_col_by_alias(cur, t_prices, ["game_id", "game_id_rawg", "id_jeu", "id_game"])
    c_plid = COLS["prices"]["platform_id"] or _infer_col_by_alias(cur, t_prices, ["platform_id", "id_plateforme", "id_platform"])
    c_price = COLS["prices"]["best_price"] or _infer_col_by_alias(cur, t_prices, ["best_price", "best_price_pc", "price"])
    c_shop = COLS["prices"]["best_shop"] or _infer_col_by_alias(cur, t_prices, ["best_shop", "best_shop_pc", "shop", "store"])
    c_url = COLS["prices"]["site_url"] or _infer_col_by_alias(cur, t_prices, ["site_url", "site_url_pc", "url", "link"])
    c_upd = COLS["prices"]["last_update"] or _infer_col_by_alias(cur, t_prices, ["last_update", "updated_at"])

    p_id = COLS["platforms"]["id"] or _infer_col_by_alias(cur, t_plat, ["platform_id", "id_platform", "id_plateforme"])
    p_name = COLS["platforms"]["name"] or _infer_col_by_alias(cur, t_plat, ["platform_name", "name", "nom"])

    rows: List[Dict[str, Any]] = []

    if c_plid:
        cur.execute(
            f"""
            SELECT
                {c_price} AS price,
                {c_shop}  AS shop,
                {c_url}   AS link,
                {c_upd}   AS last_update,
                P.{p_name} AS platform
            FROM {t_prices} T
            LEFT JOIN {t_plat} P ON P.{p_id} = T.{c_plid}
            WHERE T.{c_gid} = %s
            ORDER BY {c_price} ASC
            """,
            (gid,),
        )
        rows = cur.fetchall()
    else:
        if not t_gp:
            return []
        gp_gid = COLS["game_platforms"]["game_id"] or _infer_col_by_alias(cur, t_gp, ["game_id", "game_id_rawg", "id_jeu", "id_game"])
        gp_plid = COLS["game_platforms"]["platform_id"] or _infer_col_by_alias(cur, t_gp, ["platform_id", "id_plateforme", "id_platform"])

        cur.execute(f"SELECT {gp_plid} AS platform_id FROM {t_gp} WHERE {gp_gid}=%s", (gid,))
        pids = [r["platform_id"] for r in cur.fetchall()]

        cur.execute(
            f"""SELECT {c_price} AS price, {c_shop} AS shop, {c_url} AS link, {c_upd} AS last_update
                FROM {t_prices} WHERE {c_gid}=%s ORDER BY {c_price} ASC""",
            (gid,),
        )
        base = cur.fetchall()

        pname_map: Dict[int, str] = {}
        if pids:
            in_list = ",".join(["%s"] * len(pids))
            cur.execute(
                f"SELECT {p_id} AS pid, {p_name} AS pname FROM {t_plat} WHERE {p_id} IN ({in_list})",
                tuple(pids),
            )
            pname_map = {r["pid"]: r["pname"] for r in cur.fetchall()}

        for r in base:
            if pname_map:
                for pname in pname_map.values():
                    rows.append({**r, "platform": pname})
            else:
                rows.append({**r, "platform": None})

    for r in rows:
        r["last_update"] = _iso(r.get("last_update"))
    return rows


@app.get("/games/{game_id}/prices", tags=["prices"], dependencies=[Depends(verify_token)])
@limiter.limit("120/minute")
def get_prices_for_game(request: Request, game_id: int):
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                for gid in _candidate_game_ids(cur, game_id):
                    rows = _query_prices_joined_by_gid(cur, gid)
                    if rows:
                        return {"prices": rows}
        raise HTTPException(status_code=404, detail="Aucun prix trouvé pour ce jeu")
    except HTTPException:
        raise
    except Exception as e:
        _http500("get_prices_for_game failed", e)


@app.get("/games/by-title/{title}/prices", tags=["prices"], dependencies=[Depends(verify_token)])
@app.get("/games/title/{title}/prices", tags=["prices"], dependencies=[Depends(verify_token)])
@limiter.limit("120/minute")
def get_prices_for_game_by_title(request: Request, title: str):
    try:
        with connect_to_db() as conn:
            with conn.cursor() as cur:
                gids = _first_game_ids_by_title(cur, title)
                if not gids:
                    raise HTTPException(status_code=404, detail="Aucun jeu trouvé pour ce titre")
                for gid in gids:
                    rows = _query_prices_joined_by_gid(cur, gid)
                    if rows:
                        return {"prices": rows}
        raise HTTPException(status_code=404, detail="Aucun prix trouvé pour ce titre")
    except HTTPException:
        raise
    except Exception as e:
        _http500("get_prices_for_game_by_title failed", e)


# ---- PLATFORMS ----
@app.get("/games/{game_id}/platforms", tags=["games"], dependencies=[Depends(verify_token)])
@limiter.limit("120/minute")
def get_platforms_for_game(request: Request, game_id: int):
    try:
        t_gp = TABLES["game_platforms"]
        t_plat = TABLES["platforms"]
        if not (t_gp and t_plat):
            raise HTTPException(status_code=404, detail="Tables plateformes introuvables")

        with connect_to_db() as conn:
            with conn.cursor() as cur:
                gp_gid = COLS["game_platforms"]["game_id"] or _infer_col_by_alias(cur, t_gp, ["game_id", "game_id_rawg", "id_jeu", "id_game"])
                gp_plid = COLS["game_platforms"]["platform_id"] or _infer_col_by_alias(cur, t_gp, ["platform_id", "id_plateforme", "id_platform"])
                p_id = COLS["platforms"]["id"] or _infer_col_by_alias(cur, t_plat, ["platform_id", "id_platform", "id_plateforme"])
                p_name = COLS["platforms"]["name"] or _infer_col_by_alias(cur, t_plat, ["platform_name", "name", "nom"])

                for gid in _candidate_game_ids(cur, game_id):
                    cur.execute(f"SELECT {gp_plid} AS platform_id FROM {t_gp} WHERE {gp_gid}=%s", (gid,))
                    pids = [r["platform_id"] for r in cur.fetchall()]
                    if not pids:
                        continue
                    in_list = ",".join(["%s"] * len(pids))
                    cur.execute(f"SELECT {p_id} AS id, {p_name} AS name FROM {t_plat} WHERE {p_id} IN ({in_list})", tuple(pids))
                    rows = cur.fetchall()
                    if rows:
                        return {"platform_ids": [r["id"] for r in rows], "platforms": [r["name"] for r in rows]}
        raise HTTPException(status_code=404, detail="Aucune plateforme trouvée pour ce jeu")
    except HTTPException:
        raise
    except Exception as e:
        _http500("get_platforms_for_game failed", e)


@app.get("/games/by-title/{title}/platforms", tags=["games"], dependencies=[Depends(verify_token)])
@app.get("/games/title/{title}/platforms", tags=["games"], dependencies=[Depends(verify_token)])
@limiter.limit("120/minute")
def get_platforms_for_title(request: Request, title: str):
    try:
        t_gp = TABLES["game_platforms"]
        t_plat = TABLES["platforms"]
        if not (t_gp and t_plat):
            raise HTTPException(status_code=404, detail="Tables plateformes introuvables")

        with connect_to_db() as conn:
            with conn.cursor() as cur:
                gids = _first_game_ids_by_title(cur, title)
                if not gids:
                    raise HTTPException(status_code=404, detail="Aucun jeu trouvé pour ce titre")

                gp_gid = COLS["game_platforms"]["game_id"] or _infer_col_by_alias(cur, t_gp, ["game_id", "game_id_rawg", "id_jeu", "id_game"])
                gp_plid = COLS["game_platforms"]["platform_id"] or _infer_col_by_alias(cur, t_gp, ["platform_id", "id_plateforme", "id_platform"])
                p_id = COLS["platforms"]["id"] or _infer_col_by_alias(cur, t_plat, ["platform_id", "id_platform", "id_plateforme"])
                p_name = COLS["platforms"]["name"] or _infer_col_by_alias(cur, t_plat, ["platform_name", "name", "nom"])

                gid = gids[0]
                cur.execute(f"SELECT {gp_plid} AS platform_id FROM {t_gp} WHERE {gp_gid}=%s", (gid,))
                pids = [r["platform_id"] for r in cur.fetchall()]
                if not pids:
                    raise HTTPException(status_code=404, detail="Aucune plateforme trouvée")

                in_list = ",".join(["%s"] * len(pids))
                cur.execute(f"SELECT {p_id} AS id, {p_name} AS name FROM {t_plat} WHERE {p_id} IN ({in_list})", tuple(pids))
                rows = cur.fetchall()
        return {"platform_ids": [r["id"] for r in rows], "platforms": [r["name"] for r in rows]}
    except HTTPException:
        raise
    except Exception as e:
        _http500("get_platforms_for_title failed", e)


# ---- RECO ----
@app.get("/recommend/by-title/{title}", tags=["ai"], dependencies=[Depends(verify_token)])
@limiter.limit("60/minute")
def recommend_by_title(request: Request, title: str, k: int = 5):
    _ensure_reco_model()
    if not (vectorizer and tfidf_matrix is not None):
        raise HTTPException(status_code=503, detail="Modèle non initialisé")
    query = vectorizer.transform([title])
    sims = cosine_similarity(query, tfidf_matrix).ravel()
    order = sims.argsort()[::-1]
    results = []
    for i in order:
        if index_to_title[i].lower() == title.lower():
            continue
        results.append({"title": index_to_title[i], "score": float(sims[i])})
        if len(results) >= k:
            break
    return {"recommendations": results}


@app.get("/recommend/by-genre/{genre}", tags=["ai"], dependencies=[Depends(verify_token)])
@limiter.limit("60/minute")
def recommend_by_genre(request: Request, genre: str, k: int = 5):
    _ensure_reco_model()
    if not (vectorizer and tfidf_matrix is not None):
        raise HTTPException(status_code=503, detail="Modèle non initialisé")
    query = vectorizer.transform([genre])
    sims = cosine_similarity(query, tfidf_matrix).ravel()
    order = sims.argsort()[::-1]
    results = []
    for i in order:
        results.append({"title": index_to_title[i], "score": float(sims[i]), "genres": index_to_genres[i]})
        if len(results) >= k:
            break
    return {"recommendations": results}


# ========= Startup =========
@app.on_event("startup")
def on_startup():
    try:
        autodetect_schema()
        _ensure_reco_model()
    except Exception:
        logger.exception("Startup tasks failed")
