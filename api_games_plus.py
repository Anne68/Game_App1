# api_games_plus.py — API FastAPI (auth + search + recos)
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Depends, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger("games-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# ========= Config =========
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="Games API (C9→C13)", version="1.2.2", description="API sécurisée avec monitoring, modèle de reco, tests et CI/CD")

# CORS (Streamlit, Render…)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Demo Data =========
# NB : remplace ceci par ta BDD MySQL si besoin
GAMES: List[Dict] = [
    {"id": 1, "title": "The Witcher 3: Wild Hunt", "genres": "RPG, Action", "rating": 4.9, "metacritic": 93, "platforms": ["PC", "PS4", "Xbox One"]},
    {"id": 2, "title": "Hades", "genres": "Action, Roguelike", "rating": 4.8, "metacritic": 93, "platforms": ["PC", "Switch", "PS4"]},
    {"id": 3, "title": "Stardew Valley", "genres": "Simulation, RPG", "rating": 4.7, "metacritic": 89, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 4, "title": "Celeste", "genres": "Platformer, Indie", "rating": 4.6, "metacritic": 94, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 5, "title": "Doom Eternal", "genres": "Action, FPS", "rating": 4.5, "metacritic": 88, "platforms": ["PC", "PS4", "Xbox One"]},
    {"id": 6, "title": "Hollow Knight", "genres": "Metroidvania, Indie", "rating": 4.8, "metacritic": 90, "platforms": ["PC", "Switch", "PS4", "Xbox One"]},
    {"id": 7, "title": "Disco Elysium", "genres": "RPG, Narrative", "rating": 4.7, "metacritic": 91, "platforms": ["PC", "PS4", "Xbox One"]},
]

USERS: Dict[str, str] = {}  # username -> hashed_password (démo mémoire)


# ========= Utils =========
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


def tokenize(text: str) -> List[str]:
    # petit tokenizer simple (lower + split non-alphanum)
    import re
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ========= System =========
@app.get("/healthz", tags=["system"])
def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/__paths", tags=["system"])
def list_paths(request: Request):
    return {"paths": sorted({r.path for r in request.app.routes})}


# prometheus /metrics
Instrumentator().instrument(app).expose(app, include_in_schema=True)


# ========= Auth =========
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


# ========= Games =========
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


# ========= Recos =========
@app.get("/recommend/by-title/{title}", tags=["recommend"])
def recommend_by_title(title: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    tok_q = tokenize(title)
    scored = []
    for g in GAMES:
        score = 0.6 * jaccard(tok_q, tokenize(g["title"])) + 0.4 * jaccard(tok_q, tokenize(g["genres"]))
        scored.append({"title": g["title"], "genres": g["genres"], "score": round(float(score), 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"recommendations": scored[:k]}


@app.get("/recommend/by-genre/{genre}", tags=["recommend"])
def recommend_by_genre(genre: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    gq = genre.strip().lower()
    filt = [g for g in GAMES if gq in g["genres"].lower()]
    # tri simple : rating puis metacritic
    filt.sort(key=lambda g: (g["rating"], g["metacritic"]), reverse=True)
    recs = [{"title": g["title"], "genres": g["genres"], "score": float(g["rating"])} for g in filt[:k]]
    return {"recommendations": recs}


# ========= Optional root/HEAD (évite 405 dans les logs) =========
@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "version": app.version}

@app.head("/", include_in_schema=False)
def root_head():
    return


# ========= Startup log =========
@app.on_event("startup")
def on_startup():
    logger.info("Application startup complete. %d jeux chargés.", len(GAMES))
