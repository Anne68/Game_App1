# api.py
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Query, HTTPException, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    SecurityScopes,
)
from pydantic import BaseModel, Field, ValidationError
from jose import JWTError, jwt
from passlib.context import CryptContext

from model_manager import get_model

# ============================================================
# Config API + CORS
# ============================================================
app = FastAPI(
    title="Game Reco API (JWT Protected)",
    version="1.1.0",
    description="""
API de recommandations de jeux vidéo.

**Sécurité** :
- Authentification **OAuth2 Password** (POST /token) → JWT Bearer
- Swagger permet de s'authentifier via le bouton **Authorize**
- Scopes:
  - `recommend:read` pour les endpoints de recommandations
  - `clusters:read` pour les endpoints d'exploration de clusters
  - `metrics:read` pour l'endpoint de métriques du modèle
""",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 🔒 restreindre en prod
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # /token est POST
    allow_headers=["*"],
)

# ============================================================
# Sécurité / Auth (OAuth2 + JWT)
# ============================================================
# ⚠️ En prod: mettre ce secret en variable d'env (ou vault)
SECRET_KEY = "CHANGE_ME_TO_A_LONG_RANDOM_SECRET_!!!"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

OAUTH2_SCOPES = {
    "recommend:read": "Lire les recommandations filtrées et par similarité",
    "clusters:read": "Explorer les clusters (tailles, top termes, échantillons)",
    "metrics:read": "Consulter les métriques du modèle",
}
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes=OAUTH2_SCOPES,
)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Modèles auth ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []

class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = False
    scopes: List[str] = []

class UserInDB(User):
    hashed_password: str

# --- Fake users DB (à remplacer par MySQL si besoin) ---
fake_users_db: Dict[str, UserInDB] = {
    "demo": UserInDB(
        username="demo",
        full_name="Demo User",
        disabled=False,
        scopes=["recommend:read", "clusters:read", "metrics:read"],
        hashed_password=pwd_context.hash("demo"),  # password: demo
    )
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db: Dict[str, UserInDB], username: str) -> Optional[UserInDB]:
    return db.get(username)

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(fake_users_db, username)
    if not user or not verify_password(password, user.hashed_password) or user.disabled:
        return None
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> User:
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Impossible de valider les identifiants",
        headers={"WWW-Authenticate": authenticate_value},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        token_scopes = payload.get("scopes", [])
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, scopes=token_scopes)
    except (JWTError, ValidationError):
        raise credentials_exception

    user = get_user(fake_users_db, token_data.username)
    if user is None:
        raise credentials_exception

    # Vérifier les scopes requis
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permissions insuffisantes (scope manquant)",
                headers={"WWW-Authenticate": authenticate_value},
            )
    return user

# ============================================================
# Endpoint Auth
# ============================================================
@app.post("/token", response_model=Token, summary="Obtenir un JWT (OAuth2 Password)")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Envoie `username` et `password` (form-data).
    Retourne un JWT Bearer (access_token).
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Nom d'utilisateur ou mot de passe incorrect")
    granted_scopes = user.scopes or []
    access_token = create_access_token(
        data={"sub": user.username, "scopes": granted_scopes},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, token_type="bearer")

# ============================================================
# Charger le modèle
# ============================================================
model = get_model()

# ============================================================
# Schémas de réponse
# ============================================================
class RecommendationItem(BaseModel):
    id: int
    title: str
    genres: Optional[str] = None
    platforms: Optional[Any] = None  # peut être list ou str selon ta source
    price: Optional[float] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    rating: float
    metacritic: int
    cluster: int

class RecommendationResponse(BaseModel):
    model_version: str
    total: int
    items: List[RecommendationItem]

class ClusterGameItem(BaseModel):
    id: int
    title: str
    genres: Optional[str] = None
    rating: float
    metacritic: int

class ClusterExploreItem(BaseModel):
    cluster: int
    size: int
    top_terms: List[str]

class ClusterExploreResponse(BaseModel):
    n_clusters: int
    sizes: List[int]
    top_terms: List[ClusterExploreItem]

class RandomClusterResponse(BaseModel):
    cluster: int
    sample: List[ClusterGameItem]

class MetricsResponse(BaseModel):
    total_predictions: int
    avg_confidence: float
    last_training: Optional[str]
    model_version: str
    feature_dim: int
    is_trained: bool
    games_count: int

# ============================================================
# Health public
# ============================================================
@app.get("/health", summary="Health check (public)")
def health():
    return {
        "ok": True,
        "is_trained": model.is_trained,
        "games_count": 0 if model.games_df is None else len(model.games_df),
        "model_version": model.model_version,
    }

# ============================================================
# Recommandations (protégées)
# ============================================================
@app.get(
    "/recommend/filters",
    response_model=RecommendationResponse,
    summary="Recommandations filtrées par genre/plateformes/prix (protégé)",
)
def recommend_with_filters(
    genre: Optional[str] = Query(None, description="Genre recherché (ex: 'Action RPG')"),
    platforms: Optional[List[str]] = Query(
        None,
        description="Plateformes (répéter: &platforms=switch&platforms=pc ou CSV: 'switch,pc')"
    ),
    max_price: Optional[float] = Query(None, gt=0, description="Prix maximum"),
    k: int = Query(10, gt=0, le=100, description="Nombre max de recommandations"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Score minimal (0-1)"),
    sort_by: str = Query("score", pattern="^(score|price)$", description="Tri: 'score' ou 'price'"),
    current_user: User = Security(get_current_user, scopes=["recommend:read"]),
):
    # CSV pour platforms supporté
    if platforms and len(platforms) == 1 and "," in platforms[0]:
        platforms = [p.strip() for p in platforms[0].split(",") if p.strip()]

    if not model.is_trained or model.games_df is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")

    try:
        recs = model.recommend_with_filters(
            genre=genre,
            platforms=platforms,
            max_price=max_price,
            k=k,
            min_confidence=min_confidence,
            sort_by=sort_by,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RecommendationResponse(
        model_version=model.model_version,
        total=len(recs),
        items=[RecommendationItem(**r) for r in recs]
    )

@app.get(
    "/recommend/query",
    response_model=RecommendationResponse,
    summary="Recommandations par requête libre (protégé)",
)
def recommend_query(
    q: str = Query(..., min_length=1, description="Texte libre (ex: 'roguelike metroidvania on switch')"),
    k: int = Query(10, gt=0, le=100),
    min_confidence: float = Query(0.1, ge=0.0, le=1.0),
    current_user: User = Security(get_current_user, scopes=["recommend:read"]),
):
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")
    recs = model.predict(query=q, k=k, min_confidence=min_confidence)
    return RecommendationResponse(
        model_version=model.model_version,
        total=len(recs),
        items=[RecommendationItem(**r) for r in recs]
    )

@app.get(
    "/recommend/by-id",
    response_model=RecommendationResponse,
    summary="Jeux similaires à un jeu donné (protégé)",
)
def recommend_by_id(
    game_id: int = Query(..., description="Identifiant du jeu existant"),
    k: int = Query(10, gt=0, le=100),
    current_user: User = Security(get_current_user, scopes=["recommend:read"]),
):
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")
    try:
        recs = model.predict_by_game_id(game_id=game_id, k=k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return RecommendationResponse(
        model_version=model.model_version,
        total=len(recs),
        items=[RecommendationItem(**r) for r in recs]
    )

@app.get(
    "/recommend/by-title",
    response_model=RecommendationResponse,
    summary="Similarité stricte par titre (protégé, fuzzy)",
)
def recommend_by_title(
    title: str = Query(..., min_length=1, description="Titre du jeu (typos acceptées)"),
    k: int = Query(10, gt=0, le=100),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    current_user: User = Security(get_current_user, scopes=["recommend:read"]),
):
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")
    recs = model.recommend_by_title_similarity(query_title=title, k=k, min_confidence=min_confidence)
    return RecommendationResponse(
        model_version=model.model_version,
        total=len(recs),
        items=[RecommendationItem(**r) for r in recs]
    )

@app.get(
    "/recommend/by-genre",
    response_model=RecommendationResponse,
    summary="Recommandations par genre (protégé)",
)
def recommend_by_genre(
    genre: str = Query(..., min_length=1, description="Genre (ex: 'Action RPG')"),
    k: int = Query(10, gt=0, le=100),
    current_user: User = Security(get_current_user, scopes=["recommend:read"]),
):
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")
    recs = model.recommend_by_genre(genre=genre, k=k)
    return RecommendationResponse(
        model_version=model.model_version,
        total=len(recs),
        items=[RecommendationItem(**r) for r in recs]
    )

# ============================================================
# Clusters (protégés)
# ============================================================
@app.get(
    "/clusters/explore",
    response_model=ClusterExploreResponse,
    summary="Explorer les clusters (tailles, top termes) (protégé)",
)
def clusters_explore(
    top_n_terms: int = Query(10, gt=0, le=50, description="Nombre de termes top par cluster"),
    current_user: User = Security(get_current_user, scopes=["clusters:read"]),
):
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")
    data = model.cluster_explore(top_n_terms=top_n_terms)
    # adapter au modèle Pydantic
    return ClusterExploreResponse(
        n_clusters=data["n_clusters"],
        sizes=data["sizes"],
        top_terms=[ClusterExploreItem(**d) for d in data["top_terms"]]
    )

@app.get(
    "/clusters/random",
    response_model=RandomClusterResponse,
    summary="Prendre un cluster au hasard (protégé)",
)
def cluster_random(
    sample: int = Query(12, gt=0, le=100, description="Taille de l'échantillon renvoyé"),
    current_user: User = Security(get_current_user, scopes=["clusters:read"]),
):
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")
    data = model.random_cluster(sample=sample)
    return RandomClusterResponse(
        cluster=data["cluster"],
        sample=[ClusterGameItem(**g) for g in data["sample"]]
    )

@app.get(
    "/clusters/{cluster_id}/games",
    response_model=List[ClusterGameItem],
    summary="Lister les jeux d'un cluster (protégé)",
)
def cluster_games(
    cluster_id: int,
    sample: Optional[int] = Query(None, gt=0, le=500, description="Échantillon aléatoire (optionnel)"),
    current_user: User = Security(get_current_user, scopes=["clusters:read"]),
):
    if not model.is_trained:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas prêt.")
    try:
        items = model.get_cluster_games(cluster_id=cluster_id, sample=sample)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [ClusterGameItem(**g) for g in items]

# ============================================================
# Métriques (protégées)
# ============================================================
@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Métriques du modèle (protégé)",
)
def metrics(
    current_user: User = Security(get_current_user, scopes=["metrics:read"]),
):
    data = model.get_metrics()
    return MetricsResponse(**data)
