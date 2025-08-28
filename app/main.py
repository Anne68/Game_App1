from fastapi import FastAPI, Depends, HTTPException, status, Form
from starlette.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from app.config import settings
from app.schemas import TokenOut
from app.auth import create_access_token
from app.routers import health, games

app = FastAPI(title="API Game", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(settings.TOKEN_PATH, response_model=TokenOut, tags=["auth"])
def token(username: str = Form(...), password: str = Form(...)):
    # ⚠️ Démo: remplacer par vérification depuis la DB (table users)
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing credentials")
    access_token = create_access_token(sub=username)
    return {"access_token": access_token, "token_type": "bearer"}

app.include_router(health.router, tags=["health"])
app.include_router(games.router, prefix="/games", tags=["ai"])

if settings.PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
