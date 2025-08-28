from fastapi import FastAPI, Depends, HTTPException, status, Form, APIRouter
from starlette.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import jwt
from datetime import datetime, timedelta, timezone
from settings import settings
from pydantic import BaseModel, Field, StrictStr, StrictFloat
import mlflow

# ---------- Auth ----------
def create_access_token(sub: str, expires_minutes: int = settings.ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = {"sub": sub, "exp": datetime.now(tz=timezone.utc) + timedelta(minutes=expires_minutes)}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

class TokenOut(BaseModel):
    access_token: StrictStr
    token_type: StrictStr = "bearer"

# ---------- Predict ----------
class PredictIn(BaseModel):
    title: StrictStr = Field(..., min_length=2, max_length=200)
    platform: StrictStr = Field(..., min_length=2, max_length=50)

class PredictOut(BaseModel):
    label: StrictStr
    score: StrictFloat

def predict_one(req: dict) -> dict:
    # Dummy prediction: based on length of title
    score = min(0.99, max(0.01, len(req["title"]) / 100.0))
    label = "recommended" if score >= 0.5 else "not_recommended"
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI or "mlruns")
    with mlflow.start_run(run_name="inference", nested=True):
        mlflow.log_metric("score", score)
        mlflow.log_param("len", len(req["title"]))
    return {"label": label, "score": float(score)}

# ---------- FastAPI ----------
app = FastAPI(title="API Game", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.ALLOW_ORIGINS],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.post(settings.TOKEN_PATH, response_model=TokenOut, tags=["auth"])
def token(username: str = Form(...), password: str = Form(...)):
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing credentials")
    return {"access_token": create_access_token(sub=username), "token_type": "bearer"}

router = APIRouter()

@router.get("/healthz")
def healthz(): return {"status": "ok"}

@router.get("/ready")
def ready(): return {"ready": True}

@router.post("/predict", response_model=PredictOut)
def predict(req: PredictIn, user: str = Depends(lambda: "demo")):
    return PredictOut(**predict_one(req.dict()))

app.include_router(router, prefix="/games", tags=["ai","health"])

if settings.PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
