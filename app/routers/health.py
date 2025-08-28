from fastapi import APIRouter

router = APIRouter()

@router.get("/healthz")
def healthz():
    return {"status": "ok"}

@router.get("/ready")
def ready():
    return {"ready": True}
