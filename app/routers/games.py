from fastapi import APIRouter, Depends
from app.auth import get_current_user
from app.schemas import PredictIn, PredictOut
from app.ml.predict import predict_one

router = APIRouter()

@router.post("/predict", response_model=PredictOut)
def predict(req: PredictIn, user: str = Depends(get_current_user)):
    return PredictOut(**predict_one(req.dict()))
