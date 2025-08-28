from pydantic import BaseModel, Field, StrictStr, StrictFloat

class TokenOut(BaseModel):
    access_token: StrictStr
    token_type: StrictStr = "bearer"

class PredictIn(BaseModel):
    title: StrictStr = Field(..., min_length=2, max_length=200)
    platform: StrictStr = Field(..., min_length=2, max_length=50)

class PredictOut(BaseModel):
    label: StrictStr
    score: StrictFloat
