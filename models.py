from typing import List, Optional
from pydantic import BaseModel, Field, constr, conint

# -------- Auth --------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenRefreshRequest(BaseModel):
    refresh_token: str

class UserCreate(BaseModel):
    username: constr(min_length=3, max_length=50)
    password: constr(min_length=8)

class UserOut(BaseModel):
    id: int
    username: str

# -------- Domaine --------
class PriceOut(BaseModel):
    platform: str
    price: float

class GameOut(BaseModel):
    id: int
    title: str
    genre: Optional[str] = None
    platform: Optional[str] = None
    release_year: Optional[int] = None
    rating: Optional[float] = None

class RecommendRequest(BaseModel):
    title: constr(min_length=1)
    k: conint(ge=1, le=50) = 5

class RecommendItem(BaseModel):
    title: str
    score: float = Field(ge=0.0)

class RecommendResponse(BaseModel):
    query: str
    k: int
    items: List[RecommendItem]
