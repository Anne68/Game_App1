import time, uuid
from datetime import timedelta, datetime
from typing import Tuple
import jwt
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.models import Token, TokenRefreshRequest
from settings import Settings

router = APIRouter(prefix="/token", tags=["auth"])
_settings = Settings()

ALGORITHM = "HS256"

# mini store en mémoire pour la démonstration
REVOKED_JTIS = set()

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    access_expires_in: int

def _create_token(sub: str, scope: str, expires_delta: timedelta) -> Tuple[str, str, int]:
    jti = str(uuid.uuid4())
    now = int(time.time())
    exp = now + int(expires_delta.total_seconds())
    payload = {"sub": sub, "scope": scope, "jti": jti, "iat": now, "exp": exp}
    token = jwt.encode(payload, _settings.SECRET_KEY, algorithm=ALGORITHM)
    return token, jti, exp - now

def create_token_pair(username: str) -> TokenPair:
    access_token, _, acc_ttl = _create_token(username, "access", timedelta(minutes=_settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    refresh_token, jti, _ = _create_token(username, "refresh", timedelta(days=_settings.REFRESH_TOKEN_EXPIRE_DAYS))
    # Option : persister le jti en DB/Redis; ici on l’autorise implicitement
    return TokenPair(access_token=access_token, refresh_token=refresh_token, access_expires_in=acc_ttl)

@router.post("/refresh", response_model=Token)
def refresh_token(req: TokenRefreshRequest):
    try:
        payload = jwt.decode(req.refresh_token, _settings.SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("scope") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token scope")

    jti = payload.get("jti")
    if jti in REVOKED_JTIS:
        raise HTTPException(status_code=401, detail="Refresh token revoked")

    # rotation : révoquer l’ancien et émettre un nouveau couple
    REVOKED_JTIS.add(jti)
    username = payload.get("sub")
    pair = create_token_pair(username)
    return Token(access_token=pair.access_token, token_type="bearer", expires_in=pair.access_expires_in)
