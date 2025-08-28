from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from app.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=settings.TOKEN_PATH)

def create_access_token(sub: str, expires_minutes: int = settings.ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = {
        "sub": sub,
        "exp": datetime.now(tz=timezone.utc) + timedelta(minutes=expires_minutes)
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload["sub"]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
