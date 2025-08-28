# settings.py
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========= Database =========
    DB_HOST: str
    DB_PORT: int = 3306            # AlwaysData par défaut (ton local Docker était 3307)
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_SSL_CA: Optional[str] = None  # facultatif

    # ========= API / CORS =========
    ALLOW_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"

    # ========= Auth / JWT =========
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # ========= Security =========
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REGEX: str = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$"

    # ========= Monitoring =========
    PROMETHEUS_ENABLED: bool = True

    @field_validator("ALLOW_ORIGINS", mode="before")
    @classmethod
    def _normalize_origins(cls, v):
        if v is None:
            return "*"
        if isinstance(v, (list, tuple)):
            return ",".join(s.strip() for s in v if s and str(s).strip())
        s = str(v)
        parts = [p.strip() for p in s.split(",")]
        parts = [p for p in parts if p]
        return ",".join(parts) if parts else "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
