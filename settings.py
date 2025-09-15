# settings.py
from __future__ import annotations

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ========= Database =========
    DB_REQUIRED: bool = Field(default=False)  # si True: on exige la DB au démarrage
    DB_HOST: Optional[str] = None
    DB_PORT: int = 3306
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_NAME: Optional[str] = None
    DB_SSL_CA: Optional[str] = None  # facultatif (TLS vérifié si fourni)

    # ========= API / CORS =========
    ALLOW_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"

    # ========= Auth / JWT =========
    SECRET_KEY: str = Field(default="dev-secret-key")  # en prod: fournir via ENV
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # ========= Security =========
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REGEX: str = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$"

    # ========= Monitoring =========
    PROMETHEUS_ENABLED: bool = True

    # ========= Dev fallback (optionnel) =========
    DEMO_LOGIN_ENABLED: bool = False
    DEMO_USERNAME: str = "demo"
    DEMO_PASSWORD: str = "demo123!"

    @property
    def db_configured(self) -> bool:
        return all([self.DB_HOST, self.DB_USER, self.DB_PASSWORD, self.DB_NAME])


@lru_cache
def get_settings() -> Settings:
    return Settings()
