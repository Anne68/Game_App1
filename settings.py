# settings.py (extrait utile)
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # DB
    DB_HOST: str
    DB_PORT: int = 3307
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    # API
    ALLOW_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # Security
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REGEX: str = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$"

    # Monitoring
    PROMETHEUS_ENABLED: bool = True

def get_settings():
    return Settings()
