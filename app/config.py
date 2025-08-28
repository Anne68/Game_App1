from pydantic import BaseModel, Field, AnyUrl
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    SECRET_KEY: str = Field(..., min_length=32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    DB_URL: str

    API_PREFIX: str = ""
    TOKEN_PATH: str = "/token"
    ALLOW_ORIGINS: List[AnyUrl] = []

    PROMETHEUS_ENABLED: bool = True
    MLFLOW_TRACKING_URI: Optional[str] = "mlruns"

    class Config:
        env_file = ".env"

settings = Settings()
