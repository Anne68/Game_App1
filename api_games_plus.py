# api_games_plus.py
from __future__ import annotations

import os
import time
import logging
import inspect
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple

import pymysql
from fastapi import FastAPI, HTTPException, Depends, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from settings import get_settings
from model_manager import get_model
from monitoring_metrics import (
    prediction_latency,
    model_prediction_counter,
    get_monitor,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("games-api-ml")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")

# ---------------------------------------------------------------------
# Settings & Security
# ---------------------------------------------------------------------
settings = get_settings()

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Passlib : accepte anciens formats et produit du bcrypt
pwd_ctx = CryptContext(
    schemes=["bcrypt", "pbkdf2_sha256", "sha256_crypt"],
    deprecated="auto",
)

# ---------------------------------------------------------------------
# Flags DB
# ---------------------------------------------------------------------
DB_READY: bool = False
DB_LAST_ERROR: Optional[str] = None

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title="Games API ML (C9→C13)",
    version="2.4.0",
    description="API avec modèle ML, monitoring avancé et MySQL pour les utilisateurs",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in settings.ALLOW_ORIGINS.split(",") if o] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Models (Pydantic)
# ---------------------------------------------------------------------
class TrainRequest(BaseModel):
    version: Optional[str] = None
    force_retrain: bool = False

class PredictionRequest(BaseModel):
    query: str
    k: int = 10
    min_confidence: float = 0.1

class ModelMetricsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_version: str
    is_trained: bool
    total_predictions: int
    avg_confidence: float
    last_training: Optional[str]
    games_count: int
    feature_dimension: int

class TrainResponse(BaseModel):
    status: str
    version: Optional[str] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
def _ssl_kwargs() -> dict:
    if settings.DB_SSL_CA:
        return {"ssl": {"ca": settings.DB_SSL_CA}}
    return {"ssl": {}}

def get_db_conn():
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
        read_timeout=10,
        write_timeout=10,
        **_ssl_kwargs(),
    )

# ---------- USERS TABLE ----------
def ensure_users_table():
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'users';")
        if not cur.fetchone():
            cur.execute(
                """
                CREATE TABLE users (
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  username VARCHAR(190) UNIQUE NOT NULL,
                  hashed_password VARCHAR(255) NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            conn.commit()
            logger.info("Created users table with 'hashed_password'.")
            return

        cur.execute("SHOW COLUMNS FROM users LIKE 'hashed
