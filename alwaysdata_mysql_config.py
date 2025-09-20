#!/usr/bin/env python3
"""
CONFIGURATION MYSQL ALWAYSDATA POUR API GAMES

Configuration complète pour connecter l'API à la base de données
MySQL hébergée sur AlwaysData avec SSL et optimisations.
"""

import os
import pymysql
import logging
from pathlib import Path

class AlwaysDataConfig:
    """Configuration pour base de données AlwaysData"""
    
    def __init__(self):
        self.connection_params = {
            "host": "mysql-anne.alwaysdata.net",
            "port": 3306,
            "user": "anne",
            "password": "Vicky@18",  # À changer dans les variables d'environnement
            "database": "anne_games_db",
            "charset": "utf8mb4",
            "autocommit": True,
            "connect_timeout": 10,
            "read_timeout": 30,
            "write_timeout": 30,
            # SSL pour sécurité (AlwaysData supporte SSL)
            "ssl_disabled": False,
            "ssl_verify_cert": False,
            "ssl_verify_identity": False
        }
    
    def create_env_file(self):
        """Crée le fichier .env avec configuration AlwaysData"""
        env_content = '''# Configuration pour base de données AlwaysData
# IMPORTANT: Utilisez ces valeurs dans Render Environment Variables

# Base de données MySQL AlwaysData
DB_HOST=mysql-anne.alwaysdata.net
DB_PORT=3306
DB_USER=anne
DB_PASSWORD=Vicky@18
DB_NAME=anne_games_db
DB_REQUIRED=true

# SSL Configuration (AlwaysData)
DB_SSL_DISABLED=false
DB_SSL_VERIFY_CERT=false

# API Configuration
SECRET_KEY=votre-cle-secrete-tres-longue-minimum-32-caracteres-pour-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480

# Mode production
DEMO_LOGIN_ENABLED=true
DEMO_USERNAME=demo
DEMO_PASSWORD=demo123

# Logging et CORS
LOG_LEVEL=INFO
ALLOW_ORIGINS=*

# Connection Pool Settings
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
'''
        
        with open(".env.alwaysdata", "w") as f:
            f.write(env_content)
        
        print("✅ Fichier .env.alwaysdata créé")
        return True
    
    def test_connection(self):
        """Test la connexion à AlwaysData"""
        print("🔌 Test de connexion à AlwaysData...")
        
        try:
            connection = pymysql.connect(**self.connection_params)
            
            with connection.cursor() as cursor:
                cursor.execute("SELECT VERSION()")
                version = cursor.fetchone()
                print(f"✅ Connexion réussie - MySQL {version[0]}")
                
                # Test des tables
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                print(f"📊 Tables disponibles: {len(tables)}")
                for table in tables:
                    print(f"   • {table[0]}")
                
                # Test données games
                cursor.execute("SELECT COUNT(*) FROM games")
                count = cursor.fetchone()
                print(f"🎮 Jeux dans la base: {count[0]}")
                
            connection.close()
            return True
            
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            return False
    
    def create_optimized_settings(self):
        """Crée settings.py optimisé pour AlwaysData"""
        settings_content = '''# settings.py - Configuration optimisée pour AlwaysData MySQL
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

    # ========= Database AlwaysData =========
    DB_REQUIRED: bool = Field(default=True)  # Obligatoire pour AlwaysData
    DB_HOST: str = Field(default="mysql-anne.alwaysdata.net")
    DB_PORT: int = Field(default=3306)
    DB_USER: str = Field(default="anne")
    DB_PASSWORD: str = Field(default="")  # À définir via ENV
    DB_NAME: str = Field(default="anne_games_db")
    
    # SSL Configuration pour AlwaysData
    DB_SSL_DISABLED: bool = Field(default=False)
    DB_SSL_VERIFY_CERT: bool = Field(default=False)
    DB_SSL_VERIFY_IDENTITY: bool = Field(default=False)
    
    # Connection Pool Settings
    DB_POOL_SIZE: int = Field(default=5)
    DB_MAX_OVERFLOW: int = Field(default=10)
    DB_POOL_TIMEOUT: int = Field(default=30)
    DB_POOL_RECYCLE: int = Field(default=3600)
    
    # Connection Timeouts
    DB_CONNECT_TIMEOUT: int = Field(default=10)
    DB_READ_TIMEOUT: int = Field(default=30)
    DB_WRITE_TIMEOUT: int = Field(default=30)

    # ========= API / CORS =========
    ALLOW_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"

    # ========= Auth / JWT =========
    SECRET_KEY: str = Field(default="change-me-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # ========= Demo Mode =========
    DEMO_LOGIN_ENABLED: bool = Field(default=True)
    DEMO_USERNAME: str = "demo"
    DEMO_PASSWORD: str = "demo123"

    # ========= ML Settings =========
    MODEL_VERSION: str = Field(default="2.0.0")
    AUTO_TRAIN_ON_STARTUP: bool = Field(default=False)  # Éviter auto-train en prod
    
    @property
    def db_configured(self) -> bool:
        """Vérifie si la DB est configurée"""
        return all([
            self.DB_HOST, 
            self.DB_USER, 
            self.DB_PASSWORD, 
            self.DB_NAME
        ])
    
    @property
    def mysql_url(self) -> str:
        """URL MySQL pour SQLAlchemy si nécessaire"""
        if not self.db_configured:
            return ""
        
        return (
            f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            f"?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
'''
        
        with open("settings.py", "w") as f:
            f.write(settings_content)
        
        print("✅ settings.py optimisé pour AlwaysData créé")
        return True
    
    def update_api_for_alwaysdata(self):
        """Met à jour api_games_plus.py pour AlwaysData"""
        
        # Lire le fichier actuel
        with open("api_games_plus.py", "r") as f:
            content = f.read()
        
        # Fonction de connexion optimisée pour AlwaysData
        db_function = '''
def get_db_conn():
    """Connexion optimisée pour AlwaysData MySQL"""
    if not settings.db_configured:
        raise RuntimeError("Database not configured (missing DB_* env vars)")
    
    connection_params = {
        "host": settings.DB_HOST,
        "port": settings.DB_PORT,
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
        "database": settings.DB_NAME,
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": True,
        "connect_timeout": settings.DB_CONNECT_TIMEOUT,
        "read_timeout": settings.DB_READ_TIMEOUT,
        "write_timeout": settings.DB_WRITE_TIMEOUT,
    }
    
    # SSL Configuration pour AlwaysData
    if not settings.DB_SSL_DISABLED:
        ssl_context = {}
        if not settings.DB_SSL_VERIFY_CERT:
            ssl_context["check_hostname"] = False
            ssl_context["verify_mode"] = 0  # ssl.CERT_NONE equivalent
        connection_params["ssl"] = ssl_context
    
    try:
        return pymysql.connect(**connection_params)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
'''
        
        # Remplacer la fonction get_db_conn existante
        import re
        
        # Pattern pour trouver la fonction existante
        pattern = r'def get_db_conn\(\):.*?(?=\ndef|\nclass|\n@|\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, db_function.strip(), content, flags=re.DOTALL)
        else:
            # Ajouter la fonction si elle n'existe pas
            content = content.replace(
                "from monitoring_metrics import",
                f"{db_function}\n\nfrom monitoring_metrics import"
            )
        
        # Mise à jour de la fonction startup pour AlwaysData
        startup_update = '''
@app.on_event("startup")
async def startup_event():
    global DB_READY, DB_LAST_ERROR
    logger.info("Starting Games API with AlwaysData MySQL...")

    try:
        if settings.db_configured and settings.DB_REQUIRED:
            # Test connexion AlwaysData
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT VERSION()")
                version = cur.fetchone()
                logger.info(f"Connected to AlwaysData MySQL {version['VERSION()']}")
                
                # Assurer les tables existent
                ensure_users_table()
                
                # Compter les jeux
                cur.execute("SELECT COUNT(*) as count FROM games")
                game_count = cur.fetchone()
                logger.info(f"Games in database: {game_count['count']}")
            
            conn.close()
            DB_READY = True
            DB_LAST_ERROR = None
            logger.info("AlwaysData database ready")
            
        elif settings.db_configured:
            # Mode optionnel
            try:
                conn = get_db_conn()
                ensure_users_table()
                conn.close()
                DB_READY = True
                logger.info("AlwaysData database connected (optional mode)")
            except Exception as e:
                DB_READY = False
                DB_LAST_ERROR = str(e)
                logger.warning(f"AlwaysData connection failed (continuing): {e}")
        else:
            DB_READY = False
            DB_LAST_ERROR = "Database not configured"
            logger.info("Database not configured")
    
    except Exception as e:
        DB_READY = False
        DB_LAST_ERROR = str(e)
        logger.error(f"AlwaysData initialization failed: {e}")
        if settings.DB_REQUIRED:
            logger.error("DB_REQUIRED=true but database failed")
            raise

    # Charger modèle existant
    model = get_model()
    model_path = os.path.join("model", "recommendation_model.pkl")
    try:
        if os.path.exists(model_path):
            if model.load_model():
                logger.info("Model loaded: %s", model.model_version)
            else:
                logger.info("Model loading failed - will train on request")
        else:
            logger.info("No saved model - will train on first request")
    except Exception as e:
        logger.warning("Model loading error: %s", e)

    logger.info("API startup complete")
'''
        
        # Remplacer la fonction startup existante
        startup_pattern = r'@app\.on_event\("startup"\)\nasync def startup_event\(\):.*?(?=\n\n@|\n\nif|\napp\.|\Z)'
        
        if re.search(startup_pattern, content, re.DOTALL):
            content = re.sub(startup_pattern, startup_update.strip(), content, flags=re.DOTALL)
        
        # Écrire le fichier mis à jour
        with open("api_games_plus.py", "w") as f:
            f.write(content)
        
        print("✅ api_games_plus.py mis à jour pour AlwaysData")
        return True
    
    def create_database_schema(self):
        """Crée le schéma SQL pour AlwaysData"""
        
        sql_content = '''-- Schema SQL pour AlwaysData MySQL
-- Base de données: anne_games_db

-- Table des utilisateurs
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(190) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    email VARCHAR(255) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    INDEX idx_username (username),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table des jeux (structure existante)
CREATE TABLE IF NOT EXISTS games (
    id INT AUTO_INCREMENT PRIMARY KEY,
    game_id_rawg INT UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    genres TEXT,
    rating DECIMAL(3,2) DEFAULT 0.00,
    metacritic INT DEFAULT 0,
    platforms TEXT,
    release_date DATE NULL,
    description TEXT,
    image_url VARCHAR(1000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_game_id_rawg (game_id_rawg),
    INDEX idx_title (title(100)),
    INDEX idx_rating (rating),
    INDEX idx_metacritic (metacritic),
    FULLTEXT(title, genres, description)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table des recherches utilisateur
CREATE TABLE IF NOT EXISTS user_game_searches (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    search_query VARCHAR(500) NOT NULL,
    results_count INT DEFAULT 0,
    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_ip VARCHAR(45),
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_search_timestamp (search_timestamp),
    INDEX idx_search_query (search_query(100))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table des recommandations (pour tracking)
CREATE TABLE IF NOT EXISTS user_recommendations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    query_text VARCHAR(500),
    recommended_game_id INT,
    confidence_score DECIMAL(5,4) DEFAULT 0.0000,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (recommended_game_id) REFERENCES games(game_id_rawg),
    INDEX idx_user_id (user_id),
    INDEX idx_recommended_game_id (recommended_game_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Vues utiles
CREATE OR REPLACE VIEW popular_games AS
SELECT 
    g.*,
    COUNT(ur.id) as recommendation_count
FROM games g
LEFT JOIN user_recommendations ur ON g.game_id_rawg = ur.recommended_game_id
GROUP BY g.id
ORDER BY recommendation_count DESC, g.rating DESC;

CREATE OR REPLACE VIEW user_stats AS
SELECT 
    u.id,
    u.username,
    COUNT(DISTINCT us.id) as search_count,
    COUNT(DISTINCT ur.id) as recommendations_received,
    u.created_at
FROM users u
LEFT JOIN user_game_searches us ON u.id = us.user_id
LEFT JOIN user_recommendations ur ON u.id = ur.user_id
GROUP BY u.id;

-- Procédure pour nettoyer les anciennes données
DELIMITER //
CREATE PROCEDURE CleanOldData()
BEGIN
    -- Supprimer les recherches de plus de 6 mois
    DELETE FROM user_game_searches 
    WHERE search_timestamp < DATE_SUB(NOW(), INTERVAL 6 MONTH);
    
    -- Supprimer les recommandations de plus de 3 mois
    DELETE FROM user_recommendations 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL 3 MONTH);
END //
DELIMITER ;

-- Event pour nettoyage automatique (si autorisé)
-- CREATE EVENT IF NOT EXISTS cleanup_event
-- ON SCHEDULE EVERY 1 WEEK
-- DO CALL CleanOldData();
'''
        
        # Créer le dossier sql s'il n'existe pas
        sql_dir = Path("sql")
        sql_dir.mkdir(exist_ok=True)
        
        with open(sql_dir / "alwaysdata_schema.sql", "w") as f:
            f.write(sql_content)
        
        print("✅ Schéma SQL créé: sql/alwaysdata_schema.sql")
        return True
    
    def create_requirements_update(self):
        """Met à jour requirements avec optimisations MySQL"""
        
        requirements_content = '''# API Core - Optimisé pour AlwaysData MySQL
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Auth & Security
passlib==1.7.4
bcrypt==4.1.2
python-jose[cryptography]==3.3.0

# Database MySQL - AlwaysData optimisé
pymysql==1.1.0
cryptography>=3.4.8  # Pour SSL MySQL
python-dotenv==1.0.0
pydantic-settings==2.1.0

# ML & Data Processing
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0
scipy>=1.10.0,<2.0.0

# Monitoring
prometheus-client==0.19.0
prometheus-fastapi-instrumentator==6.1.0

# Security & Utils
bleach==6.1.0
psutil==5.9.0

# Build
setuptools>=68.0.0
wheel>=0.41.0

# Connection Pool (optionnel pour optimisation)
sqlalchemy>=2.0.0  # Si on veut utiliser SQLAlchemy plus tard
'''
        
        with open("requirements_api.txt", "w") as f:
            f.write(requirements_content)
        
        print("✅ requirements_api.txt mis à jour pour AlwaysData")
        return True


def main():
    """Configuration complète pour AlwaysData"""
    print("🚀 CONFIGURATION ALWAYSDATA MYSQL")
    print("=" * 40)
    
    config = AlwaysDataConfig()
    
    steps = [
        ("1. Test connexion AlwaysData", config.test_connection),
        ("2. Création .env AlwaysData", config.create_env_file),
        ("3. Settings optimisé", config.create_optimized_settings),
        ("4. API pour AlwaysData", config.update_api_for_alwaysdata),
        ("5. Schéma SQL", config.create_database_schema),
        ("6. Requirements MySQL", config.create_requirements_update)
    ]
    
    success_count = 0
    
    for name, step_func in steps:
        print(f"\n📋 {name}...")
        try:
            if step_func():
                success_count += 1
        except Exception as e:
            print(f"❌ Erreur {name}: {e}")
    
    print(f"\n🎯 RÉSULTATS: {success_count}/{len(steps)} étapes réussies")
    
    if success_count >= 4:
        print("\n✅ CONFIGURATION ALWAYSDATA TERMINÉE!")
        print("\n📋 PROCHAINES ÉTAPES:")
        print("1. Copiez les variables de .env.alwaysdata vers Render:")
        print("   - DB_HOST=mysql-anne.alwaysdata.net")
        print("   - DB_PORT=3306")
        print("   - DB_USER=anne")
        print("   - DB_PASSWORD=Alex,Anne69360")
        print("   - DB_NAME=anne_games_db")
        print("   - DB_REQUIRED=true")
        print()
        print("2. Committez et déployez:")
        print("   git add .")
        print("   git commit -m 'feat: Configure AlwaysData MySQL database'")
        print("   git push")
        print()
        print("3. L'API utilisera maintenant AlwaysData!")
        return True
    else:
        print("\n⚠️ Configuration incomplète")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
