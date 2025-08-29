# .env.example - Copier vers .env et configurer

# Database (AlwaysData)
DB_HOST=mysql-anne.alwaysdata.net
DB_PORT=3306
DB_USER=anne
DB_PASSWORD=@nnevi
DB_NAME=anne_games_db

# API Configuration
SECRET_KEY=your-secret-key-change-me-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
REFRESH_TOKEN_EXPIRE_MINUTES=10080

# Security
PASSWORD_MIN_LENGTH=8
PASSWORD_REGEX=^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$

# CORS
ALLOW_ORIGINS=*

# Logging
LOG_LEVEL=INFO

# Monitoring
PROMETHEUS_ENABLED=true

# Model Configuration
MODEL_VERSION=1.0.0
MODEL_PATH=model/recommendation_model.pkl
AUTO_TRAIN_ON_STARTUP=true
MIN_CONFIDENCE_THRESHOLD=0.1

# Render API URLs (Production)
PROD_API_URL=https://game-app-y8be.onrender.com
STAGING_API_URL=https://game-app-staging.onrender.com

# Streamlit
STREAMLIT_PORT=8501

# Grafana (Docker)
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin123

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
