# Architecture Technique - Application IA

## 1. Vue d'ensemble de l'architecture

### 1.1 Architecture applicative multicouche
```
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE PRÉSENTATION                     │
├─────────────────────────────────────────────────────────────┤
│ • Interface Streamlit (Port 8501)                          │
│ • API REST FastAPI (Port 8000)                             │
│ • Documentation Swagger/OpenAPI                            │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE MÉTIER IA                        │
├─────────────────────────────────────────────────────────────┤
│ • Modèle TF-IDF + SVD (model_manager.py)                   │
│ • Pipeline ML (entraînement/prédiction)                    │
│ • Monitoring métier (drift detection)                      │
│ • Gestion versions modèles                                 │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  COUCHE SERVICES                           │
├─────────────────────────────────────────────────────────────┤
│ • Service authentification JWT                             │
│ • Service recommandation                                   │
│ • Service monitoring (Prometheus)                          │
│ • Service recherche/cache                                  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   COUCHE DONNÉES                           │
├─────────────────────────────────────────────────────────────┤
│ • Base MySQL (utilisateurs, jeux, prix)                    │
│ • Stockage modèles (fichiers .pkl)                         │
│ • Métriques Prometheus/Grafana                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. Choix techniques ML justifiés

### 2.1 Pipeline de traitement
```python
# 1. Feature Engineering
combined_features = title + genres + platforms

# 2. Vectorisation TF-IDF
tfidf_matrix = TfidfVectorizer(
    max_features=100,        # Dimensionnalité contrôlée
    ngram_range=(1, 2),      # Uni + bigrammes
    stop_words='english'     # Filtrage bruit
)

# 3. Réduction dimensionnelle
svd = TruncatedSVD(n_components=20)  # Compression + débruitage

# 4. Features numériques normalisées
numeric_features = [rating_norm, metacritic_norm]

# 5. Similarité cosinus pour recommandations
similarities = cosine_similarity(query_vector, game_matrix)
```

### 2.2 Justifications algorithmiques
- **TF-IDF** : Pondération efficace des termes rares/fréquents
- **SVD** : Réduction bruit + accélération calculs
- **Cosinus** : Métrique adaptée aux vecteurs sparse
- **Normalisation** : Égalisation des échelles numériques

## 3. Architecture MLOps

### 3.1 Cycle de vie du modèle
```
Data → Feature Engineering → Training → Validation → Deployment
  ↑                                                        ↓
Monitoring ← Production ← Testing ← Versioning ← Packaging
```

## 4. Scalabilité et performance

### 4.1 Métriques de performance cibles
- **Latence P95** : < 500ms
- **Throughput** : > 100 req/s
- **Memory footprint** : < 512MB par instance
- **CPU usage** : < 80% en moyenne

Cette architecture garantit **maintenabilité**, **scalabilité** et **observabilité** pour un système IA en production.
