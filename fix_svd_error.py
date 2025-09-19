#!/usr/bin/env python3
"""
Correction de l'erreur SVD dans model_manager.py
"""

import re
from pathlib import Path

def fix_svd_dimensions():
    """Corrige les dimensions SVD pour éviter l'erreur n_components > n_features"""
    
    model_file = Path("model_manager.py")
    
    if not model_file.exists():
        print("❌ model_manager.py non trouvé")
        return False
    
    with open(model_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Correction principale : SVD adaptatif basé sur les features disponibles
    svd_fix = '''        # SVD adaptatif basé sur les features disponibles
        n_features = self.tfidf_matrix.shape[1]
        n_components = min(128, max(10, n_features - 1))  # Sécurisé
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)'''
    
    # Remplacer la ligne problématique
    patterns = [
        r'self\.svd = TruncatedSVD\(n_components=128, random_state=42\)',
        r'TruncatedSVD\(\s*n_components=128[^)]*\)'
    ]
    
    fixed = False
    for pattern in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, svd_fix.strip(), content)
            fixed = True
            print(f"✅ Corrigé pattern SVD: {pattern}")
    
    # Correction supplémentaire : ajouter validation dans train()
    train_method_fix = '''    def train(self, games: List[Dict]) -> Dict:
        logger.info(f"Training model v{self.model_version} with {len(games)} games")

        self.games_df = self.prepare_features(games)

        # TF-IDF contenu -> SVD ADAPTATIF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.games_df["combined_features"])
        
        # SVD adaptatif pour éviter l'erreur n_components > n_features
        n_features = self.tfidf_matrix.shape[1]
        n_components = min(128, max(10, n_features - 1))
        logger.info(f"Using {n_components} SVD components for {n_features} features")
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.features_reduced = self.svd.fit_transform(self.tfidf_matrix)'''
    
    # Remplacer méthode train si elle existe avec le pattern problématique
    train_pattern = r'def train\(self, games: List\[Dict\]\) -> Dict:(.*?)self\.features_reduced = self\.svd\.fit_transform\(self\.tfidf_matrix\)'
    
    if re.search(train_pattern, content, re.DOTALL):
        # Correction complexe de la méthode train complète
        content = re.sub(
            r'(def train\(self, games: List\[Dict\]\) -> Dict:.*?)(self\.tfidf_matrix = self\.vectorizer\.fit_transform\(self\.games_df\["combined_features"\]\))(.*?)(self\.features_reduced = self\.svd\.fit_transform\(self\.tfidf_matrix\))',
            lambda m: f'''{m.group(1)}{m.group(2)}
        
        # SVD adaptatif pour éviter l'erreur n_components > n_features
        n_features = self.tfidf_matrix.shape[1]
        n_components = min(128, max(10, n_features - 1))
        logger.info(f"Using {{n_components}} SVD components for {{n_features}} features")
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.features_reduced = self.svd.fit_transform(self.tfidf_matrix)''',
            content,
            flags=re.DOTALL
        )
        fixed = True
        print("✅ Méthode train() corrigée")
    
    if fixed:
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ model_manager.py corrigé")
        return True
    else:
        print("⚠️ Pattern SVD non trouvé - correction manuelle nécessaire")
        return False

def create_backup_model_manager():
    """Crée une version de secours avec SVD adaptatif"""
    
    backup_content = '''# model_manager.py - Version corrigée avec SVD adaptatif
from __future__ import annotations

import os
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

logger = logging.getLogger("model-manager")

class RecommendationModel:
    """Modèle de recommandation avec SVD adaptatif"""

    def __init__(self, model_version: str = "2.0.1-fixed"):
        self.model_version = model_version

        # Vectoriseurs
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Réduit pour éviter les problèmes
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        
        # SVD sera initialisé dynamiquement
        self.svd = None
        
        # Clustering et KNN
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")  # Réduit clusters
        self.knn = NearestNeighbors(metric="cosine", n_neighbors=20, algorithm="auto")

        # État
        self.is_trained: bool = False
        self.games_df: Optional[pd.DataFrame] = None
        self.game_features: Optional[np.ndarray] = None
        self.features_reduced: Optional[np.ndarray] = None
        self.tfidf_matrix: Optional[np.ndarray] = None

    def train(self, games: List[Dict]) -> Dict:
        """Entraînement avec SVD adaptatif"""
        logger.info(f"Training model v{self.model_version} with {len(games)} games")

        # Préparer les données
        df = pd.DataFrame(games)
        df["platforms_str"] = df["platforms"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x or "")
        )
        df["combined_features"] = (
            df["title"].fillna("") + " " +
            df["genres"].fillna("") + " " +
            df["platforms_str"].fillna("")
        )
        
        # Normalisation numérique
        df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(4.0)
        df["metacritic"] = pd.to_numeric(df.get("metacritic", 0), errors="coerce").fillna(80)
        
        # Normalisation 0-1
        df["rating_norm"] = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min() + 1e-8)
        df["metacritic_norm"] = (df["metacritic"] - df["metacritic"].min()) / (df["metacritic"].max() - df["metacritic"].min() + 1e-8)
        
        self.games_df = df

        # TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(df["combined_features"])
        
        # SVD ADAPTATIF - CORRECTION PRINCIPALE
        n_features = self.tfidf_matrix.shape[1]
        n_components = min(50, max(5, n_features - 1))  # Sécurisé et réduit
        
        logger.info(f"SVD: {n_components} components for {n_features} features")
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.features_reduced = self.svd.fit_transform(self.tfidf_matrix)

        # Features finales
        numeric = df[["rating_norm", "metacritic_norm"]].values
        self.game_features = np.hstack([self.features_reduced, numeric])

        # Clustering et KNN
        self.cluster_labels = self.kmeans.fit_predict(self.features_reduced)
        self.knn.fit(self.game_features)

        self.is_trained = True
        
        return {
            "status": "success",
            "model_version": self.model_version,
            "games_count": len(games),
            "n_features": n_features,
            "n_components": n_components,
            "feature_dim": self.game_features.shape[1]
        }

    def predict(self, query: str, k: int = 10, min_confidence: float = 0.1) -> List[Dict]:
        """Prédictions avec gestion d'erreur robuste"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        try:
            # Projeter la requête
            q_tfidf = self.vectorizer.transform([query])
            q_reduced = self.svd.transform(q_tfidf)
            
            # Features numériques moyennes
            avg_rating = self.games_df["rating_norm"].mean()
            avg_metacritic = self.games_df["metacritic_norm"].mean()
            q_features = np.hstack([q_reduced[0], [avg_rating, avg_metacritic]])

            # KNN
            distances, indices = self.knn.kneighbors([q_features], n_neighbors=min(k * 2, len(self.games_df)))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if len(results) >= k:
                    break
                    
                similarity = 1.0 - distances[0][i]
                if similarity < min_confidence:
                    continue
                    
                row = self.games_df.iloc[idx]
                results.append({
                    "id": int(row["id"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "confidence": float(similarity),
                    "rating": float(row["rating"]),
                    "metacritic": int(row["metacritic"])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return []

    def get_metrics(self) -> Dict:
        """Métriques du modèle"""
        return {
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "games_count": len(self.games_df) if self.games_df is not None else 0,
            "feature_dim": self.game_features.shape[1] if self.game_features is not None else 0
        }

# Singleton
_model_instance: Optional[RecommendationModel] = None

def get_model() -> RecommendationModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = RecommendationModel()
    return _model_instance

def reset_model():
    global _model_instance
    _model_instance = None
'''
    
    with open("model_manager_backup.py", "w", encoding="utf-8") as f:
        f.write(backup_content)
    
    print("✅ model_manager_backup.py créé")

def main():
    print("🔧 CORRECTION ERREUR SVD RENDER")
    print("=" * 35)
    
    print("Problème détecté: ValueError: n_components(128) must be <= n_features(48)")
    print("Solution: SVD adaptatif basé sur le nombre de features disponibles")
    print()
    
    # Tenter correction directe
    if fix_svd_dimensions():
        print("✅ Correction appliquée avec succès")
    else:
        print("⚠️ Correction directe échouée - création backup")
        create_backup_model_manager()
        print("📋 Actions manuelles requises:")
        print("1. Remplacer model_manager.py par model_manager_backup.py")
        print("2. Ou appliquer les corrections SVD manuellement")
    
    print("\n📋 Prochaines étapes:")
    print("1. git add model_manager.py")
    print("2. git commit -m 'fix: SVD adaptatif pour éviter erreur dimensions'")
    print("3. git push")
    print("4. L'API Render devrait maintenant démarrer correctement")

if __name__ == "__main__":
    main()
