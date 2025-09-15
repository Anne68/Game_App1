# model_manager.py - Gestion du modèle de recommandation avec ML
from __future__ import annotations

import os
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pandas as pd

logger = logging.getLogger("model-manager")

class RecommendationModel:
    """Modèle de recommandation basé sur TF-IDF et similarité cosinus"""
    
    def __init__(self, model_version: str = "1.0.0"):
        self.model_version = model_version
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.svd = TruncatedSVD(n_components=20, random_state=42)
        self.game_features = None
        self.games_df = None
        self.is_trained = False
        
        # Métriques pour monitoring
        self.metrics = {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "last_training": None,
            "model_version": model_version,
            "feature_dim": 0
        }
    
    def prepare_features(self, games: List[Dict]) -> pd.DataFrame:
        """Prépare les features pour l'entraînement"""
        df = pd.DataFrame(games)
        
        # Combiner titre, genres et platforms pour créer un corpus
        df['combined_features'] = (
            df['title'] + ' ' + 
            df['genres'] + ' ' + 
            df['platforms'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        )
        
        # Ajouter des features numériques normalisées
        df['rating_norm'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
        df['metacritic_norm'] = (df['metacritic'] - df['metacritic'].min()) / (df['metacritic'].max() - df['metacritic'].min())
        
        return df
    
    def train(self, games: List[Dict]) -> Dict:
        """Entraîne le modèle de recommandation"""
        logger.info(f"Training model v{self.model_version} with {len(games)} games")
        
        # Préparer les données
        self.games_df = self.prepare_features(games)
        
        # Vectorisation TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(self.games_df['combined_features'])
        
        # Réduction de dimension avec SVD
        tfidf_reduced = self.svd.fit_transform(tfidf_matrix)
        
        # Combiner features textuelles et numériques
        numeric_features = self.games_df[['rating_norm', 'metacritic_norm']].values
        self.game_features = np.hstack([tfidf_reduced, numeric_features])
        
        self.is_trained = True
        self.metrics["last_training"] = datetime.utcnow().isoformat()
        self.metrics["feature_dim"] = self.game_features.shape[1]
        
        # Calculer des métriques de validation
        validation_metrics = self._validate_model()
        
        logger.info(f"Model trained successfully. Feature dimension: {self.game_features.shape}")
        return {
            "status": "success",
            "model_version": self.model_version,
            "games_count": len(games),
            "feature_dim": self.game_features.shape[1],
            "validation_metrics": validation_metrics
        }
    
    def _validate_model(self) -> Dict:
        """Validation interne du modèle"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        # Calculer la cohérence des similarités
        sample_size = min(10, len(self.games_df))
        sample_indices = np.random.choice(len(self.games_df), sample_size, replace=False)
        
        similarities = []
        for idx in sample_indices:
            sims = cosine_similarity([self.game_features[idx]], self.game_features)[0]
            # Exclure la similarité avec soi-même
            sims = np.delete(sims, idx)
            similarities.extend(sims)
        
        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities))
        }
    
    def predict(self, query: str, k: int = 10, min_confidence: float = 0.1) -> List[Dict]:
        """Génère des recommandations basées sur une requête"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Vectoriser la requête
        query_vec = self.vectorizer.transform([query])
        query_reduced = self.svd.transform(query_vec)
        
        # Ajouter des features numériques moyennes pour la requête
        avg_numeric = np.mean(self.games_df[['rating_norm', 'metacritic_norm']].values, axis=0)
        query_features = np.hstack([query_reduced[0], avg_numeric])
        
        # Calculer les similarités
        similarities = cosine_similarity([query_features], self.game_features)[0]
        
        # Créer les recommandations
        recommendations = []
        for idx, score in enumerate(similarities):
            if score >= min_confidence:
                recommendations.append({
                    "title": self.games_df.iloc[idx]['title'],
                    "genres": self.games_df.iloc[idx]['genres'],
                    "confidence": float(score),
                    "rating": float(self.games_df.iloc[idx]['rating']),
                    "metacritic": int(self.games_df.iloc[idx]['metacritic'])
                })
        
        # Trier par score de confiance
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Mise à jour des métriques
        self.metrics["total_predictions"] += 1
        if recommendations:
            avg_conf = np.mean([r['confidence'] for r in recommendations[:k]])
            self.metrics["avg_confidence"] = (
                (self.metrics["avg_confidence"] * (self.metrics["total_predictions"] - 1) + avg_conf) 
                / self.metrics["total_predictions"]
            )
        
        return recommendations[:k]
    
    def predict_by_game_id(self, game_id: int, k: int = 10) -> List[Dict]:
        """Recommandations basées sur un jeu existant"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Trouver l'index du jeu
        game_idx = self.games_df[self.games_df['id'] == game_id].index
        if len(game_idx) == 0:
            raise ValueError(f"Game with id {game_id} not found")
        
        game_idx = game_idx[0]
        
        # Calculer les similarités
        similarities = cosine_similarity([self.game_features[game_idx]], self.game_features)[0]
        
        # Exclure le jeu lui-même
        similarities[game_idx] = -1
        
        # Top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:
                recommendations.append({
                    "title": self.games_df.iloc[idx]['title'],
                    "genres": self.games_df.iloc[idx]['genres'],
                    "confidence": float(similarities[idx]),
                    "rating": float(self.games_df.iloc[idx]['rating']),
                    "metacritic": int(self.games_df.iloc[idx]['metacritic'])
                })
        
        return recommendations
    
    def save_model(self, filepath: str = "model/recommendation_model.pkl") -> bool:
        """Sauvegarde le modèle entraîné"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                "version": self.model_version,
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "game_features": self.game_features,
                "games_df": self.games_df,
                "metrics": self.metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str = "model/recommendation_model.pkl") -> bool:
        """Charge un modèle pré-entraîné"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_version = model_data["version"]
            self.vectorizer = model_data["vectorizer"]
            self.svd = model_data["svd"]
            self.game_features = model_data["game_features"]
            self.games_df = model_data["games_df"]
            self.metrics = model_data["metrics"]
            self.is_trained = True
            
            logger.info(f"Model v{self.model_version} loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_metrics(self) -> Dict:
        """Retourne les métriques du modèle pour le monitoring"""
        return {
            **self.metrics,
            "is_trained": self.is_trained,
            "games_count": len(self.games_df) if self.games_df is not None else 0
        }


# Singleton pour gérer l'instance du modèle
_model_instance: Optional[RecommendationModel] = None

def get_model() -> RecommendationModel:
    """Retourne l'instance singleton du modèle"""
    global _model_instance
    if _model_instance is None:
        _model_instance = RecommendationModel()
        # Essayer de charger un modèle existant
        if os.path.exists("model/recommendation_model.pkl"):
            _model_instance.load_model()
    return _model_instance
