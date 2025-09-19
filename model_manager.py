# model_manager.py - Simplifié avec TF-IDF + SVD + KMeans + KNN
from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("model-manager")

class SimpleRecommendationModel:
    """Modèle de recommandation simplifié avec TF-IDF + SVD + KMeans + KNN"""
    
    def __init__(self, model_version: str = "1.0.0"):
        self.model_version = model_version
        self.is_trained = False
        
        # Composants ML
        self.tfidf = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        self.knn = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.scaler = StandardScaler()
        
        # Données
        self.games_df = None
        self.features_matrix = None
        self.tfidf_matrix = None
        self.svd_features = None
        self.clusters = None
        
        # Métriques
        self.metrics = {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "last_training": None,
            "games_count": 0,
            "clusters_count": 0
        }
        
        # Cache pour optimisation
        self._similarity_cache = {}
        
    def prepare_data(self, games: List[Dict]) -> pd.DataFrame:
        """Préparation simplifiée des données"""
        df = pd.DataFrame(games)
        
        # Features textuelles combinées
        df['text_features'] = (
            df['title'].fillna('').astype(str) + ' ' + 
            df['genres'].fillna('').astype(str) + ' ' + 
            df.get('platforms', [''] * len(df)).apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            ).fillna('')
        )
        
        # Features numériques nettoyées
        df['rating'] = pd.to_numeric(df.get('rating', 0), errors='coerce').fillna(3.5)
        df['metacritic'] = pd.to_numeric(df.get('metacritic', 0), errors='coerce').fillna(70)
        
        # Normalisation
        df['rating_norm'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min() + 1e-6)
        df['metacritic_norm'] = (df['metacritic'] - df['metacritic'].min()) / (df['metacritic'].max() - df['metacritic'].min() + 1e-6)
        
        return df
    
    def train(self, games: List[Dict]) -> Dict[str, Any]:
        """Entraînement du modèle simplifié"""
        logger.info(f"Training simple model v{self.model_version} with {len(games)} games")
        
        try:
            # Préparation des données
            self.games_df = self.prepare_data(games)
            
            # 1. TF-IDF sur les features textuelles
            self.tfidf_matrix = self.tfidf.fit_transform(self.games_df['text_features'])
            logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            
            # 2. SVD pour réduction de dimensionnalité
            self.svd_features = self.svd.fit_transform(self.tfidf_matrix)
            logger.info(f"SVD features shape: {self.svd_features.shape}")
            
            # 3. Features numériques
            numeric_features = self.games_df[['rating_norm', 'metacritic_norm']].values
            numeric_features = self.scaler.fit_transform(numeric_features)
            
            # 4. Combinaison des features
            self.features_matrix = np.hstack([
                self.svd_features,
                numeric_features
            ])
            logger.info(f"Combined features shape: {self.features_matrix.shape}")
            
            # 5. KMeans clustering
            self.clusters = self.kmeans.fit_predict(self.features_matrix)
            self.games_df['cluster'] = self.clusters
            
            # 6. KNN pour similarité
            self.knn.fit(self.features_matrix)
            
            # Mise à jour état
            self.is_trained = True
            self.last_updated = datetime.utcnow()
            
            # Métriques
            self.metrics.update({
                "games_count": len(games),
                "clusters_count": len(np.unique(self.clusters)),
                "last_training": datetime.utcnow().isoformat(),
                "feature_dimension": self.features_matrix.shape[1]
            })
            
            logger.info(f"Training completed. {self.metrics['clusters_count']} clusters created")
            
            return {
                "status": "success",
                "games_count": self.metrics["games_count"],
                "clusters_count": self.metrics["clusters_count"],
                "feature_dimension": self.features_matrix.shape[1],
                "model_version": self.model_version
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict_similar_by_text(self, query: str, k: int = 10, min_confidence: float = 0.1) -> List[Dict]:
        """Recommandations basées sur une requête textuelle"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Vectorisation de la requête
        query_tfidf = self.tfidf.transform([query])
        query_svd = self.svd.transform(query_tfidf)
        
        # Features numériques moyennes
        mean_features = self.games_df[['rating_norm', 'metacritic_norm']].mean().values.reshape(1, -1)
        query_features = np.hstack([query_svd, mean_features])
        
        # Similarités cosinus
        similarities = cosine_similarity(query_features, self.features_matrix)[0]
        
        # Top-k
        top_indices = np.argsort(similarities)[::-1][:k * 2]  # Plus de candidats
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] >= min_confidence and len(recommendations) < k:
                game = self.games_df.iloc[idx]
                recommendations.append({
                    "id": int(game['id']),
                    "title": game['title'],
                    "genres": game['genres'],
                    "confidence": float(similarities[idx]),
                    "rating": float(game['rating']),
                    "metacritic": int(game['metacritic']),
                    "cluster": int(game['cluster']),
                    "method": "text_similarity"
                })
        
        self.metrics["total_predictions"] += 1
        return recommendations
    
    def predict_similar_by_game(self, game_id: int, k: int = 10) -> List[Dict]:
        """Recommandations basées sur un jeu spécifique (KNN)"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Trouver le jeu
        game_mask = self.games_df['id'] == game_id
        if not game_mask.any():
            raise ValueError(f"Game {game_id} not found")
        
        game_idx = game_mask.idxmax()
        game_features = self.features_matrix[game_idx].reshape(1, -1)
        
        # KNN pour trouver les similaires
        distances, indices = self.knn.kneighbors(game_features, n_neighbors=k + 1)
        
        recommendations = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != game_idx:  # Exclure le jeu lui-même
                game = self.games_df.iloc[idx]
                confidence = 1.0 - dist  # Convertir distance en similarité
                
                recommendations.append({
                    "id": int(game['id']),
                    "title": game['title'],
                    "genres": game['genres'],
                    "confidence": float(max(0, confidence)),
                    "rating": float(game['rating']),
                    "metacritic": int(game['metacritic']),
                    "cluster": int(game['cluster']),
                    "method": "knn_similarity",
                    "distance": float(dist)
                })
        
        # Trier par confiance
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        self.metrics["total_predictions"] += 1
        
        return recommendations[:k]
    
    def get_cluster_games(self, cluster_id: int, k: int = 10) -> List[Dict]:
        """Récupère les jeux d'un cluster spécifique"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        cluster_games = self.games_df[self.games_df['cluster'] == cluster_id]
        
        if len(cluster_games) == 0:
            return []
        
        # Trier par rating décroissant dans le cluster
        cluster_games_sorted = cluster_games.sort_values('rating', ascending=False)
        
        recommendations = []
        for _, game in cluster_games_sorted.head(k).iterrows():
            recommendations.append({
                "id": int(game['id']),
                "title": game['title'],
                "genres": game['genres'],
                "confidence": float(game['rating_norm']),
                "rating": float(game['rating']),
                "metacritic": int(game['metacritic']),
                "cluster": int(game['cluster']),
                "method": "cluster_based"
            })
        
        return recommendations
    
    def get_cluster_info(self) -> Dict[int, Dict]:
        """Informations sur les clusters"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        cluster_info = {}
        
        for cluster_id in range(self.metrics["clusters_count"]):
            cluster_games = self.games_df[self.games_df['cluster'] == cluster_id]
            
            if len(cluster_games) == 0:
                continue
            
            # Analyse du cluster
            genres_text = ' '.join(cluster_games['genres'].fillna(''))
            common_genres = []
            for genre in ['Action', 'RPG', 'Strategy', 'Indie', 'Adventure', 'Simulation', 'Sports']:
                if genre.lower() in genres_text.lower():
                    common_genres.append(genre)
            
            cluster_info[cluster_id] = {
                "size": len(cluster_games),
                "avg_rating": float(cluster_games['rating'].mean()),
                "avg_metacritic": float(cluster_games['metacritic'].mean()),
                "common_genres": common_genres[:3],
                "sample_games": cluster_games['title'].head(3).tolist()
            }
        
        return cluster_info
    
    def recommend_by_title_similarity(self, title: str, k: int = 10) -> List[Dict]:
        """Recommandations basées sur la similarité de titre"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Recherche fuzzy sur les titres
        title_lower = title.lower()
        
        # Score de similarité de titre simple
        title_scores = []
        for idx, game_title in enumerate(self.games_df['title']):
            game_title_lower = str(game_title).lower()
            
            # Score basique (mots en commun)
            title_words = set(title_lower.split())
            game_words = set(game_title_lower.split())
            
            if title_words & game_words:  # Intersection non vide
                score = len(title_words & game_words) / len(title_words | game_words)
                title_scores.append((idx, score))
        
        # Trier par score
        title_scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, score in title_scores[:k]:
            game = self.games_df.iloc[idx]
            recommendations.append({
                "id": int(game['id']),
                "title": game['title'],
                "genres": game['genres'],
                "confidence": float(score),
                "rating": float(game['rating']),
                "metacritic": int(game['metacritic']),
                "cluster": int(game['cluster']),
                "method": "title_similarity"
            })
        
        return recommendations
    
    def recommend_by_genre(self, genre: str, k: int = 10) -> List[Dict]:
        """Recommandations par genre"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Filtrer par genre
        genre_games = self.games_df[
            self.games_df['genres'].str.contains(genre, case=False, na=False)
        ]
        
        if len(genre_games) == 0:
            return []
        
        # Trier par rating
        genre_games_sorted = genre_games.sort_values('rating', ascending=False)
        
        recommendations = []
        for _, game in genre_games_sorted.head(k).iterrows():
            recommendations.append({
                "id": int(game['id']),
                "title": game['title'],
                "genres": game['genres'],
                "confidence": float(game['rating_norm']),
                "rating": float(game['rating']),
                "metacritic": int(game['metacritic']),
                "cluster": int(game['cluster']),
                "method": "genre_filter"
            })
        
        return recommendations
    
    def get_random_cluster_games(self, k: int = 10) -> List[Dict]:
        """Jeux aléatoires d'un cluster aléatoire"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Cluster aléatoire
        random_cluster = np.random.choice(range(self.metrics["clusters_count"]))
        
        # Jeux du cluster
        cluster_games = self.games_df[self.games_df['cluster'] == random_cluster]
        
        if len(cluster_games) == 0:
            return []
        
        # Sample aléatoire
        sample_size = min(k, len(cluster_games))
        sampled_games = cluster_games.sample(n=sample_size)
        
        recommendations = []
        for _, game in sampled_games.iterrows():
            recommendations.append({
                "id": int(game['id']),
                "title": game['title'],
                "genres": game['genres'],
                "confidence": float(game['rating_norm']),
                "rating": float(game['rating']),
                "metacritic": int(game['metacritic']),
                "cluster": int(game['cluster']),
                "method": "random_cluster"
            })
        
        return recommendations
    
    def get_info(self) -> Dict[str, Any]:
        """Informations sur le modèle"""
        return {
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "games_count": self.metrics.get("games_count", 0),
            "clusters_count": self.metrics.get("clusters_count", 0),
            "total_predictions": self.metrics["total_predictions"],
            "last_training": self.metrics.get("last_training"),
            "feature_dimension": self.metrics.get("feature_dimension", 0)
        }
    
    def save_model(self, filepath: str) -> bool:
        """Sauvegarde le modèle"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    @staticmethod
    def load_model(filepath: str) -> 'SimpleRecommendationModel':
        """Charge le modèle"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

# Instance globale pour compatibilité
_recommendation_model = None

def get_model() -> SimpleRecommendationModel:
    """Retourne le modèle de recommandation principal"""
    global _recommendation_model
    if _recommendation_model is None:
        _recommendation_model = SimpleRecommendationModel()
        
        # Essayer de charger un modèle existant
        model_path = "model/simple_recommendation_model.pkl"
        if os.path.exists(model_path):
            try:
                _recommendation_model = SimpleRecommendationModel.load_model(model_path)
                logger.info("Existing model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
                _recommendation_model = SimpleRecommendationModel()
    
    return _recommendation_model

def reset_model():
    """Reset le modèle global (pour les tests)"""
    global _recommendation_model
    _recommendation_model = None
