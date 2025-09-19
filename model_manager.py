# model_manager.py - Gestion du modèle de recommandation avec K-Means + KNN
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
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd

logger = logging.getLogger("model-manager")

class RecommendationModel:
    """Modèle de recommandation basé sur TF-IDF → SVD → K-Means → KNN"""
    
    def __init__(self, model_version: str = "2.0.0"):
        self.model_version = model_version
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.svd = TruncatedSVD(n_components=20, random_state=42)
        self.kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        self.knn = NearestNeighbors(n_neighbors=20, metric='cosine')
        
        # Données du modèle
        self.game_features = None
        self.games_df = None
        self.cluster_labels = None
        self.is_trained = False
        
        # Cache pour optimisations
        self._cluster_cache = {}
        self._similarity_cache = {}
        
        # Métriques pour monitoring
        self.metrics = {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "last_training": None,
            "model_version": model_version,
            "feature_dim": 0,
            "n_clusters": 8
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
        
        # Normaliser les features numériques
        if df['rating'].max() > df['rating'].min():
            df['rating_norm'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
        else:
            df['rating_norm'] = 0.5
            
        if df['metacritic'].max() > df['metacritic'].min():
            df['metacritic_norm'] = (df['metacritic'] - df['metacritic'].min()) / (df['metacritic'].max() - df['metacritic'].min())
        else:
            df['metacritic_norm'] = 0.5
        
        return df
    
    def train(self, games: List[Dict]) -> Dict:
        """Entraîne le modèle avec pipeline TF-IDF → SVD → K-Means → KNN"""
        logger.info(f"Training model v{self.model_version} with {len(games)} games")
        
        # Préparer les données
        self.games_df = self.prepare_features(games)
        
        # Pipeline ML
        # 1. Vectorisation TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(self.games_df['combined_features'])
        
        # 2. Réduction de dimension avec SVD
        tfidf_reduced = self.svd.fit_transform(tfidf_matrix)
        
        # 3. Combiner features textuelles et numériques
        numeric_features = self.games_df[['rating_norm', 'metacritic_norm']].values
        self.game_features = np.hstack([tfidf_reduced, numeric_features])
        
        # 4. Clustering K-Means
        self.cluster_labels = self.kmeans.fit_predict(self.game_features)
        self.games_df['cluster'] = self.cluster_labels
        
        # 5. Entraîner KNN pour recherche de voisins
        self.knn.fit(self.game_features)
        
        # Mise à jour des métriques
        self.is_trained = True
        self.metrics["last_training"] = datetime.utcnow().isoformat()
        self.metrics["feature_dim"] = self.game_features.shape[1]
        
        # Validation et analyse des clusters
        validation_metrics = self._validate_model()
        cluster_analysis = self._analyze_clusters()
        
        logger.info(f"Model trained successfully. Feature dimension: {self.game_features.shape}")
        logger.info(f"Clusters created: {len(np.unique(self.cluster_labels))}")
        
        return {
            "status": "success",
            "model_version": self.model_version,
            "games_count": len(games),
            "feature_dim": self.game_features.shape[1],
            "n_clusters": len(np.unique(self.cluster_labels)),
            "validation_metrics": validation_metrics,
            "cluster_analysis": cluster_analysis
        }
    
    def _analyze_clusters(self) -> Dict:
        """Analyse la qualité des clusters"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        cluster_info = {}
        for cluster_id in np.unique(self.cluster_labels):
            cluster_games = self.games_df[self.games_df['cluster'] == cluster_id]
            
            # Genres les plus fréquents dans ce cluster
            all_genres = ' '.join(cluster_games['genres'].fillna(''))
            top_genres = [g.strip() for g in all_genres.split(',') if g.strip()]
            genre_counts = pd.Series(top_genres).value_counts().head(3)
            
            cluster_info[f"cluster_{cluster_id}"] = {
                "size": len(cluster_games),
                "avg_rating": float(cluster_games['rating'].mean()),
                "avg_metacritic": float(cluster_games['metacritic'].mean()),
                "top_genres": genre_counts.to_dict() if not genre_counts.empty else {},
                "sample_games": cluster_games['title'].head(3).tolist()
            }
        
        return cluster_info
    
    def _validate_model(self) -> Dict:
        """Validation interne du modèle"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        # Calculer la silhouette score approximative
        from sklearn.metrics import silhouette_score
        try:
            silhouette_avg = silhouette_score(self.game_features, self.cluster_labels)
        except Exception:
            silhouette_avg = 0.0
        
        # Inertie des clusters
        inertia = self.kmeans.inertia_
        
        return {
            "silhouette_score": float(silhouette_avg),
            "inertia": float(inertia),
            "n_clusters": len(np.unique(self.cluster_labels))
        }
    
    def predict(self, query: str, k: int = 10, min_confidence: float = 0.1) -> List[Dict]:
        """Recommandations générales basées sur une requête"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Vectoriser la requête
        query_vec = self.vectorizer.transform([query])
        query_reduced = self.svd.transform(query_vec)
        
        # Ajouter features numériques moyennes
        avg_numeric = np.mean(self.games_df[['rating_norm', 'metacritic_norm']].values, axis=0)
        query_features = np.hstack([query_reduced[0], avg_numeric])
        
        # Trouver le cluster le plus proche
        query_cluster = self.kmeans.predict([query_features])[0]
        
        # Rechercher dans le cluster + clusters voisins
        cluster_games = self.games_df[self.games_df['cluster'] == query_cluster]
        
        # Utiliser KNN pour affiner
        distances, indices = self.knn.kneighbors([query_features], n_neighbors=min(k*2, len(self.games_df)))
        
        recommendations = []
        for i, idx in enumerate(indices[0]):
            score = 1.0 - distances[0][i]  # Convertir distance en similarité
            if score >= min_confidence:
                game = self.games_df.iloc[idx]
                recommendations.append({
                    "title": game['title'],
                    "genres": game['genres'],
                    "confidence": float(score),
                    "rating": float(game['rating']),
                    "metacritic": int(game['metacritic']),
                    "cluster": int(game['cluster'])
                })
        
        # Trier par confiance
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Mise à jour des métriques
        self._update_prediction_metrics(recommendations[:k])
        
        return recommendations[:k]
    
    def predict_similar_game(self, game_id: int, k: int = 10) -> List[Dict]:
        """KNN sur un jeu spécifique"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Trouver l'index du jeu
        game_idx = self.games_df[self.games_df['id'] == game_id].index
        if len(game_idx) == 0:
            raise ValueError(f"Game with id {game_id} not found")
        
        game_idx = game_idx[0]
        source_game = self.games_df.iloc[game_idx]
        
        # Utiliser KNN pour trouver les voisins
        distances, indices = self.knn.kneighbors([self.game_features[game_idx]], n_neighbors=k+1)
        
        recommendations = []
        for i, idx in enumerate(indices[0][1:]):  # Exclure le jeu lui-même
            score = 1.0 - distances[0][i+1]
            game = self.games_df.iloc[idx]
            recommendations.append({
                "title": game['title'],
                "genres": game['genres'],
                "confidence": float(score),
                "rating": float(game['rating']),
                "metacritic": int(game['metacritic']),
                "cluster": int(game['cluster'])
            })
        
        return recommendations
    
    def get_cluster_games(self, cluster_id: int, k: int = 20) -> List[Dict]:
        """Retourne les jeux d'un cluster spécifique"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if cluster_id not in range(8):
            raise ValueError(f"Cluster {cluster_id} not found. Available: 0-7")
        
        cluster_games = self.games_df[self.games_df['cluster'] == cluster_id]
        
        # Trier par rating décroissant
        cluster_games = cluster_games.sort_values('rating', ascending=False)
        
        results = []
        for _, game in cluster_games.head(k).iterrows():
            results.append({
                "title": game['title'],
                "genres": game['genres'],
                "rating": float(game['rating']),
                "metacritic": int(game['metacritic']),
                "cluster": int(game['cluster'])
            })
        
        return results
    
    def explore_clusters(self) -> Dict:
        """Analyse complète des clusters"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        return self._analyze_clusters()
    
    def get_random_cluster_games(self, k: int = 10) -> List[Dict]:
        """Découverte aléatoire d'un cluster"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Choisir un cluster aléatoire
        random_cluster = np.random.choice(range(8))
        
        cluster_games = self.games_df[self.games_df['cluster'] == random_cluster]
        
        # Échantillonner aléatoirement
        if len(cluster_games) > k:
            sampled = cluster_games.sample(n=k)
        else:
            sampled = cluster_games
        
        results = []
        for _, game in sampled.iterrows():
            results.append({
                "title": game['title'],
                "genres": game['genres'],
                "rating": float(game['rating']),
                "metacritic": int(game['metacritic']),
                "cluster": int(game['cluster'])
            })
        
        return {
            "cluster_id": random_cluster,
            "games": results,
            "cluster_info": self._analyze_clusters().get(f"cluster_{random_cluster}", {})
        }
    
    def predict_by_title_similarity(self, title: str, k: int = 10) -> List[Dict]:
        """Similarité basée sur les titres"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Recherche approximative par titre
        title_lower = title.lower()
        matches = self.games_df[self.games_df['title'].str.lower().str.contains(title_lower, na=False)]
        
        if matches.empty:
            return []
        
        # Prendre le premier match comme référence
        ref_idx = matches.index[0]
        return self.predict_similar_game(int(self.games_df.iloc[ref_idx]['id']), k)
    
    def predict_by_genre(self, genre: str, k: int = 10) -> List[Dict]:
        """Filtrage par genre"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        genre_lower = genre.lower()
        matches = self.games_df[self.games_df['genres'].str.lower().str.contains(genre_lower, na=False)]
        
        if matches.empty:
            return []
        
        # Trier par rating et retourner les meilleurs
        best_games = matches.sort_values('rating', ascending=False).head(k)
        
        results = []
        for _, game in best_games.iterrows():
            results.append({
                "title": game['title'],
                "genres": game['genres'],
                "rating": float(game['rating']),
                "metacritic": int(game['metacritic']),
                "cluster": int(game['cluster'])
            })
        
        return results
    
    def _update_prediction_metrics(self, recommendations: List[Dict]):
        """Met à jour les métriques de prédiction"""
        self.metrics["total_predictions"] += 1
        if recommendations:
            avg_conf = np.mean([r.get('confidence', 0.0) for r in recommendations])
            self.metrics["avg_confidence"] = (
                (self.metrics["avg_confidence"] * (self.metrics["total_predictions"] - 1) + avg_conf) 
                / self.metrics["total_predictions"]
            )
    
    def save_model(self, filepath: str = "model/recommendation_model.pkl") -> bool:
        """Sauvegarde le modèle entraîné"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                "version": self.model_version,
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "kmeans": self.kmeans,
                "knn": self.knn,
                "game_features": self.game_features,
                "games_df": self.games_df,
                "cluster_labels": self.cluster_labels,
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
            self.kmeans = model_data["kmeans"]
            self.knn = model_data["knn"]
            self.game_features = model_data["game_features"]
            self.games_df = model_data["games_df"]
            self.cluster_labels = model_data["cluster_labels"]
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
