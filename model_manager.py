# enhanced_model_manager.py - Modèle hybride avec Gradient Boosting
from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Imports ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger("enhanced-model-manager")

@dataclass
class ModelMetrics:
    """Métriques du modèle hybride"""
    content_based_accuracy: float
    collaborative_accuracy: float
    gb_score: float
    combined_score: float
    training_time: float
    prediction_time: float

class HybridRecommendationModel:
    """
    Modèle de recommandation hybride avec Gradient Boosting
    
    Architecture:
    1. Content-Based: TF-IDF + SVD (similarité contenu)
    2. Collaborative: Matrix Factorization (NMF-style)
    3. Gradient Boosting: Prédiction de rating basée sur features
    4. Ensemble: Combinaison pondérée des 3 approches
    """
    
    def __init__(self, model_version: str = "3.0.0-hybrid"):
        self.model_version = model_version
        
        # === Composants Content-Based ===
        self.content_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words="english",
            lowercase=True,
            min_df=2
        )
        self.content_svd = TruncatedSVD(n_components=50, random_state=42)
        
        # === Composants Collaborative ===
        self.user_item_matrix = None
        self.user_features = None
        self.item_features = None
        self.collab_knn = NearestNeighbors(metric="cosine", n_neighbors=20)
        
        # === Gradient Boosting pour Rating Prediction ===
        self.gb_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        self.feature_scaler = StandardScaler()
        self.genre_encoder = LabelEncoder()
        self.platform_encoder = LabelEncoder()
        
        # === Clustering ===
        self.kmeans = KMeans(n_clusters=12, random_state=42, n_init="auto")
        
        # === État du modèle ===
        self.is_trained = False
        self.games_df = None
        self.content_features = None
        self.numerical_features = None
        self.cluster_labels = None
        
        # === Poids pour l'ensemble ===
        self.ensemble_weights = {
            "content": 0.4,
            "collaborative": 0.3,
            "gradient_boosting": 0.3
        }
        
        # === Métriques ===
        self.metrics = ModelMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.prediction_count = 0
        
    def prepare_features(self, games: List[Dict]) -> pd.DataFrame:
        """Préparation avancée des features pour le modèle hybride"""
        df = pd.DataFrame(games)
        
        # === Nettoyage des données ===
        df["title"] = df["title"].fillna("Unknown Game")
        df["genres"] = df["genres"].fillna("Unknown")
        df["platforms"] = df["platforms"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x or "Unknown")
        )
        
        # === Features textuelles combinées ===
        df["combined_text"] = (
            df["title"] + " " + 
            df["genres"] + " " + 
            df["platforms"]
        )
        
        # === Features numériques robustes ===
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0)
        df["metacritic"] = pd.to_numeric(df["metacritic"], errors="coerce").fillna(75)
        
        # === Features d'ingénierie ===
        # Popularité basée sur le nombre de plateformes
        df["platform_count"] = df["platforms"].apply(lambda x: len(x.split()))
        
        # Score composite
        df["composite_score"] = (df["rating"] * 0.7 + df["metacritic"] / 20 * 0.3)
        
        # Catégorisation des genres
        df["is_action"] = df["genres"].str.contains("Action", case=False, na=False).astype(int)
        df["is_rpg"] = df["genres"].str.contains("RPG", case=False, na=False).astype(int)
        df["is_indie"] = df["genres"].str.contains("Indie", case=False, na=False).astype(int)
        df["is_strategy"] = df["genres"].str.contains("Strategy", case=False, na=False).astype(int)
        
        # Normalisation
        for col in ["rating", "metacritic", "composite_score"]:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max > col_min:
                df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[f"{col}_norm"] = 0.5
        
        return df
    
    def train(self, games: List[Dict]) -> Dict[str, Any]:
        """Entraînement du modèle hybride"""
        start_time = datetime.utcnow()
        logger.info(f"Training hybrid model v{self.model_version} with {len(games)} games")
        
        # === 1. Préparation des données ===
        self.games_df = self.prepare_features(games)
        
        # === 2. Entraînement Content-Based ===
        content_metrics = self._train_content_based()
        
        # === 3. Entraînement Collaborative ===
        collab_metrics = self._train_collaborative()
        
        # === 4. Entraînement Gradient Boosting ===
        gb_metrics = self._train_gradient_boosting()
        
        # === 5. Clustering ===
        self._train_clustering()
        
        # === 6. Validation croisée ===
        validation_metrics = self._cross_validate()
        
        self.is_trained = True
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        # === 7. Mise à jour des métriques ===
        self.metrics = ModelMetrics(
            content_based_accuracy=content_metrics.get("accuracy", 0.0),
            collaborative_accuracy=collab_metrics.get("accuracy", 0.0),
            gb_score=gb_metrics.get("r2_score", 0.0),
            combined_score=validation_metrics.get("ensemble_score", 0.0),
            training_time=training_time,
            prediction_time=0.0
        )
        
        logger.info(f"Hybrid model trained in {training_time:.2f}s")
        
        return {
            "status": "success",
            "model_version": self.model_version,
            "training_time": training_time,
            "games_count": len(games),
            "content_metrics": content_metrics,
            "collaborative_metrics": collab_metrics,
            "gb_metrics": gb_metrics,
            "validation_metrics": validation_metrics,
            "ensemble_weights": self.ensemble_weights
        }
    
    def _train_content_based(self) -> Dict[str, float]:
        """Entraînement du composant content-based"""
        logger.info("Training content-based component...")
        
        # TF-IDF sur le texte combiné
        tfidf_matrix = self.content_vectorizer.fit_transform(self.games_df["combined_text"])
        
        # SVD pour réduction dimensionnelle
        n_features = tfidf_matrix.shape[1]
        n_components = min(50, max(5, n_features - 1))
        self.content_svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        self.content_features = self.content_svd.fit_transform(tfidf_matrix)
        
        # Métriques de qualité (variance expliquée par SVD)
        explained_variance = self.content_svd.explained_variance_ratio_.sum()
        
        return {
            "accuracy": explained_variance,
            "n_components": n_components,
            "n_features": n_features,
            "explained_variance": explained_variance
        }
    
    def _train_collaborative(self) -> Dict[str, float]:
        """Entraînement du composant collaborative filtering"""
        logger.info("Training collaborative filtering component...")
        
        # Simulation d'interactions utilisateur-jeu pour la démo
        # En production, ces données viendraient de vraies interactions
        n_users = min(100, len(self.games_df) * 2)
        n_games = len(self.games_df)
        
        # Création d'une matrice d'interactions simulée
        np.random.seed(42)
        user_game_matrix = np.zeros((n_users, n_games))
        
        # Simulation d'interactions basées sur les genres préférés
        for user_id in range(n_users):
            # Chaque utilisateur a des préférences pour certains genres
            preferred_genres = np.random.choice(
                ["Action", "RPG", "Indie", "Strategy", "Simulation"], 
                size=np.random.randint(1, 4), 
                replace=False
            )
            
            for game_idx, game_row in self.games_df.iterrows():
                # Probabilité d'interaction basée sur les genres
                genre_match = any(genre in game_row["genres"] for genre in preferred_genres)
                
                if genre_match and np.random.random() < 0.3:
                    # Rating basé sur le rating réel du jeu + bruit
                    base_rating = game_row["rating"]
                    noise = np.random.normal(0, 0.5)
                    rating = np.clip(base_rating + noise, 1, 5)
                    user_game_matrix[user_id, game_idx] = rating
        
        self.user_item_matrix = user_game_matrix
        
        # Factorisation matricielle simple avec SVD
        from scipy.sparse.linalg import svds
        
        # Centrer les données
        user_ratings_mean = np.mean(user_game_matrix, axis=1)
        user_game_matrix_centered = user_game_matrix - user_ratings_mean.reshape(-1, 1)
        
        # SVD sur la matrice centrée
        k = min(20, min(user_game_matrix.shape) - 1)
        U, sigma, Vt = svds(user_game_matrix_centered, k=k)
        
        self.user_features = U
        self.item_features = Vt.T
        
        # Entraîner KNN sur les features d'items
        self.collab_knn.fit(self.item_features)
        
        # Métrique: reconstruction error
        reconstructed = np.dot(np.dot(U, np.diag(sigma)), Vt)
        mse = np.mean((user_game_matrix_centered - reconstructed) ** 2)
        accuracy = max(0, 1 - mse / np.var(user_game_matrix_centered))
        
        return {
            "accuracy": accuracy,
            "n_factors": k,
            "reconstruction_mse": mse,
            "sparsity": np.count_nonzero(user_game_matrix) / user_game_matrix.size
        }
    
    def _train_gradient_boosting(self) -> Dict[str, float]:
        """Entraînement du modèle Gradient Boosting pour prédiction de rating"""
        logger.info("Training Gradient Boosting component...")
        
        # === Préparation des features pour GB ===
        gb_features = []
        
        # Features numériques
        numerical_cols = ["metacritic", "platform_count", "composite_score", 
                         "is_action", "is_rpg", "is_indie", "is_strategy"]
        
        for col in numerical_cols:
            if col in self.games_df.columns:
                gb_features.append(self.games_df[col].values)
        
        # Features de contenu (réduites)
        content_features_reduced = self.content_features[:, :10]  # Top 10 composantes
        gb_features.append(content_features_reduced)
        
        # Encodage des genres principaux
        main_genres = self.games_df["genres"].apply(lambda x: x.split()[0] if x.split() else "Unknown")
        self.genre_encoder.fit(main_genres)
        genre_encoded = self.genre_encoder.transform(main_genres)
        gb_features.append(genre_encoded.reshape(-1, 1))
        
        # Concaténation de toutes les features
        X = np.hstack(gb_features)
        y = self.games_df["rating"].values
        
        # === Division train/test ===
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # === Normalisation ===
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # === Entraînement Gradient Boosting ===
        self.gb_regressor.fit(X_train_scaled, y_train)
        
        # === Évaluation ===
        y_pred = self.gb_regressor.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Sauvegarder les features pour prédiction
        self.numerical_features = X
        
        return {
            "r2_score": r2,
            "mse": mse,
            "feature_importance": self.gb_regressor.feature_importances_.tolist(),
            "n_features": X.shape[1],
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
    
    def _train_clustering(self):
        """Entraînement du clustering sur les features de contenu"""
        self.cluster_labels = self.kmeans.fit_predict(self.content_features)
        logger.info(f"Clustering completed with {self.kmeans.n_clusters} clusters")
    
    def _cross_validate(self) -> Dict[str, float]:
        """Validation croisée de l'ensemble"""
        # Simulation d'une validation croisée simple
        # En production, cela serait plus sophistiqué
        
        ensemble_score = (
            self.metrics.content_based_accuracy * self.ensemble_weights["content"] +
            self.metrics.collaborative_accuracy * self.ensemble_weights["collaborative"] +
            self.metrics.gb_score * self.ensemble_weights["gradient_boosting"]
        )
        
        return {
            "ensemble_score": ensemble_score,
            "validation_method": "weighted_average"
        }
    
    def predict(self, query: str, k: int = 10, min_confidence: float = 0.1) -> List[Dict]:
        """Prédiction hybride combinant les 3 approches"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        start_time = datetime.utcnow()
        
        # === 1. Prédictions Content-Based ===
        content_scores = self._predict_content_based(query, k * 2)
        
        # === 2. Prédictions Collaborative ===
        collab_scores = self._predict_collaborative(k * 2)
        
        # === 3. Prédictions Gradient Boosting ===
        gb_scores = self._predict_gradient_boosting()
        
        # === 4. Combinaison Ensemble ===
        combined_scores = self._combine_predictions(
            content_scores, collab_scores, gb_scores, k, min_confidence
        )
        
        # === 5. Mise à jour des métriques ===
        prediction_time = (datetime.utcnow() - start_time).total_seconds()
        self.metrics.prediction_time = (
            self.metrics.prediction_time * self.prediction_count + prediction_time
        ) / (self.prediction_count + 1)
        self.prediction_count += 1
        
        return combined_scores
    
    def _predict_content_based(self, query: str, k: int) -> Dict[int, float]:
        """Prédictions basées sur le contenu"""
        query_tfidf = self.content_vectorizer.transform([query])
        query_features = self.content_svd.transform(query_tfidf)
        
        similarities = cosine_similarity(query_features, self.content_features)[0]
        
        scores = {}
        for idx, similarity in enumerate(similarities):
            game_id = int(self.games_df.iloc[idx]["id"])
            scores[game_id] = float(similarity)
        
        return scores
    
    def _predict_collaborative(self, k: int) -> Dict[int, float]:
        """Prédictions collaboratives (simulation)"""
        # Simulation d'un utilisateur moyen
        if self.item_features is not None:
            avg_user_profile = np.mean(self.user_features, axis=0)
            similarities = cosine_similarity([avg_user_profile], self.item_features)[0]
            
            scores = {}
            for idx, similarity in enumerate(similarities):
                game_id = int(self.games_df.iloc[idx]["id"])
                scores[game_id] = float(similarity)
            
            return scores
        
        return {}
    
    def _predict_gradient_boosting(self) -> Dict[int, float]:
        """Prédictions Gradient Boosting"""
        if self.numerical_features is not None:
            X_scaled = self.feature_scaler.transform(self.numerical_features)
            predicted_ratings = self.gb_regressor.predict(X_scaled)
            
            scores = {}
            for idx, rating in enumerate(predicted_ratings):
                game_id = int(self.games_df.iloc[idx]["id"])
                # Normaliser le rating en score 0-1
                scores[game_id] = float(rating / 5.0)
            
            return scores
        
        return {}
    
    def _combine_predictions(self, content_scores: Dict, collab_scores: Dict, 
                           gb_scores: Dict, k: int, min_confidence: float) -> List[Dict]:
        """Combinaison des prédictions avec poids d'ensemble"""
        
        all_game_ids = set(content_scores.keys()) | set(collab_scores.keys()) | set(gb_scores.keys())
        
        combined_scores = []
        
        for game_id in all_game_ids:
            content_score = content_scores.get(game_id, 0.0)
            collab_score = collab_scores.get(game_id, 0.0)
            gb_score = gb_scores.get(game_id, 0.0)
            
            # Score combiné
            final_score = (
                content_score * self.ensemble_weights["content"] +
                collab_score * self.ensemble_weights["collaborative"] +
                gb_score * self.ensemble_weights["gradient_boosting"]
            )
            
            if final_score >= min_confidence:
                # Récupérer les infos du jeu
                game_row = self.games_df[self.games_df["id"] == game_id].iloc[0]
                
                combined_scores.append({
                    "id": int(game_id),
                    "title": game_row["title"],
                    "genres": game_row["genres"],
                    "confidence": final_score,
                    "rating": float(game_row["rating"]),
                    "metacritic": int(game_row["metacritic"]),
                    "prediction_breakdown": {
                        "content_based": content_score,
                        "collaborative": collab_score,
                        "gradient_boosting": gb_score
                    },
                    "algorithm": "hybrid_ensemble"
                })
        
        # Trier par score et retourner top-k
        combined_scores.sort(key=lambda x: x["confidence"], reverse=True)
        return combined_scores[:k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Informations détaillées sur le modèle hybride"""
        return {
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "components": {
                "content_based": {
                    "vectorizer_features": len(self.content_vectorizer.get_feature_names_out()) if self.is_trained else 0,
                    "svd_components": self.content_svd.n_components if self.is_trained else 0
                },
                "collaborative": {
                    "user_features_shape": self.user_features.shape if self.user_features is not None else None,
                    "item_features_shape": self.item_features.shape if self.item_features is not None else None
                },
                "gradient_boosting": {
                    "n_estimators": self.gb_regressor.n_estimators,
                    "max_depth": self.gb_regressor.max_depth,
                    "learning_rate": self.gb_regressor.learning_rate
                }
            },
            "ensemble_weights": self.ensemble_weights,
            "metrics": {
                "content_accuracy": self.metrics.content_based_accuracy,
                "collaborative_accuracy": self.metrics.collaborative_accuracy,
                "gb_r2_score": self.metrics.gb_score,
                "combined_score": self.metrics.combined_score,
                "training_time": self.metrics.training_time,
                "avg_prediction_time": self.metrics.prediction_time,
                "total_predictions": self.prediction_count
            }
        }

# Fonction singleton mise à jour
_hybrid_model_instance: Optional[HybridRecommendationModel] = None

def get_hybrid_model() -> HybridRecommendationModel:
    """Récupère l'instance du modèle hybride"""
    global _hybrid_model_instance
    if _hybrid_model_instance is None:
        _hybrid_model_instance = HybridRecommendationModel()
    return _hybrid_model_instance

def reset_hybrid_model():
    """Reset l'instance du modèle hybride"""
    global _hybrid_model_instance
    _hybrid_model_instance = None
