# model_manager_enhanced.py - Modèle hybride avec Gradient Boosting
from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("enhanced-model-manager")

@dataclass
class ModelPerformance:
    """Métriques de performance du modèle"""
    mse: float
    r2_score: float
    cross_val_score: float
    feature_importance: Dict[str, float]
    training_time: float
    model_size_mb: float

class HybridRecommendationModel:
    """
    Modèle de recommandation hybride avec Gradient Boosting
    
    Pipeline amélioré :
    1. TF-IDF + SVD pour les features textuelles
    2. Gradient Boosting pour prédire les ratings
    3. Random Forest pour la classification des genres
    4. Système de scoring combiné
    5. Cold start handling
    """

    def __init__(self, model_version: str = "3.0.0-hybrid"):
        self.model_version = model_version
        
        # Modèles textuels (existants)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # Augmenté pour plus de contexte
            stop_words="english",
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        self.svd = None  # Initialisé dynamiquement
        
        # Nouveaux modèles ML
        self.rating_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            validation_fraction=0.2,
            n_iter_no_change=10,
            tol=1e-4
        )
        
        self.genre_classifier = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.popularity_model = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.15,
            max_depth=4,
            random_state=42
        )
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.genre_encoder = LabelEncoder()
        self.platform_encoder = LabelEncoder()
        
        # Modèles de clustering (existants)
        self.kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
        self.knn = NearestNeighbors(metric="cosine", n_neighbors=50, algorithm="auto")
        
        # État du modèle
        self.is_trained: bool = False
        self.games_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.performance_metrics: Optional[ModelPerformance] = None
        
        # Cache pour optimisation
        self._feature_cache: Dict[str, np.ndarray] = {}
        self._prediction_cache: Dict[str, List[Dict]] = {}
        
        # Métriques
        self.metrics = {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "last_training": None,
            "model_version": model_version,
            "hybrid_performance": {}
        }

    def prepare_enhanced_features(self, games: List[Dict]) -> pd.DataFrame:
        """Préparation des features enrichies pour ML"""
        df = pd.DataFrame(games)
        
        # Features textuelles (existant)
        df["platforms_str"] = df["platforms"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x or "")
        )
        df["combined_features"] = (
            df["title"].fillna("") + " " +
            df["genres"].fillna("") + " " +
            df["platforms_str"].fillna("")
        )
        
        # Features numériques enrichies
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.5)
        df["metacritic"] = pd.to_numeric(df["metacritic"], errors="coerce").fillna(70)
        
        # Nouvelles features engineered
        df["rating_category"] = pd.cut(df["rating"], bins=[0, 3, 4, 4.5, 5], labels=[0, 1, 2, 3])
        df["metacritic_category"] = pd.cut(df["metacritic"], bins=[0, 60, 75, 85, 100], labels=[0, 1, 2, 3])
        
        # Features temporelles
        if "release_date" in df.columns:
            df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
            df["release_year"] = df["release_date"].dt.year.fillna(2020)
            df["game_age"] = 2024 - df["release_year"]
            df["is_recent"] = (df["game_age"] <= 3).astype(int)
        else:
            df["release_year"] = 2020
            df["game_age"] = 4
            df["is_recent"] = 0
        
        # Features de popularité simulées
        df["platform_count"] = df["platforms"].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        )
        df["title_length"] = df["title"].str.len()
        df["genre_count"] = df["genres"].str.count(",") + 1
        
        # Features d'interaction
        df["rating_metacritic_product"] = df["rating"] * df["metacritic"] / 100
        df["quality_score"] = (df["rating"] * 0.6 + df["metacritic"] / 20 * 0.4)
        
        # Normalisation
        for col in ["rating", "metacritic", "game_age", "platform_count", "title_length"]:
            if col in df.columns:
                col_data = df[col]
                col_min, col_max = col_data.min(), col_data.max()
                if col_max != col_min:
                    df[f"{col}_norm"] = (col_data - col_min) / (col_max - col_min)
                else:
                    df[f"{col}_norm"] = 0.5
        
        return df

    def train(self, games: List[Dict]) -> Dict:
        """Entraînement du modèle hybride"""
        start_time = datetime.utcnow()
        logger.info(f"Training hybrid model v{self.model_version} with {len(games)} games")
        
        try:
            # Préparation des données
            self.games_df = self.prepare_enhanced_features(games)
            
            # 1. Pipeline textuel (TF-IDF + SVD)
            self._train_textual_pipeline()
            
            # 2. Modèles ML supervisés
            ml_performance = self._train_ml_models()
            
            # 3. Clustering et KNN
            self._train_clustering_models()
            
            # 4. Assemblage final
            self._create_ensemble_features()
            
            # Performance globale
            training_duration = (datetime.utcnow() - start_time).total_seconds()
            
            self.performance_metrics = ModelPerformance(
                mse=ml_performance.get("rating_mse", 0.0),
                r2_score=ml_performance.get("rating_r2", 0.0),
                cross_val_score=ml_performance.get("cross_val", 0.0),
                feature_importance=ml_performance.get("feature_importance", {}),
                training_time=training_duration,
                model_size_mb=self._estimate_model_size()
            )
            
            self.is_trained = True
            self.metrics["last_training"] = start_time.isoformat()
            self.metrics["hybrid_performance"] = ml_performance
            
            logger.info(f"Hybrid model trained successfully in {training_duration:.2f}s")
            
            return {
                "status": "success",
                "model_version": self.model_version,
                "training_duration": training_duration,
                "performance": ml_performance,
                "games_count": len(games),
                "feature_count": self.feature_matrix.shape[1] if self.feature_matrix is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _train_textual_pipeline(self):
        """Entraîne le pipeline textuel TF-IDF + SVD"""
        # TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(self.games_df["combined_features"])
        
        # SVD adaptatif
        n_features = tfidf_matrix.shape[1]
        n_components = min(100, max(10, n_features - 1))
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.textual_features = self.svd.fit_transform(tfidf_matrix)
        
        logger.info(f"Textual pipeline: {n_components} SVD components from {n_features} features")

    def _train_ml_models(self) -> Dict:
        """Entraîne les modèles ML supervisés"""
        performance = {}
        
        try:
            # Préparation des features pour ML
            ml_features = self._prepare_ml_features()
            
            # 1. Prédicteur de rating
            target_rating = self.games_df["rating"].values
            if len(set(target_rating)) > 1:  # Vérifier la variance
                X_train, X_test, y_train, y_test = train_test_split(
                    ml_features, target_rating, test_size=0.2, random_state=42
                )
                
                self.rating_predictor.fit(X_train, y_train)
                
                # Évaluation
                y_pred = self.rating_predictor.predict(X_test)
                performance["rating_mse"] = mean_squared_error(y_test, y_pred)
                performance["rating_r2"] = r2_score(y_test, y_pred)
                
                # Feature importance
                feature_names = self._get_ml_feature_names()
                importance_dict = dict(zip(feature_names, self.rating_predictor.feature_importances_))
                performance["feature_importance"] = importance_dict
                
                logger.info(f"Rating predictor R²: {performance['rating_r2']:.3f}")
            
            # 2. Modèle de popularité (basé sur metacritic)
            target_popularity = self.games_df["metacritic"].values
            if len(set(target_popularity)) > 1:
                self.popularity_model.fit(ml_features, target_popularity)
                pop_pred = self.popularity_model.predict(ml_features)
                performance["popularity_r2"] = r2_score(target_popularity, pop_pred)
            
            # 3. Validation croisée simple
            if len(ml_features) > 10:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(
                    self.rating_predictor, ml_features, target_rating, 
                    cv=min(5, len(ml_features)), scoring="r2"
                )
                performance["cross_val"] = float(np.mean(cv_scores))
                
        except Exception as e:
            logger.warning(f"ML training partially failed: {e}")
            performance["warning"] = str(e)
        
        return performance

    def _prepare_ml_features(self) -> np.ndarray:
        """Prépare les features pour les modèles ML"""
        features_list = []
        
        # Features textuelles (SVD)
        features_list.append(self.textual_features)
        
        # Features numériques normalisées
        numeric_cols = [
            "rating_norm", "metacritic_norm", "game_age", "platform_count", 
            "title_length", "genre_count", "quality_score"
        ]
        
        numeric_features = []
        for col in numeric_cols:
            if col in self.games_df.columns:
                numeric_features.append(self.games_df[col].values)
        
        if numeric_features:
            numeric_matrix = np.column_stack(numeric_features)
            numeric_matrix = self.scaler.fit_transform(numeric_matrix)
            features_list.append(numeric_matrix)
        
        # Features catégorielles
        categorical_features = []
        
        # Encodage des genres principaux
        main_genres = self.games_df["genres"].str.split().str[0].fillna("Unknown")
        try:
            genre_encoded = self.genre_encoder.fit_transform(main_genres)
            categorical_features.append(genre_encoded.reshape(-1, 1))
        except Exception:
            pass
        
        if categorical_features:
            cat_matrix = np.column_stack(categorical_features)
            features_list.append(cat_matrix)
        
        # Assemblage final
        if len(features_list) > 1:
            return np.hstack(features_list)
        elif len(features_list) == 1:
            return features_list[0]
        else:
            return np.zeros((len(self.games_df), 1))

    def _get_ml_feature_names(self) -> List[str]:
        """Retourne les noms des features ML"""
        names = []
        
        # SVD features
        if self.svd:
            names.extend([f"svd_{i}" for i in range(self.svd.n_components)])
        
        # Numeric features
        numeric_cols = [
            "rating_norm", "metacritic_norm", "game_age", "platform_count", 
            "title_length", "genre_count", "quality_score"
        ]
        names.extend([col for col in numeric_cols if col in self.games_df.columns])
        
        # Categorical features
        names.append("main_genre")
        
        return names

    def _train_clustering_models(self):
        """Entraîne les modèles de clustering"""
        # Clustering sur les features textuelles
        self.cluster_labels = self.kmeans.fit_predict(self.textual_features)
        
        # KNN sur les features combinées
        if hasattr(self, 'feature_matrix') and self.feature_matrix is not None:
            self.knn.fit(self.feature_matrix)
        else:
            # Fallback sur features textuelles + numériques
            numeric_features = self.games_df[["rating_norm", "metacritic_norm"]].values
            combined_features = np.hstack([self.textual_features, numeric_features])
            self.knn.fit(combined_features)

    def _create_ensemble_features(self):
        """Crée la matrice de features finale pour l'ensemble"""
        features_list = [self.textual_features]
        
        # Features prédites par les modèles ML
        if hasattr(self.rating_predictor, 'predict'):
            ml_features = self._prepare_ml_features()
            predicted_ratings = self.rating_predictor.predict(ml_features).reshape(-1, 1)
            features_list.append(predicted_ratings)
        
        # Features numériques de base
        numeric_features = self.games_df[["rating_norm", "metacritic_norm"]].values
        features_list.append(numeric_features)
        
        # Features de clustering
        cluster_features = np.eye(self.kmeans.n_clusters)[self.cluster_labels]
        features_list.append(cluster_features)
        
        self.feature_matrix = np.hstack(features_list)

    def predict(self, query: str, k: int = 10, min_confidence: float = 0.1) -> List[Dict]:
        """Prédiction hybride avec tous les modèles"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        cache_key = f"{query}_{k}_{min_confidence}"
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        try:
            # 1. Features textuelles de la requête
            query_tfidf = self.vectorizer.transform([query])
            query_textual = self.svd.transform(query_tfidf)
            
            # 2. Prédiction ML (features synthétiques pour la requête)
            query_ml_features = self._create_query_ml_features(query, query_textual)
            
            if hasattr(self.rating_predictor, 'predict') and query_ml_features is not None:
                predicted_rating = self.rating_predictor.predict([query_ml_features])[0]
                predicted_popularity = self.popularity_model.predict([query_ml_features])[0]
            else:
                predicted_rating = 4.0
                predicted_popularity = 75.0
            
            # 3. Features hybrides pour KNN
            query_features = np.hstack([
                query_textual[0],
                [predicted_rating / 5.0],  # Normalisé
                [predicted_popularity / 100.0]  # Normalisé
            ])
            
            # Padding si nécessaire
            if len(query_features) < self.feature_matrix.shape[1]:
                padding = np.zeros(self.feature_matrix.shape[1] - len(query_features))
                query_features = np.hstack([query_features, padding])
            elif len(query_features) > self.feature_matrix.shape[1]:
                query_features = query_features[:self.feature_matrix.shape[1]]
            
            # 4. Recherche KNN
            distances, indices = self.knn.kneighbors([query_features], n_neighbors=min(k * 2, len(self.games_df)))
            
            # 5. Scoring hybride
            recommendations = []
            for i, idx in enumerate(indices[0]):
                similarity = 1.0 - distances[0][i]
                
                if similarity < min_confidence:
                    continue
                
                row = self.games_df.iloc[idx]
                
                # Score hybride
                text_score = similarity
                rating_bonus = (row["rating"] - 3.0) / 2.0  # Bonus pour bons ratings
                popularity_bonus = (row["metacritic"] - 60) / 40.0  # Bonus pour popularité
                
                hybrid_score = (
                    text_score * 0.5 +
                    rating_bonus * 0.3 +
                    popularity_bonus * 0.2
                )
                
                hybrid_score = max(0.0, min(1.0, hybrid_score))
                
                if hybrid_score >= min_confidence:
                    recommendations.append({
                        "id": int(row["id"]),
                        "title": row["title"],
                        "genres": row["genres"],
                        "confidence": float(hybrid_score),
                        "rating": float(row["rating"]),
                        "metacritic": int(row["metacritic"]),
                        "cluster": int(self.cluster_labels[idx]),
                        "predicted_rating": float(predicted_rating),
                        "algorithm": "hybrid_ml"
                    })
            
            # Tri par score hybride
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            final_recommendations = recommendations[:k]
            
            # Cache et métriques
            self._prediction_cache[cache_key] = final_recommendations
            self._update_metrics(final_recommendations, k)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Hybrid prediction failed: {e}")
            # Fallback sur méthode simple
            return self._fallback_prediction(query, k, min_confidence)

    def _create_query_ml_features(self, query: str, query_textual: np.ndarray) -> Optional[np.ndarray]:
        """Crée des features ML pour une requête"""
        try:
            # Features de base simulées pour la requête
            features = list(query_textual[0])
            
            # Features numériques moyennes
            avg_features = [
                self.games_df["rating_norm"].mean(),
                self.games_df["metacritic_norm"].mean(),
                self.games_df["game_age"].mean(),
                len(query.split()),  # Longueur de la requête
                1,  # Genre count simulé
                self.games_df["quality_score"].mean()
            ]
            
            features.extend(avg_features)
            
            # Feature catégorielle (genre principal simulé)
            features.append(0)  # Genre neutre
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"Could not create ML features for query: {e}")
            return None

    def _fallback_prediction(self, query: str, k: int, min_confidence: float) -> List[Dict]:
        """Méthode de fallback en cas d'erreur"""
        try:
            query_tfidf = self.vectorizer.transform([query])
            query_reduced = self.svd.transform(query_tfidf)
            
            # Similarité cosinus simple
            similarities = cosine_similarity(query_reduced, self.textual_features)[0]
            
            # Top-k
            top_indices = np.argsort(similarities)[::-1][:k]
            
            recommendations = []
            for idx in top_indices:
                if similarities[idx] >= min_confidence:
                    row = self.games_df.iloc[idx]
                    recommendations.append({
                        "id": int(row["id"]),
                        "title": row["title"],
                        "genres": row["genres"],
                        "confidence": float(similarities[idx]),
                        "rating": float(row["rating"]),
                        "metacritic": int(row["metacritic"]),
                        "algorithm": "fallback"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return []

    def _estimate_model_size(self) -> float:
        """Estime la taille du modèle en MB"""
        import sys
        
        size = 0
        for attr in ['vectorizer', 'svd', 'rating_predictor', 'genre_classifier', 
                     'popularity_model', 'kmeans', 'knn', 'feature_matrix']:
            obj = getattr(self, attr, None)
            if obj is not None:
                size += sys.getsizeof(obj)
        
        return size / (1024 * 1024)  # MB

    def get_model_analysis(self) -> Dict:
        """Analyse détaillée du modèle hybride"""
        if not self.is_trained or not self.performance_metrics:
            return {"error": "Model not trained"}
        
        analysis = {
            "model_type": "Hybrid ML (TF-IDF + SVD + Gradient Boosting)",
            "version": self.model_version,
            "performance": {
                "mse": self.performance_metrics.mse,
                "r2_score": self.performance_metrics.r2_score,
                "cross_validation": self.performance_metrics.cross_val_score,
                "training_time": self.performance_metrics.training_time,
                "model_size_mb": self.performance_metrics.model_size_mb
            },
            "feature_importance": self.performance_metrics.feature_importance,
            "components": {
                "textual_features": self.textual_features.shape[1] if self.textual_features is not None else 0,
                "ml_features": self.feature_matrix.shape[1] if self.feature_matrix is not None else 0,
                "clusters": self.kmeans.n_clusters,
                "cached_predictions": len(self._prediction_cache)
            },
            "recommendations": {
                "total_predictions": self.metrics["total_predictions"],
                "avg_confidence": self.metrics["avg_confidence"]
            }
        }
        
        return analysis

    def _update_metrics(self, recommendations: List[Dict], k: int):
        """Met à jour les métriques du modèle"""
        self.metrics["total_predictions"] += 1
        
        if recommendations:
            avg_conf = float(np.mean([r["confidence"] for r in recommendations[:k]]))
            n = self.metrics["total_predictions"]
            self.metrics["avg_confidence"] = (self.metrics["avg_confidence"] * (n - 1) + avg_conf) / n

    # Méthodes existantes conservées pour compatibilité
    def predict_by_game_id(self, game_id: int, k: int = 10) -> List[Dict]:
        """Recommandations par ID de jeu (méthode existante)"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        game_idx = None
        for idx, row in self.games_df.iterrows():
            if int(row["id"]) == game_id:
                game_idx = idx
                break
        
        if game_idx is None:
            raise ValueError(f"Game with id {game_id} not found")
        
        # Utiliser KNN sur la game existante
        game_features = self.feature_matrix[game_idx].reshape(1, -1)
        distances, indices = self.knn.kneighbors(game_features, n_neighbors=k + 1)
        
        recommendations = []
        for i, idx in enumerate(indices[0][1:]):  # Exclure le jeu lui-même
            row = self.games_df.iloc[idx]
            similarity = 1.0 - distances[0][i + 1]
            
            recommendations.append({
                "id": int(row["id"]),
                "title": row["title"],
                "genres": row["genres"],
                "confidence": float(similarity),
                "rating": float(row["rating"]),
                "metacritic": int(row["metacritic"]),
                "algorithm": "hybrid_knn"
            })
        
        return recommendations

    def save_model(self, filepath: str = "model/hybrid_recommendation_model.pkl") -> bool:
        """Sauvegarde du modèle hybride"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                "version": self.model_version,
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "rating_predictor": self.rating_predictor,
                "genre_classifier": self.genre_classifier,
                "popularity_model": self.popularity_model,
                "scaler": self.scaler,
                "genre_encoder": self.genre_encoder,
                "kmeans": self.kmeans,
                "knn": self.knn,
                "textual_features": self.textual_features,
                "feature_matrix": self.feature_matrix,
                "cluster_labels": self.cluster_labels,
                "games_df": self.games_df,
                "performance_metrics": self.performance_metrics,
                "metrics": self.metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Hybrid model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save hybrid model: {e}")
            return False

    def load_model(self, filepath: str = "model/hybrid_recommendation_model.pkl") -> bool:
        """Chargement du modèle hybride"""
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)
            
            # Restaurer tous les composants
            self.model_version = model_data["version"]
            self.vectorizer = model_data["vectorizer"]
            self.svd = model_data["svd"]
            self.rating_predictor = model_data["rating_predictor"]
            self.genre_classifier = model_data["genre_classifier"]
            self.popularity_model = model_data["popularity_model"]
            self.scaler = model_data["scaler"]
            self.genre_encoder = model_data["genre_encoder"]
            self.kmeans = model_data["kmeans"]
            self.knn = model_data["knn"]
            self.textual_features = model_data["textual_features"]
            self.feature_matrix = model_data["feature_matrix"]
            self.cluster_labels = model_data["cluster_labels"]
            self.games_df = model_data["games_df"]
            self.performance_metrics = model_data["performance_metrics"]
            self.metrics = model_data["metrics"]
            
            self.is_trained = True
            self._prediction_cache.clear()
            
            logger.info(f"Hybrid model v{self.model_version} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load hybrid model: {e}")
            return False


# Singleton pour compatibilité
_hybrid_model_instance: Optional[HybridRecommendationModel] = None

def get_hybrid_model() -> HybridRecommendationModel:
    global _hybrid_model_instance
    if _hybrid_model_instance is None:
        _hybrid_model_instance = HybridRecommendationModel()
        
        # Tentative de chargement automatique
        model_path = "model/hybrid_recommendation_model.pkl"
        if os.path.exists(model_path):
            _hybrid_model_instance.load_model(model_path)
    
    return _hybrid_model_instance

def reset_hybrid_model():
    global _hybrid_model_instance
    _hybrid_model_instance = None
