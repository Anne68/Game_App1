# model_manager.py - Enhanced ML capabilities with multiple models
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
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger("model-manager")

class ModelType:
    """Types de modèles disponibles"""
    RECOMMENDATION = "recommendation"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    SENTIMENT = "sentiment"
    PREDICTION = "prediction"

class BaseMLModel:
    """Classe de base pour tous les modèles ML"""
    
    def __init__(self, model_type: str, model_version: str = "1.0.0"):
        self.model_type = model_type
        self.model_version = model_version
        self.is_trained = False
        self.created_at = datetime.utcnow()
        self.last_updated = None
        
    def get_info(self) -> Dict[str, Any]:
        return {
            "type": self.model_type,
            "version": self.model_version,
            "is_trained": self.is_trained,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None
        }

class RecommendationModel(BaseMLModel):
    """Modèle de recommandation amélioré avec plusieurs algorithmes"""
    
    def __init__(self, model_version: str = "1.0.0", algorithm: str = "hybrid"):
        super().__init__(ModelType.RECOMMENDATION, model_version)
        self.algorithm = algorithm  # "tfidf", "nmf", "hybrid"
        
        # Composants du modèle
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        self.svd = TruncatedSVD(n_components=50, random_state=42)
        self.nmf = NMF(n_components=30, random_state=42, max_iter=100)
        self.scaler = StandardScaler()
        
        # Données
        self.game_features = None
        self.games_df = None
        self.user_profiles = {}
        self.game_embeddings = None
        
        # Métriques
        self.metrics = {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "last_training": None,
            "model_version": model_version,
            "feature_dim": 0,
            "algorithm": algorithm,
            "training_accuracy": 0.0,
            "diversity_score": 0.0
        }
    
    def prepare_features(self, games: List[Dict]) -> pd.DataFrame:
        """Préparation avancée des features"""
        df = pd.DataFrame(games)
        
        # Features textuelles enrichies
        df['combined_features'] = (
            df['title'].fillna('') + ' ' + 
            df['genres'].fillna('') + ' ' + 
            df['platforms'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).fillna('')
        )
        
        # Features numériques avec outlier handling
        numeric_cols = ['rating', 'metacritic']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
                # Normalisation robuste
                q75, q25 = np.percentile(df[col], [75, 25])
                iqr = q75 - q25
                df[f'{col}_norm'] = (df[col] - q25) / (iqr + 1e-6)
        
        # Features dérivées
        df['title_length'] = df['title'].str.len()
        df['genre_count'] = df['genres'].str.count(',') + 1
        df['platform_count'] = df['platforms'].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        )
        
        # Features catégorielles
        df['is_indie'] = df['genres'].str.contains('Indie', case=False, na=False).astype(int)
        df['is_action'] = df['genres'].str.contains('Action', case=False, na=False).astype(int)
        df['is_rpg'] = df['genres'].str.contains('RPG', case=False, na=False).astype(int)
        
        return df
    
    def train(self, games: List[Dict]) -> Dict:
        """Entraînement avec algorithmes multiples"""
        logger.info(f"Training {self.algorithm} model v{self.model_version} with {len(games)} games")
        
        try:
            # Préparation des données
            self.games_df = self.prepare_features(games)
            
            if self.algorithm == "hybrid":
                return self._train_hybrid_model()
            elif self.algorithm == "nmf":
                return self._train_nmf_model()
            else:  # tfidf par défaut
                return self._train_tfidf_model()
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _train_tfidf_model(self) -> Dict:
        """Modèle TF-IDF + SVD classique"""
        # Vectorisation TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(self.games_df['combined_features'])
        
        # Réduction de dimension
        tfidf_reduced = self.svd.fit_transform(tfidf_matrix)
        
        # Features numériques
        numeric_features = self.games_df[['rating_norm', 'metacritic_norm']].fillna(0).values
        numeric_features = self.scaler.fit_transform(numeric_features)
        
        # Combinaison
        self.game_features = np.hstack([tfidf_reduced, numeric_features])
        self.is_trained = True
        self.last_updated = datetime.utcnow()
        
        return self._compute_training_metrics()
    
    def _train_nmf_model(self) -> Dict:
        """Modèle NMF pour factorisation matricielle"""
        tfidf_matrix = self.vectorizer.fit_transform(self.games_df['combined_features'])
        
        # NMF pour découvrir des topics latents
        nmf_features = self.nmf.fit_transform(tfidf_matrix)
        
        # Features additionnelles
        numeric_features = self.games_df[['rating_norm', 'metacritic_norm']].fillna(0).values
        categorical_features = self.games_df[['is_indie', 'is_action', 'is_rpg']].values
        
        self.game_features = np.hstack([
            nmf_features,
            self.scaler.fit_transform(numeric_features),
            categorical_features
        ])
        
        self.is_trained = True
        self.last_updated = datetime.utcnow()
        
        return self._compute_training_metrics()
    
    def _train_hybrid_model(self) -> Dict:
        """Modèle hybride combinant TF-IDF, NMF et features engineered"""
        tfidf_matrix = self.vectorizer.fit_transform(self.games_df['combined_features'])
        
        # Composantes TF-IDF
        tfidf_reduced = self.svd.fit_transform(tfidf_matrix)
        
        # Composantes NMF 
        nmf_features = self.nmf.fit_transform(tfidf_matrix)
        
        # Features numériques et catégorielles
        numeric_cols = ['rating_norm', 'metacritic_norm', 'title_length', 'genre_count', 'platform_count']
        numeric_features = self.games_df[numeric_cols].fillna(0).values
        numeric_features = self.scaler.fit_transform(numeric_features)
        
        categorical_features = self.games_df[['is_indie', 'is_action', 'is_rpg']].values
        
        # Combinaison pondérée
        self.game_features = np.hstack([
            tfidf_reduced * 0.4,        # 40% TF-IDF
            nmf_features * 0.3,         # 30% NMF  
            numeric_features * 0.2,     # 20% numérique
            categorical_features * 0.1   # 10% catégoriel
        ])
        
        self.is_trained = True
        self.last_updated = datetime.utcnow()
        
        return self._compute_training_metrics()
    
    def _compute_training_metrics(self) -> Dict:
        """Calcul des métriques d'entraînement"""
        metrics = {
            "status": "success",
            "model_version": self.model_version,
            "algorithm": self.algorithm,
            "games_count": len(self.games_df),
            "feature_dim": self.game_features.shape[1],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Métrique de diversité (distance moyenne entre jeux)
        if len(self.games_df) > 1:
            sample_indices = np.random.choice(
                len(self.game_features), 
                min(50, len(self.game_features)), 
                replace=False
            )
            similarities = cosine_similarity(self.game_features[sample_indices])
            # Diversité = 1 - similarité moyenne (excluant diagonale)
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            avg_similarity = similarities[mask].mean()
            diversity = 1 - avg_similarity
            
            metrics["diversity_score"] = float(diversity)
            self.metrics["diversity_score"] = float(diversity)
        
        # Mise à jour des métriques
        self.metrics.update({
            "last_training": metrics["timestamp"],
            "feature_dim": metrics["feature_dim"]
        })
        
        logger.info(f"Training completed. Algorithm: {self.algorithm}, Features: {metrics['feature_dim']}")
        return metrics
    
    def predict(self, query: str, k: int = 10, min_confidence: float = 0.1, 
                user_profile: Optional[Dict] = None) -> List[Dict]:
        """Prédictions avancées avec profil utilisateur optionnel"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Vectorisation de la requête
        query_tfidf = self.vectorizer.transform([query])
        
        if self.algorithm == "nmf":
            query_features = self.nmf.transform(query_tfidf)
        else:
            query_features = self.svd.transform(query_tfidf)
        
        # Features numériques moyennes pour la requête
        numeric_means = self.games_df[['rating_norm', 'metacritic_norm']].mean().values
        
        if self.algorithm == "hybrid":
            # Reconstruction complète pour hybrid
            query_vector = np.hstack([
                query_features * 0.4,
                self.nmf.transform(query_tfidf) * 0.3,
                numeric_means.reshape(1, -1) * 0.2,
                np.array([[0, 0, 0]]) * 0.1  # Features catégorielles neutres
            ])
        else:
            query_vector = np.hstack([query_features[0], numeric_means])
        
        # Calcul des similarités
        similarities = cosine_similarity([query_vector], self.game_features)[0]
        
        # Ajustement avec profil utilisateur
        if user_profile:
            similarities = self._adjust_with_user_profile(similarities, user_profile)
        
        # Création des recommandations
        recommendations = []
        for idx, score in enumerate(similarities):
            if score >= min_confidence:
                game_data = self.games_df.iloc[idx]
                recommendations.append({
                    "id": int(game_data['id']),
                    "title": game_data['title'],
                    "genres": game_data['genres'],
                    "confidence": float(score),
                    "rating": float(game_data.get('rating', 0)),
                    "metacritic": int(game_data.get('metacritic', 0)),
                    "platforms": game_data.get('platforms', []),
                    "algorithm": self.algorithm
                })
        
        # Tri et limitation
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        final_recs = recommendations[:k]
        
        # Mise à jour métriques
        self._update_prediction_metrics(final_recs)
        
        return final_recs
    
    def _adjust_with_user_profile(self, similarities: np.ndarray, 
                                  user_profile: Dict) -> np.ndarray:
        """Ajuste les similarités selon le profil utilisateur"""
        adjusted = similarities.copy()
        
        # Boost basé sur les préférences de genre
        if 'preferred_genres' in user_profile:
            for genre in user_profile['preferred_genres']:
                genre_mask = self.games_df['genres'].str.contains(genre, case=False, na=False)
                adjusted[genre_mask] *= 1.2
        
        # Boost basé sur les plateformes
        if 'preferred_platforms' in user_profile:
            for platform in user_profile['preferred_platforms']:
                platform_mask = self.games_df['platforms'].apply(
                    lambda x: platform in x if isinstance(x, list) else platform in str(x)
                )
                adjusted[platform_mask] *= 1.1
        
        # Pénalité pour les jeux déjà vus
        if 'seen_games' in user_profile:
            seen_ids = set(user_profile['seen_games'])
            for idx, game_id in enumerate(self.games_df['id']):
                if game_id in seen_ids:
                    adjusted[idx] *= 0.5
        
        return adjusted
    
    def predict_by_game_id(self, game_id: int, k: int = 10) -> List[Dict]:
        """Recommandations basées sur un jeu existant"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        game_mask = self.games_df['id'] == game_id
        if not game_mask.any():
            raise ValueError(f"Game with id {game_id} not found")
        
        game_idx = game_mask.idxmax()
        game_vector = self.game_features[game_idx]
        
        # Similarités
        similarities = cosine_similarity([game_vector], self.game_features)[0]
        similarities[game_idx] = -1  # Exclure le jeu lui-même
        
        # Top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:
                game_data = self.games_df.iloc[idx]
                recommendations.append({
                    "id": int(game_data['id']),
                    "title": game_data['title'], 
                    "genres": game_data['genres'],
                    "confidence": float(similarities[idx]),
                    "rating": float(game_data.get('rating', 0)),
                    "metacritic": int(game_data.get('metacritic', 0)),
                    "algorithm": self.algorithm
                })
        
        return recommendations
    
    def _update_prediction_metrics(self, recommendations: List[Dict]):
        """Mise à jour des métriques de prédiction"""
        self.metrics["total_predictions"] += 1
        
        if recommendations:
            confidences = [r['confidence'] for r in recommendations]
            avg_conf = np.mean(confidences)
            
            # Moyenne mobile
            alpha = 0.1
            self.metrics["avg_confidence"] = (
                alpha * avg_conf + (1 - alpha) * self.metrics["avg_confidence"]
            )

class GameClassificationModel(BaseMLModel):
    """Modèle de classification de jeux par catégories"""
    
    def __init__(self, model_version: str = "1.0.0"):
        super().__init__(ModelType.CLASSIFICATION, model_version)
        self.vectorizer = TfidfVectorizer(max_features=150, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.feature_importance = None
        self.classes = None
        self.accuracy = 0.0
    
    def train(self, games: List[Dict]) -> Dict:
        """Entraîne le classificateur de genres"""
        df = pd.DataFrame(games)
        
        # Préparation des features
        text_features = self.vectorizer.fit_transform(
            df['title'] + ' ' + df['genres'].fillna('')
        )
        
        numeric_features = df[['rating', 'metacritic']].fillna(0)
        numeric_features = self.scaler.fit_transform(numeric_features)
        
        # Combinaison des features
        X = np.hstack([text_features.toarray(), numeric_features])
        
        # Labels (genre principal)
        y = df['genres'].fillna('Unknown').apply(lambda x: x.split(',')[0].strip())
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Entraînement
        self.classifier.fit(X_train, y_train)
        
        # Évaluation
        self.accuracy = self.classifier.score(X_test, y_test)
        self.feature_importance = self.classifier.feature_importances_
        self.classes = self.label_encoder.classes_
        
        self.is_trained = True
        self.last_updated = datetime.utcnow()
        
        logger.info(f"Classification model trained. Accuracy: {self.accuracy:.3f}")
        
        return {
            "status": "success",
            "accuracy": self.accuracy,
            "classes": list(self.classes),
            "n_features": X.shape[1]
        }
    
    def predict(self, game_description: str, title: str = "", 
                rating: float = 0.0, metacritic: int = 0) -> Dict:
        """Prédit la catégorie d'un jeu"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Features textuelles
        text = f"{title} {game_description}"
        text_features = self.vectorizer.transform([text])
        
        # Features numériques
        numeric_features = self.scaler.transform([[rating, metacritic]])
        
        # Combinaison
        X = np.hstack([text_features.toarray(), numeric_features])
        
        # Prédiction
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        
        # Résultats
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        
        # Top 3 classes probables
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        
        for idx in top_indices:
            class_name = self.label_encoder.inverse_transform([idx])[0]
            prob = float(probabilities[idx])
            top_predictions.append({
                "class": class_name,
                "probability": prob
            })
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "model_version": self.model_version
        }

class GameClusteringModel(BaseMLModel):
    """Modèle de clustering pour segmentation de jeux"""
    
    def __init__(self, n_clusters: int = 8, model_version: str = "1.0.0"):
        super().__init__(ModelType.CLUSTERING, model_version)
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.scaler = StandardScaler()
        
        self.cluster_centers = None
        self.cluster_labels = None
        self.cluster_descriptions = {}
    
    def train(self, games: List[Dict]) -> Dict:
        """Entraîne le modèle de clustering"""
        df = pd.DataFrame(games)
        
        # Features pour clustering
        text_features = self.vectorizer.fit_transform(
            df['title'] + ' ' + df['genres'].fillna('')
        )
        
        numeric_features = df[['rating', 'metacritic']].fillna(
            [df['rating'].median(), df['metacritic'].median()]
        )
        numeric_features = self.scaler.fit_transform(numeric_features)
        
        # Combinaison
        X = np.hstack([text_features.toarray(), numeric_features])
        
        # Clustering
        self.cluster_labels = self.kmeans.fit_predict(X)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Analyse des clusters
        df['cluster'] = self.cluster_labels
        self._analyze_clusters(df)
        
        self.is_trained = True
        self.last_updated = datetime.utcnow()
        
        logger.info(f"Clustering model trained with {self.n_clusters} clusters")
        
        return {
            "status": "success",
            "n_clusters": self.n_clusters,
            "inertia": float(self.kmeans.inertia_),
            "cluster_descriptions": self.cluster_descriptions
        }
    
    def _analyze_clusters(self, df: pd.DataFrame):
        """Analyse et description des clusters"""
        for cluster_id in range(self.n_clusters):
            cluster_games = df[df['cluster'] == cluster_id]
            
            if len(cluster_games) == 0:
                continue
            
            # Statistiques du cluster
            avg_rating = cluster_games['rating'].mean()
            avg_metacritic = cluster_games['metacritic'].mean()
            
            # Genres les plus fréquents
            all_genres = ' '.join(cluster_games['genres'].fillna(''))
            common_words = []
            for word in ['Action', 'RPG', 'Strategy', 'Indie', 'Adventure', 'Simulation']:
                if word.lower() in all_genres.lower():
                    common_words.append(word)
            
            # Échantillon de jeux
            sample_titles = cluster_games['title'].head(3).tolist()
            
            self.cluster_descriptions[cluster_id] = {
                "size": len(cluster_games),
                "avg_rating": float(avg_rating),
                "avg_metacritic": float(avg_metacritic),
                "common_genres": common_words,
                "sample_games": sample_titles
            }
    
    def predict(self, game_description: str, title: str = "", 
                rating: float = 0.0, metacritic: int = 0) -> Dict:
        """Prédit le cluster d'appartenance d'un jeu"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Préparation des features
        text = f"{title} {game_description}"
        text_features = self.vectorizer.transform([text])
        numeric_features = self.scaler.transform([[rating, metacritic]])
        
        X = np.hstack([text_features.toarray(), numeric_features])
        
        # Prédiction
        cluster = self.kmeans.predict(X)[0]
        distances = self.kmeans.transform(X)[0]
        confidence = 1.0 / (1.0 + distances[cluster])  # Confiance basée sur la distance
        
        return {
            "cluster": int(cluster),
            "confidence": float(confidence),
            "cluster_description": self.cluster_descriptions.get(cluster, {}),
            "distances_to_centers": distances.tolist()
        }

# Gestionnaire de modèles global
class MLModelManager:
    """Gestionnaire centralisé pour tous les modèles ML"""
    
    def __init__(self):
        self.models = {}
        self.default_models = {
            ModelType.RECOMMENDATION: RecommendationModel,
            ModelType.CLASSIFICATION: GameClassificationModel,
            ModelType.CLUSTERING: GameClusteringModel
        }
    
    def create_model(self, model_type: str, model_id: str = None, **kwargs) -> BaseMLModel:
        """Crée une instance de modèle"""
        if model_type not in self.default_models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_id = model_id or f"{model_type}_default"
        model_class = self.default_models[model_type]
        model = model_class(**kwargs)
        
        self.models[model_id] = model
        return model
    
    def get_model(self, model_id: str) -> Optional[BaseMLModel]:
        """Récupère un modèle par ID"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[Dict]:
        """Liste tous les modèles disponibles"""
        return [
            {
                "id": model_id,
                "info": model.get_info()
            }
            for model_id, model in self.models.items()
        ]
    
    def save_model(self, model_id: str, filepath: str = None) -> bool:
        """Sauvegarde un modèle"""
        model = self.get_model(model_id)
        if not model:
            return False
        
        if filepath is None:
            filepath = f"model/{model_id}_{model.model_type}.pkl"
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model {model_id} saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str, filepath: str) -> bool:
        """Charge un modèle depuis un fichier"""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            self.models[model_id] = model
            logger.info(f"Model {model_id} loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False

# Instance globale pour compatibilité
_model_manager = MLModelManager()
_recommendation_model = None

def get_model() -> RecommendationModel:
    """Retourne le modèle de recommandation principal (compatibilité)"""
    global _recommendation_model
    if _recommendation_model is None:
        _recommendation_model = RecommendationModel()
        # Essayer de charger un modèle existant
        if os.path.exists("model/recommendation_model.pkl"):
            try:
                with open("model/recommendation_model.pkl", 'rb') as f:
                    _recommendation_model = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
    return _recommendation_model

def get_model_manager() -> MLModelManager:
    """Retourne le gestionnaire de modèles"""
    return _model_manager
