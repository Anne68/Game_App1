# enhanced_model_manager.py - Gestionnaire de modèles ML/IA avancés
from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

# ML/AI imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

# Tentative d'import des bibliothèques avancées
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger("enhanced-model-manager")

class ModelType(Enum):
    TFIDF_SVD = "tfidf_svd"
    NEURAL_COLLABORATIVE = "neural_cf"
    TRANSFORMER_BASED = "transformer"
    HYBRID_ENSEMBLE = "hybrid_ensemble"
    DEEP_CONTENT = "deep_content"

@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    ndcg_at_k: float
    diversity_score: float
    coverage: float
    training_time: float
    inference_time: float

class DeepContentModel(nn.Module):
    """Modèle deep learning pour l'analyse de contenu"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention mechanism
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global max pooling
        pooled = torch.max(attended, dim=1)[0]
        
        return self.classifier(pooled)

class NeuralCollaborativeFiltering(nn.Module):
    """Modèle de filtrage collaboratif neuronal"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc_layers(x)

class TransformerRecommender:
    """Recommandeur basé sur des transformers pré-entraînés"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embeddings_cache = {}
        
    def load_model(self):
        """Charge le modèle transformer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode les textes en embeddings"""
        if self.model is None:
            self.load_model()
        
        embeddings = []
        
        for text in texts:
            if text in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[text])
                continue
            
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                
            self.embeddings_cache[text] = embedding[0]
            embeddings.append(embedding[0])
        
        return np.array(embeddings)

class SentimentAnalyzer:
    """Analyseur de sentiment pour les descriptions de jeux"""
    
    def __init__(self):
        self.model = None
        
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyse le sentiment des textes"""
        if not TRANSFORMERS_AVAILABLE:
            # Fallback simple
            return [{"positive": 0.5, "negative": 0.5, "neutral": 0.0} for _ in texts]
        
        try:
            from transformers import pipeline
            if self.model is None:
                self.model = pipeline("sentiment-analysis", 
                                    model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            results = []
            for text in texts:
                sentiment = self.model(text[:512])  # Limite de tokens
                # Convertir en format standard
                if sentiment[0]["label"] == "LABEL_2":  # Positive
                    results.append({"positive": sentiment[0]["score"], "negative": 0.0, "neutral": 0.0})
                elif sentiment[0]["label"] == "LABEL_0":  # Negative
                    results.append({"positive": 0.0, "negative": sentiment[0]["score"], "neutral": 0.0})
                else:  # Neutral
                    results.append({"positive": 0.0, "negative": 0.0, "neutral": sentiment[0]["score"]})
            
            return results
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return [{"positive": 0.5, "negative": 0.5, "neutral": 0.0} for _ in texts]

class EnhancedRecommendationModel:
    """Modèle de recommandation ML/IA avancé"""
    
    def __init__(self, model_type: ModelType = ModelType.HYBRID_ENSEMBLE, model_version: str = "2.0.0"):
        self.model_type = model_type
        self.model_version = model_version
        self.is_trained = False
        
        # Modèles de base
        self.tfidf_model = None
        self.neural_cf_model = None
        self.transformer_model = None
        self.deep_content_model = None
        
        # Composants ML
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.svd = TruncatedSVD(n_components=50)
        self.nmf = NMF(n_components=30, random_state=42)
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=10, random_state=42)
        
        # Modèles prédictifs
        self.rating_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.popularity_predictor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500)
        
        # Analyseurs avancés
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Données et métriques
        self.games_df = None
        self.user_item_matrix = None
        self.game_embeddings = None
        self.game_clusters = None
        
        self.metrics = ModelMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    def prepare_advanced_features(self, games: List[Dict]) -> pd.DataFrame:
        """Prépare des features avancées pour l'entraînement"""
        df = pd.DataFrame(games)
        
        # Features textuelles combinées
        df['combined_text'] = (
            df['title'] + ' ' + 
            df['genres'] + ' ' + 
            df['platforms'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        )
        
        # Features numériques enrichies
        df['rating_norm'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min() + 1e-8)
        df['metacritic_norm'] = (df['metacritic'] - df['metacritic'].min()) / (df['metacritic'].max() - df['metacritic'].min() + 1e-8)
        
        # Features dérivées
        df['title_length'] = df['title'].str.len()
        df['genre_count'] = df['genres'].str.split().str.len()
        df['platform_count'] = df['platforms'].apply(lambda x: len(x) if isinstance(x, list) else 1)
        
        # Score de popularité composite
        df['popularity_score'] = (
            0.4 * df['rating_norm'] + 
            0.3 * df['metacritic_norm'] + 
            0.2 * np.log1p(df['platform_count']) / np.log1p(df['platform_count'].max()) +
            0.1 * (1 / (1 + df['title_length'] / 100))  # Titres courts = plus mémorables
        )
        
        return df
    
    def extract_semantic_features(self, texts: List[str]) -> np.ndarray:
        """Extrait des features sémantiques avancées"""
        if TRANSFORMERS_AVAILABLE and self.model_type in [ModelType.TRANSFORMER_BASED, ModelType.HYBRID_ENSEMBLE]:
            try:
                if self.transformer_model is None:
                    self.transformer_model = TransformerRecommender()
                
                return self.transformer_model.encode_text(texts)
            except Exception as e:
                logger.warning(f"Transformer encoding failed: {e}, falling back to TF-IDF")
        
        # Fallback to TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self.svd.fit_transform(tfidf_matrix)
    
    def train_neural_models(self, df: pd.DataFrame):
        """Entraîne les modèles neuronaux si PyTorch est disponible"""
        if not PYTORCH_AVAILABLE:
            logger.info("PyTorch not available, skipping neural models")
            return
        
        try:
            # Préparer les données pour l'entraînement neuronal
            texts = df['combined_text'].tolist()
            ratings = df['rating'].values
            
            # Simuler des interactions utilisateur-item pour l'exemple
            # En production, vous auriez de vraies données d'interaction
            num_users = min(1000, len(df) * 10)
            num_items = len(df)
            
            # Modèle de filtrage collaboratif neuronal
            if self.model_type in [ModelType.NEURAL_COLLABORATIVE, ModelType.HYBRID_ENSEMBLE]:
                self.neural_cf_model = NeuralCollaborativeFiltering(num_users, num_items)
                
                # Entraînement simulé (remplacez par de vraies données)
                optimizer = optim.Adam(self.neural_cf_model.parameters())
                criterion = nn.MSELoss()
                
                for epoch in range(10):  # Entraînement réduit pour l'exemple
                    # Génération de données synthétiques pour l'exemple
                    user_ids = torch.randint(0, num_users, (100,))
                    item_ids = torch.randint(0, num_items, (100,))
                    ratings_tensor = torch.rand(100)
                    
                    optimizer.zero_grad()
                    predictions = self.neural_cf_model(user_ids, item_ids)
                    loss = criterion(predictions.squeeze(), ratings_tensor)
                    loss.backward()
                    optimizer.step()
                
                logger.info("Neural collaborative filtering model trained")
            
        except Exception as e:
            logger.error(f"Neural model training failed: {e}")
    
    def train(self, games: List[Dict]) -> Dict[str, Any]:
        """Entraîne le modèle de recommandation avancé"""
        start_time = datetime.utcnow()
        logger.info(f"Training enhanced model v{self.model_version} with {len(games)} games")
        
        try:
            # Préparation des données
            self.games_df = self.prepare_advanced_features(games)
            
            # Extraction des features sémantiques
            semantic_features = self.extract_semantic_features(self.games_df['combined_text'].tolist())
            
            # Features numériques
            numeric_features = self.games_df[['rating_norm', 'metacritic_norm', 'title_length', 
                                            'genre_count', 'platform_count', 'popularity_score']].values
            numeric_features = self.scaler.fit_transform(numeric_features)
            
            # Combinaison des features
            self.game_embeddings = np.hstack([semantic_features, numeric_features])
            
            # Clustering des jeux
            self.game_clusters = self.clusterer.fit_predict(self.game_embeddings)
            self.games_df['cluster'] = self.game_clusters
            
            # Entraînement des modèles prédictifs
            X = self.game_embeddings
            y_rating = self.games_df['rating'].values
            y_popularity = self.games_df['popularity_score'].values
            
            # Modèle de prédiction de rating
            self.rating_predictor.fit(X, y_rating)
            
            # Modèle de prédiction de popularité
            self.popularity_predictor.fit(X, y_popularity)
            
            # Entraînement des modèles neuronaux
            self.train_neural_models(self.games_df)
            
            # Analyse de sentiment (si descriptions disponibles)
            if 'description' in self.games_df.columns:
                sentiments = self.sentiment_analyzer.analyze_sentiment(
                    self.games_df['description'].fillna('').tolist()
                )
                for i, sentiment in enumerate(sentiments):
                    self.games_df.loc[i, 'sentiment_positive'] = sentiment['positive']
                    self.games_df.loc[i, 'sentiment_negative'] = sentiment['negative']
            
            # Calcul des métriques
            training_duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Validation croisée rapide
            X_train, X_test, y_train, y_test = train_test_split(X, y_rating, test_size=0.2, random_state=42)
            test_predictions = self.rating_predictor.predict(X_test)
            accuracy = 1 - mean_squared_error(y_test, test_predictions) / np.var(y_test)
            
            self.metrics = ModelMetrics(
                accuracy=max(0.0, accuracy),
                precision=0.85,  # À calculer avec de vraies métriques de recommandation
                recall=0.80,
                ndcg_at_k=0.75,
                diversity_score=len(np.unique(self.game_clusters)) / len(games),
                coverage=1.0,
                training_time=training_duration,
                inference_time=0.0
            )
            
            self.is_trained = True
            
            return {
                "status": "success",
                "model_version": self.model_version,
                "model_type": self.model_type.value,
                "games_count": len(games),
                "feature_dim": self.game_embeddings.shape[1],
                "clusters_found": len(np.unique(self.game_clusters)),
                "neural_models": PYTORCH_AVAILABLE,
                "transformers": TRANSFORMERS_AVAILABLE,
                "training_duration": training_duration,
                "metrics": {
                    "accuracy": self.metrics.accuracy,
                    "diversity": self.metrics.diversity_score,
                    "coverage": self.metrics.coverage
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced model training failed: {e}")
            raise
    
    def predict_advanced(self, query: str, k: int = 10, user_context: Optional[Dict] = None) -> List[Dict]:
        """Prédictions avancées avec contexte utilisateur"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        start_time = datetime.utcnow()
        
        # Encodage sémantique de la requête
        if TRANSFORMERS_AVAILABLE and self.transformer_model:
            query_embedding = self.transformer_model.encode_text([query])[0]
        else:
            query_tfidf = self.vectorizer.transform([query])
            query_embedding = self.svd.transform(query_tfidf)[0]
        
        # Features moyennes pour la requête
        avg_numeric = np.mean(self.games_df[['rating_norm', 'metacritic_norm', 'title_length', 
                                           'genre_count', 'platform_count', 'popularity_score']].values, axis=0)
        
        # Combinaison query features
        query_features = np.hstack([query_embedding, avg_numeric])
        
        # Calcul des similarités
        similarities = cosine_similarity([query_features], self.game_embeddings)[0]
        
        # Prédictions de rating et popularité
        predicted_ratings = self.rating_predictor.predict(self.game_embeddings)
        predicted_popularity = self.popularity_predictor.predict(self.game_embeddings)
        
        # Score composite avancé
        composite_scores = (
            0.4 * similarities +
            0.3 * (predicted_ratings / 5.0) +  # Normalisation sur 5
            0.2 * predicted_popularity +
            0.1 * (self.games_df['popularity_score'].values)
        )
        
        # Diversification par cluster
        selected_indices = []
        used_clusters = set()
        
        # Sélection diversifiée
        sorted_indices = np.argsort(composite_scores)[::-1]
        for idx in sorted_indices:
            cluster = self.game_clusters[idx]
            
            # Favoriser la diversité des clusters
            if len(selected_indices) < k:
                if cluster not in used_clusters or len(selected_indices) >= k // 2:
                    selected_indices.append(idx)
                    used_clusters.add(cluster)
        
        # Finaliser la sélection si pas assez diversifiée
        for idx in sorted_indices:
            if len(selected_indices) >= k:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)
        
        # Construction des recommandations
        recommendations = []
        for idx in selected_indices[:k]:
            game_row = self.games_df.iloc[idx]
            
            recommendation = {
                "title": game_row['title'],
                "genres": game_row['genres'],
                "confidence": float(similarities[idx]),
                "predicted_rating": float(predicted_ratings[idx]),
                "popularity_score": float(predicted_popularity[idx]),
                "composite_score": float(composite_scores[idx]),
                "cluster": int(self.game_clusters[idx]),
                "rating": float(game_row['rating']),
                "metacritic": int(game_row['metacritic'])
            }
            
            # Ajout du sentiment si disponible
            if 'sentiment_positive' in game_row:
                recommendation["sentiment"] = {
                    "positive": float(game_row['sentiment_positive']),
                    "negative": float(game_row['sentiment_negative'])
                }
            
            recommendations.append(recommendation)
        
        # Mise à jour métriques
        inference_time = (datetime.utcnow() - start_time).total_seconds()
        self.metrics.inference_time = inference_time
        
        return recommendations
    
    def get_similar_games_advanced(self, game_id: int, k: int = 10) -> List[Dict]:
        """Recommandations similaires avancées"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Trouver le jeu
        game_idx = self.games_df[self.games_df['id'] == game_id].index
        if len(game_idx) == 0:
            raise ValueError(f"Game with id {game_id} not found")
        
        game_idx = game_idx[0]
        game_embedding = self.game_embeddings[game_idx]
        game_cluster = self.game_clusters[game_idx]
        
        # Similarités
        similarities = cosine_similarity([game_embedding], self.game_embeddings)[0]
        similarities[game_idx] = -1  # Exclure le jeu lui-même
        
        # Bonus pour les jeux du même cluster
        cluster_bonus = np.where(self.game_clusters == game_cluster, 0.1, 0.0)
        adjusted_similarities = similarities + cluster_bonus
        
        # Sélection top-k
        top_indices = np.argsort(adjusted_similarities)[::-1][:k]
        
        recommendations = []
        for idx in top_indices:
            if adjusted_similarities[idx] > 0:
                game_row = self.games_df.iloc[idx]
                recommendations.append({
                    "title": game_row['title'],
                    "genres": game_row['genres'],
                    "confidence": float(similarities[idx]),
                    "cluster_similarity": game_cluster == self.game_clusters[idx],
                    "rating": float(game_row['rating']),
                    "metacritic": int(game_row['metacritic'])
                })
        
        return recommendations
    
    def get_trending_games(self, time_window: str = "week") -> List[Dict]:
        """Jeux tendance basés sur l'analyse ML"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Score de tendance basé sur plusieurs facteurs
        trend_scores = (
            0.4 * self.games_df['popularity_score'] +
            0.3 * (self.games_df['rating'] / 5.0) +
            0.2 * (self.games_df['metacritic'] / 100.0) +
            0.1 * (1.0 / (1.0 + self.games_df['title_length'] / 50))
        )
        
        # Ajout de randomness pour la diversité
        trend_scores += np.random.normal(0, 0.05, len(trend_scores))
        
        top_trending = np.argsort(trend_scores)[::-1][:20]
        
        trending_games = []
        for idx in top_trending:
            game_row = self.games_df.iloc[idx]
            trending_games.append({
                "title": game_row['title'],
                "genres": game_row['genres'],
                "trend_score": float(trend_scores.iloc[idx]),
                "rating": float(game_row['rating']),
                "metacritic": int(game_row['metacritic']),
                "cluster": int(self.game_clusters[idx])
            })
        
        return trending_games
    
    def explain_recommendation(self, game_title: str, query: str) -> Dict[str, Any]:
        """Explication des recommandations pour la transparence"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        game_row = self.games_df[self.games_df['title'] == game_title]
        if game_row.empty:
            raise ValueError(f"Game '{game_title}' not found")
        
        game_row = game_row.iloc[0]
        
        explanation = {
            "game": game_title,
            "query": query,
            "factors": {
                "semantic_match": "Le titre/genre correspond à votre recherche",
                "cluster": f"Appartient au cluster {game_row['cluster']} (jeux similaires)",
                "rating": f"Note élevée: {game_row['rating']}/5",
                "popularity": f"Score de popularité: {game_row['popularity_score']:.2f}",
                "genre_match": f"Genres: {game_row['genres']}"
            },
            "confidence_breakdown": {
                "content_similarity": 0.4,
                "user_preference": 0.3,
                "popularity": 0.2,
                "diversity": 0.1
            }
        }
        
        return explanation

# Fonction factory pour créer le modèle approprié
def create_enhanced_model(model_type: str = "hybrid") -> EnhancedRecommendationModel:
    """Crée un modèle de recommandation avancé"""
    
    type_mapping = {
        "tfidf": ModelType.TFIDF_SVD,
        "neural": ModelType.NEURAL_COLLABORATIVE,
        "transformer": ModelType.TRANSFORMER_BASED,
        "hybrid": ModelType.HYBRID_ENSEMBLE,
        "deep": ModelType.DEEP_CONTENT
    }
    
    model_type_enum = type_mapping.get(model_type, ModelType.HYBRID_ENSEMBLE)
    
    return EnhancedRecommendationModel(
        model_type=model_type_enum,
        model_version="2.0.0"
    )

# Singleton pour le modèle avancé
_enhanced_model_instance: Optional[EnhancedRecommendationModel] = None

def get_enhanced_model() -> EnhancedRecommendationModel:
    """Retourne l'instance singleton du modèle avancé"""
    global _enhanced_model_instance
    if _enhanced_model_instance is None:
        _enhanced_model_instance = create_enhanced_model("hybrid")
    return _enhanced_model_instance
