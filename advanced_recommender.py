import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("advanced-recommender")

class HybridRecommender:
    """
    Système de recommandation hybride inspiré de Netflix
    Combine collaborative filtering et content-based
    """
    
    def __init__(self):
        self.collaborative_model = None
        self.content_model = None
        self.user_profiles = {}
        self.item_features = None
        self.user_item_matrix = None
        
    def train_collaborative_filtering(self, interactions_data: pd.DataFrame):
        """
        Entraîne le modèle collaboratif avec NMF (Netflix approach)
        """
        # Créer matrice user-item
        self.user_item_matrix = interactions_data.pivot_table(
            index='user_id', 
            columns='game_id', 
            values='rating',
            fill_value=0
        )
        
        # NMF pour factorisation matricielle
        self.collaborative_model = NMF(
            n_components=50,
            init='random',
            random_state=42,
            max_iter=300
        )
        
        self.user_factors = self.collaborative_model.fit_transform(self.user_item_matrix)
        self.item_factors = self.collaborative_model.components_
        
        logger.info(f"Collaborative model trained with {len(self.user_item_matrix)} users")
    
    def predict_collaborative(self, user_id: int, game_id: int) -> float:
        """Prédit le rating avec le modèle collaboratif"""
        if user_id not in self.user_item_matrix.index:
            return 3.0  # Rating moyen par défaut
            
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        game_idx = self.user_item_matrix.columns.get_loc(game_id) if game_id in self.user_item_matrix.columns else None
        
        if game_idx is None:
            return 3.0
            
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[:, game_idx])
        return min(max(prediction, 1.0), 5.0)
    
    def get_hybrid_recommendations(self, user_id: int, k: int = 10) -> List[Dict]:
        """
        Recommandations hybrides combinant collaborative et content-based
        """
        # 60% collaborative, 40% content-based (Netflix ratio)
        collab_weight = 0.6
        content_weight = 0.4
        
        collaborative_recs = self._get_collaborative_recs(user_id, k * 2)
        content_recs = self._get_content_recs(user_id, k * 2)
        
        # Combiner les scores
        combined_scores = {}
        
        for rec in collaborative_recs:
            game_id = rec['game_id']
            combined_scores[game_id] = {
                'score': rec['score'] * collab_weight,
                'title': rec['title'],
                'genres': rec['genres']
            }
        
        for rec in content_recs:
            game_id = rec['game_id']
            if game_id in combined_scores:
                combined_scores[game_id]['score'] += rec['score'] * content_weight
            else:
                combined_scores[game_id] = {
                    'score': rec['score'] * content_weight,
                    'title': rec['title'],
                    'genres': rec['genres']
                }
        
        # Trier et retourner top-k
        sorted_recs = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:k]
        
        return [
            {
                'game_id': game_id,
                'title': data['title'],
                'genres': data['genres'],
                'confidence': data['score'],
                'algorithm': 'hybrid'
            }
            for game_id, data in sorted_recs
        ]
