# test_model.py - Tests automatisés pour le modèle de recommandation
import pytest
import numpy as np
from model_manager import RecommendationModel
import pandas as pd

# Données de test
SAMPLE_GAMES = [
    {"id": 1, "title": "The Witcher 3", "genres": "RPG, Action", "rating": 4.9, "metacritic": 93, "platforms": ["PC", "PS4"]},
    {"id": 2, "title": "Hades", "genres": "Action, Roguelike", "rating": 4.8, "metacritic": 93, "platforms": ["PC", "Switch"]},
    {"id": 3, "title": "Stardew Valley", "genres": "Simulation, RPG", "rating": 4.7, "metacritic": 89, "platforms": ["PC", "Switch"]},
    {"id": 4, "title": "Celeste", "genres": "Platformer, Indie", "rating": 4.6, "metacritic": 94, "platforms": ["PC", "Switch"]},
    {"id": 5, "title": "Doom Eternal", "genres": "Action, FPS", "rating": 4.5, "metacritic": 88, "platforms": ["PC", "PS4"]},
]

class TestRecommendationModel:
    """Tests unitaires pour le modèle de recommandation"""
    
    @pytest.fixture
    def model(self):
        """Fixture pour créer une instance du modèle"""
        return RecommendationModel(model_version="test-1.0.0")
    
    @pytest.fixture
    def trained_model(self, model):
        """Fixture pour un modèle entraîné"""
        model.train(SAMPLE_GAMES)
        return model
    
    def test_model_initialization(self, model):
        """Test l'initialisation du modèle"""
        assert model.model_version == "test-1.0.0"
        assert not model.is_trained
        assert model.game_features is None
        assert model.metrics["total_predictions"] == 0
    
    def test_prepare_features(self, model):
        """Test la préparation des features"""
        df = model.prepare_features(SAMPLE_GAMES)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(SAMPLE_GAMES)
        assert 'combined_features' in df.columns
        assert 'rating_norm' in df.columns
        assert 'metacritic_norm' in df.columns
        
        # Vérifier la normalisation
        assert df['rating_norm'].min() >= 0
        assert df['rating_norm'].max() <= 1
    
    def test_model_training(self, model):
        """Test l'entraînement du modèle"""
        result = model.train(SAMPLE_GAMES)
        
        assert result["status"] == "success"
        assert model.is_trained
        assert model.game_features is not None
        assert model.game_features.shape[0] == len(SAMPLE_GAMES)
        assert model.metrics["last_training"] is not None
        assert "validation_metrics" in result
    
    def test_prediction_without_training(self, model):
        """Test la prédiction sans entraînement"""
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict("RPG games")
    
    def test_prediction_basic(self, trained_model):
        """Test une prédiction basique"""
        recommendations = trained_model.predict("Action RPG", k=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        for rec in recommendations:
            assert "title" in rec
            assert "confidence" in rec
            assert 0 <= rec["confidence"] <= 1
    
    def test_prediction_confidence_threshold(self, trained_model):
        """Test le seuil de confiance"""
        # Avec un seuil très élevé
        recommendations = trained_model.predict("Action", k=10, min_confidence=0.99)
        assert len(recommendations) == 0 or all(r["confidence"] >= 0.99 for r in recommendations)
        
        # Avec un seuil bas
        recommendations = trained_model.predict("Action", k=10, min_confidence=0.01)
        assert len(recommendations) > 0
    
    def test_predict_by_game_id(self, trained_model):
        """Test la recommandation par ID de jeu"""
        recommendations = trained_model.predict_by_game_id(1, k=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        # Le jeu lui-même ne doit pas être dans les recommandations
        assert all(rec["title"] != "The Witcher 3" for rec in recommendations)
    
    def test_predict_by_invalid_game_id(self, trained_model):
        """Test avec un ID de jeu invalide"""
        with pytest.raises(ValueError, match="Game with id .* not found"):
            trained_model.predict_by_game_id(999)
    
    def test_metrics_update(self, trained_model):
        """Test la mise à jour des métriques"""
        initial_predictions = trained_model.metrics["total_predictions"]
        
        trained_model.predict("RPG", k=5)
        
        assert trained_model.metrics["total_predictions"] == initial_predictions + 1
        assert trained_model.metrics["avg_confidence"] > 0
    
    def test_model_save_load(self, trained_model, tmp_path):
        """Test la sauvegarde et le chargement du modèle"""
        filepath = tmp_path / "test_model.pkl"
        
        # Sauvegarder
        assert trained_model.save_model(str(filepath))
        
        # Créer un nouveau modèle et charger
        new_model = RecommendationModel()
        assert new_model.load_model(str(filepath))
        
        # Vérifier que le modèle chargé fonctionne
        assert new_model.is_trained
        assert new_model.model_version == trained_model.model_version
        recommendations = new_model.predict("Action", k=3)
        assert len(recommendations) > 0
    
    def test_get_metrics(self, trained_model):
        """Test l'obtention des métriques"""
        metrics = trained_model.get_metrics()
        
        assert "model_version" in metrics
        assert "is_trained" in metrics
        assert "games_count" in metrics
        assert metrics["is_trained"] is True
        assert metrics["games_count"] == len(SAMPLE_GAMES)


class TestModelValidation:
    """Tests de validation des données et du modèle"""
    
    def test_empty_games_list(self):
        """Test avec une liste de jeux vide"""
        model = RecommendationModel()
        with pytest.raises(Exception):
            model.train([])
    
    def test_malformed_game_data(self):
        """Test avec des données malformées"""
        model = RecommendationModel()
        malformed_games = [
            {"id": 1, "title": "Game 1"},  # Manque des champs
        ]
        with pytest.raises(Exception):
            model.train(malformed_games)
    
    def test_prediction_determinism(self):
        """Test que les prédictions sont déterministes"""
        model1 = RecommendationModel()
        model1.train(SAMPLE_GAMES)
        
        model2 = RecommendationModel()
        model2.train(SAMPLE_GAMES)
        
        recs1 = model1.predict("Action RPG", k=5)
        recs2 = model2.predict("Action RPG", k=5)
        
        # Les titres devraient être les mêmes dans le même ordre
        titles1 = [r["title"] for r in recs1]
        titles2 = [r["title"] for r in recs2]
        assert titles1 == titles2


class TestModelPerformance:
    """Tests de performance du modèle"""
    
    def test_training_time(self, model):
        """Test que l'entraînement est rapide"""
        import time
        
        start = time.time()
        model.train(SAMPLE_GAMES)
        duration = time.time() - start
        
        # L'entraînement devrait prendre moins de 2 secondes pour un petit dataset
        assert duration < 2.0
    
    def test_prediction_time(self, trained_model):
        """Test que la prédiction est rapide"""
        import time
        
        start = time.time()
        for _ in range(10):
            trained_model.predict("Action RPG", k=5)
        duration = time.time() - start
        
        # 10 prédictions devraient prendre moins d'1 seconde
        assert duration < 1.0
    
    def test_memory_usage(self, trained_model):
        """Test que le modèle n'utilise pas trop de mémoire"""
        import sys
        
        # Approximation de la taille du modèle
        model_size = sys.getsizeof(trained_model.game_features) if trained_model.game_features is not None else 0
        
        # Pour un petit dataset, le modèle devrait être < 10MB
        assert model_size < 10 * 1024 * 1024


@pytest.mark.parametrize("query,expected_genre", [
    ("RPG fantasy", "RPG"),
    ("Fast action shooter", "FPS"),
    ("Farming simulation", "Simulation"),
    ("Platform indie", "Platformer"),
])
def test_genre_relevance(query, expected_genre):
    """Test que les recommandations sont pertinentes par genre"""
    model = RecommendationModel()
    model.train(SAMPLE_GAMES)
    
    recommendations = model.predict(query, k=3)
    
    # Au moins une recommandation devrait contenir le genre attendu
    genres = [rec["genres"] for rec in recommendations]
    assert any(expected_genre in g for g in genres)
