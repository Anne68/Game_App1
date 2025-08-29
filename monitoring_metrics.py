# monitoring.py - Métriques personnalisées pour le monitoring du modèle
from __future__ import annotations

import time
import logging
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
from prometheus_client import Counter, Histogram, Gauge, Info
import numpy as np

logger = logging.getLogger("monitoring")

# ========= Métriques Prometheus pour ML =========

# Métriques de base
model_info = Info('model_info', 'Information about the model')
model_training_counter = Counter('model_trainings_total', 'Total number of model trainings')
model_prediction_counter = Counter('model_predictions_total', 'Total number of predictions', ['endpoint', 'status'])
model_load_counter = Counter('model_loads_total', 'Total number of model loads', ['status'])

# Métriques de performance
prediction_latency = Histogram('prediction_latency_seconds', 'Latency of model predictions', ['endpoint'])
training_duration = Histogram('training_duration_seconds', 'Duration of model training')
model_confidence = Histogram('model_confidence_score', 'Confidence scores of predictions', buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))

# Métriques de qualité
model_accuracy_gauge = Gauge('model_accuracy', 'Current model accuracy')
model_feature_dimension = Gauge('model_feature_dimension', 'Number of features in the model')
model_games_count = Gauge('model_games_count', 'Number of games in the training set')
recommendation_diversity = Gauge('recommendation_diversity', 'Diversity score of recommendations')
recommendation_coverage = Gauge('recommendation_coverage', 'Coverage of catalog in recommendations')

# Métriques de drift
feature_drift_gauge = Gauge('feature_drift_score', 'Feature drift score compared to training data')
prediction_drift_gauge = Gauge('prediction_drift_score', 'Prediction drift score')

# Métriques système
model_memory_usage = Gauge('model_memory_bytes', 'Memory usage of the model in bytes')
model_last_training_timestamp = Gauge('model_last_training_timestamp', 'Timestamp of last model training')


class ModelMonitor:
    """Classe pour monitorer le modèle de recommandation"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.prediction_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.training_data_stats = None
        self.last_evaluation = None
        self.start_time = time.time()
        
    def record_training(self, model_version: str, games_count: int, feature_dim: int, duration: float, metrics: Dict):
        """Enregistre les métriques d'entraînement"""
        model_training_counter.inc()
        training_duration.observe(duration)
        model_games_count.set(games_count)
        model_feature_dimension.set(feature_dim)
        model_last_training_timestamp.set(time.time())
        
        # Mise à jour des infos du modèle
        model_info.info({
            'version': model_version,
            'games_count': str(games_count),
            'feature_dim': str(feature_dim),
            'last_training': datetime.utcnow().isoformat()
        })
        
        # Sauvegarder les stats pour la détection de drift
        self.training_data_stats = metrics.get('validation_metrics', {})
        
        logger.info(f"Training recorded: v{model_version}, {games_count} games, {feature_dim} features, {duration:.2f}s")
    
    def record_prediction(self, endpoint: str, query: str, recommendations: list, latency: float):
        """Enregistre les métriques de prédiction"""
        model_prediction_counter.labels(endpoint=endpoint, status='success').inc()
        prediction_latency.labels(endpoint=endpoint).observe(latency)
        
        # Enregistrer les scores de confiance
        if recommendations:
            confidences = [r.get('confidence', 0) for r in recommendations]
            avg_confidence = np.mean(confidences)
            model_confidence.observe(avg_confidence)
            self.confidence_history.append(avg_confidence)
            
            # Calculer la diversité
            diversity = self._calculate_diversity(recommendations)
            recommendation_diversity.set(diversity)
        
        self.prediction_history.append({
            'timestamp': time.time(),
            'endpoint': endpoint,
            'query': query,
            'num_results': len(recommendations),
            'latency': latency
        })
        
        self.latency_history.append(latency)
        
        # Détecter le drift si on a assez d'historique
        if len(self.confidence_history) >= 100:
            drift_score = self._detect_drift()
            prediction_drift_gauge.set(drift_score)
    
    def record_error(self, endpoint: str, error: str):
        """Enregistre une erreur de prédiction"""
        model_prediction_counter.labels(endpoint=endpoint, status='error').inc()
        logger.error(f"Prediction error on {endpoint}: {error}")
    
    def record_model_load(self, success: bool, model_size_bytes: Optional[int] = None):
        """Enregistre le chargement d'un modèle"""
        status = 'success' if success else 'failure'
        model_load_counter.labels(status=status).inc()
        
        if success and model_size_bytes:
            model_memory_usage.set(model_size_bytes)
    
    def _calculate_diversity(self, recommendations: list) -> float:
        """Calcule la diversité des recommandations (basée sur les genres)"""
        if not recommendations:
            return 0.0
        
        genres = set()
        for rec in recommendations:
            if 'genres' in rec:
                genres.update(g.strip() for g in rec['genres'].split(','))
        
        # Diversité = nombre de genres uniques / nombre de recommandations
        return len(genres) / len(recommendations) if recommendations else 0.0
    
    def _detect_drift(self) -> float:
        """Détecte le drift dans les prédictions"""
        if len(self.confidence_history) < 100:
            return 0.0
        
        # Comparer la distribution récente vs historique
        recent = list(self.confidence_history)[-50:]
        historical = list(self.confidence_history)[-100:-50]
        
        # Test de Kolmogorov-Smirnov simplifié
        recent_mean = np.mean(recent)
        historical_mean = np.mean(historical)
        drift_score = abs(recent_mean - historical_mean)
        
        return min(drift_score, 1.0)
    
    def calculate_coverage(self, total_games: int, recommended_games: set) -> float:
        """Calcule la couverture du catalogue"""
        if total_games == 0:
            return 0.0
        coverage = len(recommended_games) / total_games
        recommendation_coverage.set(coverage)
        return coverage
    
    def get_metrics_summary(self) -> Dict:
        """Retourne un résumé des métriques"""
        uptime = time.time() - self.start_time
        
        summary = {
            "uptime_seconds": uptime,
            "total_predictions": len(self.prediction_history),
            "avg_latency_ms": np.mean(self.latency_history) * 1000 if self.latency_history else 0,
            "p95_latency_ms": np.percentile(self.latency_history, 95) * 1000 if self.latency_history else 0,
            "avg_confidence": np.mean(self.confidence_history) if self.confidence_history else 0,
            "predictions_per_minute": len(self.prediction_history) / (uptime / 60) if uptime > 0 else 0
        }
        
        # Ajouter les métriques récentes (dernière heure)
        one_hour_ago = time.time() - 3600
        recent_predictions = [p for p in self.prediction_history if p['timestamp'] > one_hour_ago]
        
        if recent_predictions:
            summary["recent_predictions_1h"] = len(recent_predictions)
            summary["recent_avg_latency_ms"] = np.mean([p['latency'] for p in recent_predictions]) * 1000
        
        return summary
    
    def evaluate_model(self, model, test_queries: list) -> Dict:
        """Évalue le modèle avec des requêtes de test"""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_queries": len(test_queries),
            "successes": 0,
            "failures": 0,
            "avg_confidence": 0,
            "avg_results": 0
        }
        
        confidences = []
        result_counts = []
        
        for query in test_queries:
            try:
                start = time.time()
                recommendations = model.predict(query, k=10)
                latency = time.time() - start
                
                results["successes"] += 1
                result_counts.append(len(recommendations))
                
                if recommendations:
                    query_confidences = [r.get('confidence', 0) for r in recommendations]
                    confidences.extend(query_confidences)
                    
            except Exception as e:
                results["failures"] += 1
                logger.error(f"Evaluation failed for query '{query}': {e}")
        
        if confidences:
            results["avg_confidence"] = float(np.mean(confidences))
            model_accuracy_gauge.set(results["avg_confidence"])
        
        if result_counts:
            results["avg_results"] = float(np.mean(result_counts))
        
        self.last_evaluation = results
        return results


# Singleton pour le monitor
_monitor_instance: Optional[ModelMonitor] = None

def get_monitor() -> ModelMonitor:
    """Retourne l'instance singleton du monitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ModelMonitor()
    return _monitor_instance


# Décorateur pour mesurer la latence
def measure_latency(endpoint: str):
    """Décorateur pour mesurer la latence d'une fonction"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start
                prediction_latency.labels(endpoint=endpoint).observe(latency)
                return result
            except Exception as e:
                latency = time.time() - start
                model_prediction_counter.labels(endpoint=endpoint, status='error').inc()
                raise e
        return wrapper
    return decorator