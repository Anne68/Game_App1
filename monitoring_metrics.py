# monitoring_metrics.py - Métriques & monitoring pour l'API ML
from __future__ import annotations

import asyncio
import time
import logging
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Info

logger = logging.getLogger("monitoring")

# ========= Prometheus metrics =========
# Compteurs
model_training_counter = Counter(
    "model_training_total", "Number of model trainings executed"
)
model_prediction_counter = Counter(
    "model_predictions_total",
    "Number of predictions",
    labelnames=("endpoint", "status"),
)
model_error_counter = Counter(
    "model_errors_total", "Number of model errors", labelnames=("endpoint",)
)

# Latence (par endpoint)
prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Latency of recommendation endpoints (seconds)",
    labelnames=("endpoint",),
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# Info & gauges
model_version_info = Info("model_version", "Current model version")
model_accuracy_gauge = Gauge("model_accuracy", "Current model accuracy (eval)")
model_feature_dimension = Gauge("model_feature_dimension", "Number of features in the model")
model_games_count = Gauge("model_games_count", "Number of games used for training")
recommendation_diversity = Gauge("recommendation_diversity", "Diversity score of recommendations")
recommendation_coverage = Gauge("recommendation_coverage", "Coverage of catalog in recommendations")
feature_drift_gauge = Gauge("feature_drift_score", "Feature drift score compared to training data")
prediction_drift_gauge = Gauge("prediction_drift_score", "Prediction drift score based on confidence history")
model_memory_usage = Gauge("model_memory_bytes", "Approx model memory usage (bytes)")
model_last_training_time = Gauge("model_last_training_timestamp", "Last training timestamp (unix seconds)")

# ========= Decorator (fix 422) =========
from functools import wraps

def measure_latency(endpoint: str):
    """Mesure la latence en préservant la signature (compat FastAPI)."""
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    res = await func(*args, **kwargs)
                    prediction_latency.labels(endpoint=endpoint).observe(time.time() - start)
                    return res
                except Exception:
                    prediction_latency.labels(endpoint=endpoint).observe(time.time() - start)
                    model_prediction_counter.labels(endpoint=endpoint, status="error").inc()
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    res = func(*args, **kwargs)
                    prediction_latency.labels(endpoint=endpoint).observe(time.time() - start)
                    return res
                except Exception:
                    prediction_latency.labels(endpoint=endpoint).observe(time.time() - start)
                    model_prediction_counter.labels(endpoint=endpoint, status="error").inc()
                    raise
            return sync_wrapper
    return decorator

# ========= Monitor singleton =========
class _Monitor:
    def __init__(self) -> None:
        self.confidence_history: deque[float] = deque(maxlen=1000)
        self.last_evaluation: Optional[Dict[str, Any]] = None
        self.total_predictions: int = 0
        self.avg_confidence: float = 0.0
        self.model_version: str = "unknown"
        self.last_training_iso: Optional[str] = None

    def record_training(
        self,
        model_version: str,
        games_count: int,
        feature_dim: int,
        duration: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_training_counter.inc()
        model_version_info.info({"version": model_version})
        self.model_version = model_version
        self.last_training_iso = datetime.utcnow().isoformat()

        model_games_count.set(games_count)
        model_feature_dimension.set(feature_dim)
        model_last_training_time.set(time.time())

        if metrics:
            acc = float(metrics.get("accuracy", 0.0))
            model_accuracy_gauge.set(acc)

        mem = float(metrics.get("memory_bytes", 0.0)) if metrics else 0.0
        model_memory_usage.set(mem)

        logger.info(
            "Training recorded: version=%s games=%s feat=%s duration=%.3fs",
            model_version, games_count, feature_dim, duration
        )

    def record_prediction(
        self,
        endpoint: str,
        query: str,
        recommendations: List[Dict[str, Any]],
        latency: float,
    ) -> None:
        try:
            confs: List[float] = []
            for r in recommendations or []:
                if "confidence" in r:
                    confs.append(float(r["confidence"]))
                elif "score" in r:
                    confs.append(float(r["score"]))
            mean_conf = float(np.mean(confs)) if confs else 0.0
            self.confidence_history.append(mean_conf)
            self.total_predictions += 1
            self.avg_confidence = (0.95 * self.avg_confidence + 0.05 * mean_conf) if self.total_predictions > 1 else mean_conf

            ids = [r.get("id") for r in (recommendations or []) if r.get("id") is not None]
            unique_ids = len(set(ids))
            diversity = (unique_ids / max(1, len(ids))) if ids else 0.0
            recommendation_diversity.set(diversity)
            recommendation_coverage.set(diversity)

            model_prediction_counter.labels(endpoint=endpoint, status="ok").inc()
            prediction_latency.labels(endpoint=endpoint).observe(latency)
        except Exception as e:
            logger.exception("record_prediction failed: %s", e)
            model_prediction_counter.labels(endpoint=endpoint, status="error").inc()

    def record_error(self, endpoint: str, message: str) -> None:
        model_error_counter.labels(endpoint=endpoint).inc()
        logger.error("Error on %s: %s", endpoint, message)

    def evaluate_model(self, model, test_queries: List[str]) -> Dict[str, Any]:
        scores: List[float] = []
        ids_set: set = set()
        for q in test_queries:
            try:
                recs = model.predict(q, k=10, min_confidence=0.0)
                for r in recs:
                    if "confidence" in r:
                        scores.append(float(r["confidence"]))
                    elif "score" in r:
                        scores.append(float(r["score"]))
                    if r.get("id") is not None:
                        ids_set.add(r["id"])
            except Exception as e:
                self.record_error("evaluate_model", str(e))

        avg_score = float(np.mean(scores)) if scores else 0.0
        model_accuracy_gauge.set(avg_score)

        coverage = 0.0
        try:
            if getattr(model, "games_df", None) is not None and not model.games_df.empty:
                catalog = int(model.games_df["id"].nunique())
                coverage = len(ids_set) / max(1, catalog)
        except Exception:
            pass
        recommendation_coverage.set(coverage)

        self.last_evaluation = {
            "time": datetime.utcnow().isoformat(),
            "avg_confidence": avg_score,
            "unique_ids": len(ids_set),
            "coverage": coverage,
            "tested_queries": test_queries,
        }
        prediction_drift_gauge.set(self._detect_drift())
        return self.last_evaluation

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "is_trained": True,
            "total_predictions": self.total_predictions,
            "avg_confidence": round(self.avg_confidence, 6),
            "last_training": self.last_training_iso,
            "confidence_hist_len": len(self.confidence_history),
        }

    def _detect_drift(self) -> float:
        """Score [0..1] basé sur la variance récente des confiances."""
        if len(self.confidence_history) < 50:
            return 0.0
        arr = np.array(self.confidence_history, dtype=float)
        recent = arr[-100:] if len(arr) >= 100 else arr
        var = float(np.var(recent))
        score = var / (1.0 + var)
        return float(min(max(score, 0.0), 1.0))

# Singleton
_MONITOR: Optional[_Monitor] = None

def get_monitor() -> _Monitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = _Monitor()
    return _MONITOR
