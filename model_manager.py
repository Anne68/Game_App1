# model_manager.py - Gestion du modèle de recommandation (Pipeline TF-IDF → SVD → KMeans → KNN)
from __future__ import annotations

import os
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

logger = logging.getLogger("model-manager")

class RecommendationModel:
    """
    Modèle de recommandation avec pipeline :
      TF-IDF (titre+genres+plateformes) -> SVD -> KMeans (8 clusters) -> KNN (NearestNeighbors)
    Méthodes spécialisées :
      - predict(query)
      - predict_by_game_id(game_id)
      - recommend_by_title_similarity(query_title)
      - recommend_by_genre(genre)
      - get_cluster_games(cluster_id)
      - cluster_explore() / random_cluster()
    """

    def __init__(self, model_version: str = "2.0.0"):
        self.model_version = model_version

        # Texte (contenu)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        self.svd = TruncatedSVD(n_components=128, random_state=42)

        # Titre seul (pour similarité stricte par titre)
        self.title_vectorizer = TfidfVectorizer(
            max_features=8000,
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True
        )

        # Clustering et KNN
        self.kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
        self.knn = NearestNeighbors(metric="cosine", n_neighbors=50, algorithm="auto")

        # Données/features
        self.game_features: Optional[np.ndarray] = None          # features réduites + numériques
        self.features_reduced: Optional[np.ndarray] = None       # uniquement SVD
        self.tfidf_matrix: Optional[np.ndarray] = None           # TF-IDF brut (sparse)
        self.cluster_labels: Optional[np.ndarray] = None
        self.games_df: Optional[pd.DataFrame] = None
        self.is_trained: bool = False

        # Caches
        self._id_to_index: Dict[int, int] = {}
        self._neighbors_cache: Dict[int, List[Tuple[int, float]]] = {}

        # Métriques
        self.metrics = {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "last_training": None,
            "model_version": model_version,
            "feature_dim": 0
        }

    # -----------------------------
    # Préparation des features
    # -----------------------------
    def prepare_features(self, games: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(games)

        # Texte combiné pour le contenu
        df["platforms_str"] = df["platforms"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x or ""))
        df["combined_features"] = (df["title"].fillna("") + " " +
                                   df["genres"].fillna("") + " " +
                                   df["platforms_str"].fillna(""))

        # Normalisation numérique robuste (évite division par zéro)
        for col in ["rating", "metacritic"]:
            if col not in df:
                df[col] = 0
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
        df["metacritic"] = pd.to_numeric(df["metacritic"], errors="coerce").fillna(0.0)

        def norm_col(x: pd.Series) -> pd.Series:
            x_min, x_max = float(x.min()), float(x.max())
            if x_max == x_min:
                return pd.Series(np.zeros(len(x)))
            return (x - x_min) / (x_max - x_min)

        df["rating_norm"] = norm_col(df["rating"])
        df["metacritic_norm"] = norm_col(df["metacritic"])

        return df

    # -----------------------------
    # Entraînement
    # -----------------------------
    def train(self, games: List[Dict]) -> Dict:
        logger.info(f"Training model v{self.model_version} with {len(games)} games")

        self.games_df = self.prepare_features(games)

        # TF-IDF contenu -> SVD
        self.tfidf_matrix = self.vectorizer.fit_transform(self.games_df["combined_features"])
        self.features_reduced = self.svd.fit_transform(self.tfidf_matrix)

        # Concat avec numériques
        numeric = self.games_df[["rating_norm", "metacritic_norm"]].to_numpy(dtype=float)
        self.game_features = np.hstack([self.features_reduced, numeric])

        # KMeans + KNN
        self.cluster_labels = self.kmeans.fit_predict(self.features_reduced)
        self.knn.fit(self.game_features)

        # Vectoriseur de titres pour similarité stricte titres
        self.title_tfidf = self.title_vectorizer.fit_transform(self.games_df["title"].fillna(""))

        # Index rapides
        self._id_to_index = {int(self.games_df.iloc[idx]["id"]): idx for idx in range(len(self.games_df))}
        self._neighbors_cache.clear()

        self.is_trained = True
        self.metrics["last_training"] = datetime.utcnow().isoformat()
        self.metrics["feature_dim"] = int(self.game_features.shape[1])

        validation = self._validate_model()
        logger.info(f"Model trained. Features: {self.game_features.shape}, clusters: {len(np.unique(self.cluster_labels))}")
        return {
            "status": "success",
            "model_version": self.model_version,
            "games_count": len(games),
            "feature_dim": int(self.game_features.shape[1]),
            "validation_metrics": validation
        }

    def _validate_model(self) -> Dict:
        if not self.is_trained or self.game_features is None:
            return {"error": "Model not trained"}

        # Mesure de compacité inter/intra cluster simple
        try:
            centers = self.kmeans.cluster_centers_  # espace réduit
            # Distance moyenne jeu->centre de son cluster
            dists = []
            for i, z in enumerate(self.features_reduced):
                c = centers[self.cluster_labels[i]]
                dists.append(np.linalg.norm(z - c))
            return {
                "mean_distance_to_center": float(np.mean(dists)),
                "std_distance_to_center": float(np.std(dists)),
                "max_distance_to_center": float(np.max(dists))
            }
        except Exception:
            return {"note": "validation skipped"}

    # -----------------------------
    # Prédictions & recommandations
    # -----------------------------
    def _neighbors_for_index(self, idx: int, k: int) -> List[Tuple[int, float]]:
        if idx in self._neighbors_cache and len(self._neighbors_cache[idx]) >= k:
            return self._neighbors_cache[idx][:k]
        distances, indices = self.knn.kneighbors([self.game_features[idx]], n_neighbors=min(k + 1, len(self.game_features)))
        # Exclure soi-même (distance zéro)
        pairs = [(int(j), float(1.0 - distances[0][t])) for t, j in enumerate(indices[0]) if int(j) != idx]
        self._neighbors_cache[idx] = pairs
        return pairs[:k]

    def predict(self, query: str, k: int = 10, min_confidence: float = 0.1) -> List[Dict]:
        """Recommandations par requête libre (contenu)."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Projeter la requête dans l'espace
        q_tfidf = self.vectorizer.transform([query])
        q_red = self.svd.transform(q_tfidf)
        avg_num = self.games_df[["rating_norm", "metacritic_norm"]].to_numpy(dtype=float).mean(axis=0)
        q_feat = np.hstack([q_red[0], avg_num])

        # KNN global
        distances, indices = self.knn.kneighbors([q_feat], n_neighbors=min(k * 3, len(self.game_features)))
        recs = []
        for t, j in enumerate(indices[0]):
            score = 1.0 - float(distances[0][t])  # convertir distance cos en similarité
            if score < min_confidence:
                continue
            row = self.games_df.iloc[int(j)]
            recs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "genres": row["genres"],
                "confidence": score,
                "rating": float(row["rating"]),
                "metacritic": int(row["metacritic"]),
                "cluster": int(self.cluster_labels[int(j)])
            })
            if len(recs) >= k:
                break

        # métriques
        self._update_metrics(recs, k)
        return recs

    def predict_by_game_id(self, game_id: int, k: int = 10) -> List[Dict]:
        """Recommandations type 'jeux similaires' à un jeu existant (KNN)."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        if game_id not in self._id_to_index:
            raise ValueError(f"Game with id {game_id} not found")

        idx = self._id_to_index[game_id]
        pairs = self._neighbors_for_index(idx, k)
        recs = []
        for j, score in pairs:
            row = self.games_df.iloc[j]
            recs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "genres": row["genres"],
                "confidence": score,
                "rating": float(row["rating"]),
                "metacritic": int(row["metacritic"]),
                "cluster": int(self.cluster_labels[j])
            })
        self._update_metrics(recs, k)
        return recs

    def recommend_by_title_similarity(self, query_title: str, k: int = 10, min_confidence: float = 0.0) -> List[Dict]:
        """Similarité stricte sur les titres (TF-IDF caractères) — utile pour typo/fuzzy."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        q = self.title_vectorizer.transform([query_title])
        sims = cosine_similarity(q, self.title_tfidf)[0]
        order = np.argsort(sims)[::-1]
        recs = []
        for idx in order[:k * 2]:
            score = float(sims[idx])
            if score < min_confidence:
                continue
            row = self.games_df.iloc[int(idx)]
            recs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "genres": row["genres"],
                "confidence": score,
                "rating": float(row["rating"]),
                "metacritic": int(row["metacritic"]),
                "cluster": int(self.cluster_labels[int(idx)])
            })
            if len(recs) >= k:
                break
        self._update_metrics(recs, k)
        return recs

    def recommend_by_genre(self, genre: str, k: int = 10) -> List[Dict]:
        """Filtrage simple par genre: prend les plus proches d'une 'requête-genre'."""
        return self.predict(query=genre, k=k, min_confidence=0.0)

    # -----------------------------
    # Clusters
    # -----------------------------
    def get_cluster_games(self, cluster_id: int, sample: Optional[int] = None) -> List[Dict]:
        if not self.is_trained:
            raise ValueError("Model not trained")
        if cluster_id < 0 or cluster_id >= self.kmeans.n_clusters:
            raise ValueError(f"Cluster id must be in [0, {self.kmeans.n_clusters-1}]")
        idxs = np.where(self.cluster_labels == cluster_id)[0]
        if sample is not None and sample > 0:
            if sample < len(idxs):
                rng = np.random.default_rng(42)
                idxs = rng.choice(idxs, size=sample, replace=False)
        out = []
        for i in idxs:
            row = self.games_df.iloc[int(i)]
            out.append({
                "id": int(row["id"]),
                "title": row["title"],
                "genres": row["genres"],
                "rating": float(row["rating"]),
                "metacritic": int(row["metacritic"])
            })
        return out

    def _top_terms_per_cluster(self, top_n: int = 10) -> List[Dict]:
        """
        Approximation: moyenne TF-IDF par cluster, top features textuelles.
        """
        if self.tfidf_matrix is None:
            return []
        terms = np.array(self.vectorizer.get_feature_names_out())
        result = []
        for c in range(self.kmeans.n_clusters):
            idxs = np.where(self.cluster_labels == c)[0]
            if len(idxs) == 0:
                result.append({"cluster": c, "size": 0, "top_terms": []})
                continue
            # moyenne TF-IDF cluster (sparse -> dense mean via .mean(axis=0))
            mean_vec = self.tfidf_matrix[idxs].mean(axis=0).A1
            top_idx = np.argsort(mean_vec)[::-1][:top_n]
            result.append({
                "cluster": c,
                "size": int(len(idxs)),
                "top_terms": terms[top_idx].tolist()
            })
        return result

    def cluster_explore(self, top_n_terms: int = 10) -> Dict:
        if not self.is_trained:
            raise ValueError("Model not trained")
        sizes = np.bincount(self.cluster_labels, minlength=self.kmeans.n_clusters).tolist()
        return {
            "n_clusters": int(self.kmeans.n_clusters),
            "sizes": sizes,
            "top_terms": self._top_terms_per_cluster(top_n_terms)
        }

    def random_cluster(self, sample: int = 12) -> Dict:
        if not self.is_trained:
            raise ValueError("Model not trained")
        c = int(np.random.default_rng().integers(0, self.kmeans.n_clusters))
        games = self.get_cluster_games(c, sample=sample)
        return {"cluster": c, "sample": games}

    # -----------------------------
    # Persistance & métriques
    # -----------------------------
    def save_model(self, filepath: str = "model/recommendation_model.pkl") -> bool:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                "version": self.model_version,
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "kmeans": self.kmeans,
                "knn": self.knn,
                "title_vectorizer": self.title_vectorizer,
                "title_tfidf": getattr(self, "title_tfidf", None),
                "tfidf_matrix": self.tfidf_matrix,
                "features_reduced": self.features_reduced,
                "game_features": self.game_features,
                "cluster_labels": self.cluster_labels,
                "games_df": self.games_df,
                "metrics": self.metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filepath: str = "model/recommendation_model.pkl") -> bool:
        try:
            with open(filepath, "rb") as f:
                m = pickle.load(f)

            self.model_version = m["version"]
            self.vectorizer = m["vectorizer"]
            self.svd = m["svd"]
            self.kmeans = m["kmeans"]
            self.knn = m["knn"]
            self.title_vectorizer = m["title_vectorizer"]
            self.title_tfidf = m["title_tfidf"]
            self.tfidf_matrix = m["tfidf_matrix"]
            self.features_reduced = m["features_reduced"]
            self.game_features = m["game_features"]
            self.cluster_labels = m["cluster_labels"]
            self.games_df = m["games_df"]
            self.metrics = m["metrics"]
            self._id_to_index = {int(row.id): idx for idx, row in self.games_df[["id"]].itertuples(index=True)}
            self._neighbors_cache.clear()
            self.is_trained = True
            logger.info(f"Model v{self.model_version} loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _update_metrics(self, recs: List[Dict], k: int):
        self.metrics["total_predictions"] += 1
        if recs:
            avg_conf = float(np.mean([r["confidence"] for r in recs[:k]]))
            n = self.metrics["total_predictions"]
            self.metrics["avg_confidence"] = (self.metrics["avg_confidence"] * (n - 1) + avg_conf) / n

    def get_metrics(self) -> Dict:
        return {
            **self.metrics,
            "is_trained": self.is_trained,
            "games_count": len(self.games_df) if self.games_df is not None else 0
        }


# Singleton
_model_instance: Optional[RecommendationModel] = None

def get_model() -> RecommendationModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = RecommendationModel()
        if os.path.exists("model/recommendation_model.pkl"):
            _model_instance.load_model()
    return _model_instance
