#!/usr/bin/env python3
"""
fix_import_error_complete.py - Solution complète pour les erreurs d'import
"""

import os
import sys
import shutil
from pathlib import Path

def fix_model_manager_import():
    """Corrige l'erreur d'import de model_manager"""
    
    print("🔧 Correction de l'erreur d'import model_manager...")
    
    # Vérifier si le fichier existe
    if not Path("model_manager.py").exists():
        print("❌ model_manager.py introuvable")
        return False
    
    # Lire le contenu actuel
    with open("model_manager.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Vérifier la fonction get_model
    if "def get_model()" not in content:
        print("⚠️ Fonction get_model manquante, ajout...")
        
        # Ajouter la fonction get_model à la fin
        get_model_function = '''
# Singleton pour le modèle
_model_instance = None

def get_model():
    """Récupère l'instance du modèle de recommandation"""
    global _model_instance
    if _model_instance is None:
        _model_instance = RecommendationModel()
    return _model_instance

def reset_model():
    """Remet à zéro l'instance du modèle"""
    global _model_instance
    _model_instance = None
'''
        
        content += get_model_function
        
        with open("model_manager.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ Fonction get_model ajoutée")
    
    return True

def ensure_model_class_exists():
    """S'assure que la classe RecommendationModel existe"""
    
    model_file = Path("model_manager.py")
    
    if not model_file.exists():
        print("📁 Création de model_manager.py minimal...")
        
        minimal_model_content = '''# model_manager.py - Modèle de recommandation minimal
from __future__ import annotations

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("model-manager")

class RecommendationModel:
    """Modèle de recommandation basé sur TF-IDF + SVD"""
    
    def __init__(self, model_version: str = "2.0.1"):
        self.model_version = model_version
        self.is_trained = False
        
        # Composants ML
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        self.svd = None
        self.game_features = None
        self.games_df = None
        
        # Métriques
        self.metrics = {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "last_training": None
        }
    
    def prepare_features(self, games: List[Dict]) -> pd.DataFrame:
        """Prépare les features pour l'entraînement"""
        df = pd.DataFrame(games)
        
        # Nettoyage des données
        df["title"] = df["title"].fillna("Unknown")
        df["genres"] = df["genres"].fillna("Unknown")
        df["platforms"] = df["platforms"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x or "")
        )
        
        # Features combinées
        df["combined_features"] = (
            df["title"] + " " + 
            df["genres"] + " " + 
            df["platforms"]
        )
        
        # Features numériques
        df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(4.0)
        df["metacritic"] = pd.to_numeric(df.get("metacritic", 0), errors="coerce").fillna(80)
        
        # Normalisation
        rating_min, rating_max = df["rating"].min(), df["rating"].max()
        if rating_max > rating_min:
            df["rating_norm"] = (df["rating"] - rating_min) / (rating_max - rating_min)
        else:
            df["rating_norm"] = 0.5
        
        metacritic_min, metacritic_max = df["metacritic"].min(), df["metacritic"].max()
        if metacritic_max > metacritic_min:
            df["metacritic_norm"] = (df["metacritic"] - metacritic_min) / (metacritic_max - metacritic_min)
        else:
            df["metacritic_norm"] = 0.5
        
        return df
    
    def train(self, games: List[Dict]) -> Dict:
        """Entraîne le modèle"""
        logger.info(f"Training model v{self.model_version} with {len(games)} games")
        
        if len(games) < 2:
            raise ValueError("Need at least 2 games for training")
        
        try:
            # Préparation des données
            self.games_df = self.prepare_features(games)
            
            # TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(self.games_df["combined_features"])
            
            # SVD adaptatif
            n_features = tfidf_matrix.shape[1]
            n_components = min(20, max(2, n_features - 1))
            
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            features_reduced = self.svd.fit_transform(tfidf_matrix)
            
            # Features finales
            numeric_features = self.games_df[["rating_norm", "metacritic_norm"]].values
            self.game_features = np.hstack([features_reduced, numeric_features])
            
            self.is_trained = True
            self.metrics["last_training"] = datetime.utcnow().isoformat()
            
            logger.info(f"Model trained successfully: {n_components} components, {len(games)} games")
            
            return {
                "status": "success",
                "model_version": self.model_version,
                "games_count": len(games),
                "n_components": n_components,
                "feature_dim": self.game_features.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict(self, query: str, k: int = 10, min_confidence: float = 0.1) -> List[Dict]:
        """Génère des recommandations"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            # Vectoriser la requête
            query_tfidf = self.vectorizer.transform([query])
            query_reduced = self.svd.transform(query_tfidf)
            
            # Features numériques moyennes
            avg_rating = self.games_df["rating_norm"].mean()
            avg_metacritic = self.games_df["metacritic_norm"].mean()
            query_features = np.hstack([query_reduced[0], [avg_rating, avg_metacritic]])
            
            # Calcul similarités
            similarities = cosine_similarity([query_features], self.game_features)[0]
            
            # Sélection des meilleurs résultats
            indices = np.argsort(similarities)[::-1]
            
            recommendations = []
            for idx in indices:
                similarity = similarities[idx]
                if similarity < min_confidence:
                    continue
                if len(recommendations) >= k:
                    break
                
                game_row = self.games_df.iloc[idx]
                recommendations.append({
                    "id": int(game_row["id"]),
                    "title": game_row["title"],
                    "genres": game_row["genres"],
                    "confidence": float(similarity),
                    "rating": float(game_row["rating"]),
                    "metacritic": int(game_row["metacritic"])
                })
            
            # Mise à jour métriques
            self.metrics["total_predictions"] += 1
            if recommendations:
                avg_conf = sum(r["confidence"] for r in recommendations) / len(recommendations)
                self.metrics["avg_confidence"] = (
                    self.metrics["avg_confidence"] * 0.9 + avg_conf * 0.1
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
    
    def save_model(self, filepath: str = "model/recommendation_model.pkl") -> bool:
        """Sauvegarde le modèle"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                "model_version": self.model_version,
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "game_features": self.game_features,
                "games_df": self.games_df,
                "metrics": self.metrics,
                "is_trained": self.is_trained
            }
            
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str = "model/recommendation_model.pkl") -> bool:
        """Charge le modèle"""
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)
            
            self.model_version = model_data.get("model_version", "unknown")
            self.vectorizer = model_data.get("vectorizer")
            self.svd = model_data.get("svd")
            self.game_features = model_data.get("game_features")
            self.games_df = model_data.get("games_df")
            self.metrics = model_data.get("metrics", self.metrics)
            self.is_trained = model_data.get("is_trained", False)
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_metrics(self) -> Dict:
        """Retourne les métriques du modèle"""
        return {
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "total_predictions": self.metrics["total_predictions"],
            "avg_confidence": self.metrics["avg_confidence"],
            "last_training": self.metrics["last_training"],
            "games_count": len(self.games_df) if self.games_df is not None else 0,
            "feature_dim": self.game_features.shape[1] if self.game_features is not None else 0
        }

# Singleton
_model_instance = None

def get_model():
    """Récupère l'instance du modèle de recommandation"""
    global _model_instance
    if _model_instance is None:
        _model_instance = RecommendationModel()
    return _model_instance

def reset_model():
    """Remet à zéro l'instance du modèle"""
    global _model_instance
    _model_instance = None
'''
        
        with open("model_manager.py", "w", encoding="utf-8") as f:
            f.write(minimal_model_content)
        
        print("✅ model_manager.py minimal créé")
        return True
    
    return True

def create_ci_environment_setup():
    """Crée un script de setup pour l'environnement CI"""
    
    setup_script = '''#!/usr/bin/env python3
"""Setup script pour environnement CI"""

import os
import sys
from pathlib import Path

def setup_ci_environment():
    """Configure l'environnement pour les tests CI"""
    
    # Variables d'environnement pour les tests
    os.environ.update({
        'SECRET_KEY': 'test-secret-key-for-ci-very-long-and-secure',
        'ALGORITHM': 'HS256',
        'ACCESS_TOKEN_EXPIRE_MINUTES': '60',
        'ALLOW_ORIGINS': '*',
        'LOG_LEVEL': 'INFO',
        'DB_REQUIRED': 'false',
        'DEMO_LOGIN_ENABLED': 'true',
        'DEMO_USERNAME': 'demo',
        'DEMO_PASSWORD': 'demo123',
        'PYTHONPATH': '.'
    })
    
    # Créer les dossiers nécessaires
    for directory in ['model', 'logs', 'compliance']:
        Path(directory).mkdir(exist_ok=True)
    
    # Ajouter le répertoire courant au path Python
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    print("✅ Environnement CI configuré")

if __name__ == "__main__":
    setup_ci_environment()
'''
    
    with open("setup_ci.py", "w") as f:
        f.write(setup_script)
    
    os.chmod("setup_ci.py", 0o755)
    print("✅ Script setup CI créé")

def fix_api_imports():
    """Corrige les imports dans api_games_plus.py"""
    
    api_file = Path("api_games_plus.py")
    if not api_file.exists():
        print("⚠️ api_games_plus.py non trouvé")
        return False
    
    with open(api_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Vérifier que l'import de model_manager est correct
    if "from model_manager import get_model" not in content:
        print("🔧 Correction de l'import model_manager...")
        
        # Remplacer les imports incorrects
        import re
        
        # Pattern pour trouver les imports model_manager
        old_patterns = [
            r'from model_manager import.*get_model_manager.*',
            r'from model_manager import \(\s*get_model.*?\)',
        ]
        
        for pattern in old_patterns:
            content = re.sub(pattern, 'from model_manager import get_model, reset_model', content, flags=re.DOTALL)
        
        # S'assurer que l'import correct est présent
        if "from model_manager import get_model" not in content:
            # Ajouter l'import après les autres imports locaux
            import_section = "from settings import get_settings"
            if import_section in content:
                content = content.replace(
                    import_section,
                    import_section + "\nfrom model_manager import get_model, reset_model"
                )
        
        with open(api_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ Imports api_games_plus.py corrigés")
    
    return True

def create_test_script():
    """Crée un script de test rapide"""
    
    test_script = '''#!/usr/bin/env python3
"""Test rapide des imports après correction"""

import os
import sys

# Setup environnement
os.environ.update({
    'SECRET_KEY': 'test-secret-key-for-quick-test',
    'DB_REQUIRED': 'false',
    'DEMO_LOGIN_ENABLED': 'true',
    'LOG_LEVEL': 'INFO'
})

# Ajouter le répertoire courant au Python path
sys.path.insert(0, '.')

def test_imports():
    """Test des imports critiques"""
    try:
        print("🧪 Test imports...")
        
        # Test model_manager
        print("1. Test model_manager...")
        from model_manager import get_model, reset_model
        model = get_model()
        print(f"   ✅ Model: {model.model_version}")
        
        # Test settings
        print("2. Test settings...")
        from settings import get_settings
        settings = get_settings()
        print(f"   ✅ Settings: OK")
        
        # Test API
        print("3. Test api_games_plus...")
        import api_games_plus
        app = api_games_plus.app
        print(f"   ✅ API: {app.title}")
        
        # Test health check
        print("4. Test health check...")
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/healthz")
        print(f"   ✅ Health: {response.status_code}")
        
        print("\\n🎉 TOUS LES IMPORTS FONCTIONNENT!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
'''
    
    with open("test_imports_quick.py", "w") as f:
        f.write(test_script)
    
    os.chmod("test_imports_quick.py", 0o755)
    print("✅ Script de test rapide créé")

def update_github_workflow():
    """Met à jour le workflow GitHub pour corriger l'erreur"""
    
    workflow_file = Path(".github/workflows/ci-cd.yml")
    
    if workflow_file.exists():
        with open(workflow_file, "r") as f:
            content = f.read()
        
        # Ajouter le setup CI avant les tests
        if "Run import tests" in content and "python setup_ci.py" not in content:
            # Insérer le setup avant les tests d'import
            content = content.replace(
                "- name: Run import tests",
                '''- name: Setup CI environment
      run: python setup_ci.py
    
    - name: Run import tests'''
            )
            
            with open(workflow_file, "w") as f:
                f.write(content)
            
            print("✅ Workflow GitHub mis à jour")
    
    return True

def main():
    """Fonction principale de correction"""
    
    print("🚨 CORRECTION COMPLÈTE DES ERREURS D'IMPORT")
    print("=" * 50)
    
    corrections = [
        ("1. S'assurer que model_manager.py existe", ensure_model_class_exists),
        ("2. Corriger get_model dans model_manager", fix_model_manager_import),
        ("3. Corriger imports dans api_games_plus", fix_api_imports),
        ("4. Créer script setup CI", create_ci_environment_setup),
        ("5. Créer test rapide", create_test_script),
        ("6. Mettre à jour workflow GitHub", update_github_workflow)
    ]
    
    success_count = 0
    
    for name, func in corrections:
        print(f"\\n📋 {name}...")
        try:
            if func():
                success_count += 1
                print(f"✅ {name} - OK")
        except Exception as e:
            print(f"❌ {name} - Erreur: {e}")
    
    print(f"\\n🎯 RÉSULTATS: {success_count}/{len(corrections)} corrections appliquées")
    
    if success_count >= 4:
        print("\\n✅ CORRECTIONS PRINCIPALES APPLIQUÉES!")
        print("\\n📋 Prochaines étapes:")
        print("1. Tester localement:")
        print("   python test_imports_quick.py")
        print("\\n2. Committer les changements:")
        print("   git add .")
        print("   git commit -m 'fix: Resolve model_manager import errors in CI'")
        print("   git push")
        print("\\n3. Le pipeline CI devrait maintenant passer")
        return True
    else:
        print("\\n⚠️ Certaines corrections ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
