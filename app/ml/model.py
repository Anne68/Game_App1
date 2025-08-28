# Placeholder de modèle: à remplacer par un vrai modèle (KMeans, keras, etc.)
class DummyModel:
    def predict_proba(self, features: list[float]) -> float:
        # Renvoie une "proba" jouet en fonction d'une simple heuristique
        score = min(0.99, max(0.01, sum(features) / (len(features) * 10.0)))
        return score

_MODEL = DummyModel()

def get_model() -> DummyModel:
    return _MODEL
