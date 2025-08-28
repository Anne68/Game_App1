from app.ml.model import get_model
from app.ml.tracking import log_inference

def _featurize(title: str, platform: str) -> list[float]:
    # Featurisation toy: longueur du titre + hash simple de la plateforme
    title_len = len(title)
    plat_val = (sum(ord(c) for c in platform) % 100) / 10.0
    return [float(title_len), float(plat_val)]

def predict_one(req: dict) -> dict:
    model = get_model()
    feats = _featurize(req["title"], req["platform"])
    score = model.predict_proba(feats)
    label = "recommended" if score >= 0.5 else "not_recommended"
    log_inference(metrics={"score": float(score)}, params={"len": len(req["title"])}, tags={"platform": req["platform"]})
    return {"label": label, "score": float(score)}
