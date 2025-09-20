import json
import requests
import streamlit as st
from typing import Any, Dict, Optional

# ============================
#  Games API — UI épurée
#  Garde uniquement :
#   - Recommandations (ML)
#   - Par titre
#   - Par genre
#   - Par plateforme (filtre simple)
# ============================

st.set_page_config(page_title="Games Reco (UI light)", page_icon="🎮", layout="centered")

DEFAULT_API_BASE = "https://game-app-y8be.onrender.com"
DEFAULT_TIMEOUT = 30

# ----------------
# Session state
# ----------------
if "api_base" not in st.session_state:
    st.session_state.api_base = DEFAULT_API_BASE
if "timeout" not in st.session_state:
    st.session_state.timeout = DEFAULT_TIMEOUT
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

# ----------------
# Helpers
# ----------------

def api_url(path: str) -> str:
    base = st.session_state.api_base.rstrip("/")
    return base + path


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}


def post_form_token(username: str, password: str) -> Dict[str, Any]:
    try:
        r = requests.post(api_url("/token"), data={"username": username, "password": password}, timeout=st.session_state.timeout)
        return r.json() if r.ok else {"error": f"{r.status_code}", "detail": r.text}
    except requests.RequestException as e:
        return {"error": str(e)}


def get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        r = requests.get(api_url(path), params=params, headers=_auth_headers(), timeout=st.session_state.timeout)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
        return data if r.ok else {"error": f"{r.status_code}", "detail": data}
    except requests.RequestException as e:
        return {"error": str(e)}


def post_json(path: str, payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        r = requests.post(api_url(path), json=payload or {}, params=params, headers=_auth_headers(), timeout=st.session_state.timeout)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
        return data if r.ok else {"error": f"{r.status_code}", "detail": data}
    except requests.RequestException as e:
        return {"error": str(e)}

# ----------------
# Sidebar (minimal)
# ----------------
with st.sidebar:
    st.markdown("### ⚙️ Réglages")
    st.session_state.api_base = st.text_input("API base URL", value=st.session_state.api_base)
    st.session_state.timeout = st.slider("Timeout (s)", min_value=5, max_value=120, value=st.session_state.timeout)
    if st.button("Tester /healthz", use_container_width=True):
        res = get_json("/healthz")
        if "error" in res:
            st.error("❌ API indisponible")
            st.caption(str(res))
        else:
            st.success("✅ API OK")

# ----------------
# Header + Auth compact
# ----------------
st.title("🎮 Games Reco")
st.caption("Interface minimaliste pour l'API ML")

if not st.session_state.token:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        user = st.text_input("Utilisateur", placeholder="demo…")
    with c2:
        pwd = st.text_input("Mot de passe", type="password", placeholder="••••••••")
    with c3:
        if st.button("Se connecter", use_container_width=True):
            out = post_form_token(user, pwd)
            if out.get("access_token"):
                st.session_state.token = out["access_token"]
                st.session_state.username = user
                st.success("Connecté")
            else:
                st.error(out)
else:
    st.success(f"Connecté : {st.session_state.username}")
    if st.button("Se déconnecter"):
        st.session_state.token = None
        st.session_state.username = None
        st.experimental_rerun()

# Stop here if not authenticated
if not st.session_state.token:
    st.info("Connectez‑vous pour utiliser les recommandations.")
    st.stop()

# ----------------
# Onglets essentiels
# ----------------
TABS = st.tabs(["🔎 Reco ML", "🔤 Par titre", "🏷️ Par genre", "🎮 Par plateforme"])

# --- 1) Reco ML ---
with TABS[0]:
    st.subheader("Recommandations (ML)")
    q = st.text_input("Requête", value="RPG Action", label_visibility="collapsed")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        k = st.number_input("k", 1, 50, 10)
    with col2:
        min_conf = st.slider("Confiance min.", 0.0, 1.0, 0.3, 0.05)
    with col3:
        run = st.button("Obtenir des recos", use_container_width=True)
    if run:
        res = post_json("/recommend/ml", {"query": q.strip(), "k": int(k), "min_confidence": float(min_conf)})
        if "error" in res:
            st.error(res)
        else:
            st.caption(f"Modèle: {res.get('model_version','?')} — Latence: {res.get('latency_ms',0):.0f} ms")
            st.dataframe(res.get("recommendations", []), use_container_width=True)

# --- 2) Par titre ---
with TABS[1]:
    st.subheader("Recommandations par similarité de titre")
    title = st.text_input("Titre", value="Hades")
    k2 = st.number_input("k", 1, 50, 10, key="k_title")
    if st.button("Recommander", key="btn_title"):
        res = get_json(f"/recommend/by-title/{title}", params={"k": int(k2)})
        st.error(res) if "error" in res else st.dataframe(res.get("recommendations", []), use_container_width=True)

# --- 3) Par genre ---
with TABS[2]:
    st.subheader("Recommandations par genre")
    genre = st.text_input("Genre", value="Action")
    k3 = st.number_input("k", 1, 50, 10, key="k_genre")
    if st.button("Recommander", key="btn_genre"):
        res = get_json(f"/recommend/by-genre/{genre}", params={"k": int(k3)})
        st.error(res) if "error" in res else st.dataframe(res.get("recommendations", []), use_container_width=True)

# --- 4) Par plateforme ---
with TABS[3]:
    st.subheader("Recommandations par plateforme")
    platform = st.selectbox("Plateforme", ["PC", "PS4", "PS5", "Xbox", "Switch", "Steam", "GOG", "Epic"], index=0)
    hint = st.text_input("(Optionnel) Ajouter un mot-clé", placeholder="ex: roguelike, rpg, indie…")
    kk = st.number_input("k", 1, 50, 10, key="k_platform")
    if st.button("Recommander", key="btn_platform"):
        # Pas d'endpoint dédié côté API → on utilise /recommend/ml avec la plateforme comme requête
        query = f"{platform} {hint}".strip()
        res = post_json("/recommend/ml", {"query": query, "k": int(kk), "min_confidence": 0.1})
        if "error" in res:
            st.error(res)
        else:
            # Filtre léger côté client si le champ 'platforms' est présent dans la réponse
            recs = res.get("recommendations", [])
            filtered = []
            for r in recs:
                plats = str(r.get("platforms", ""))
                if platform.lower() in plats.lower():
                    filtered.append(r)
            st.caption(f"Modèle: {res.get('model_version','?')} — Latence: {res.get('latency_ms',0):.0f} ms")
            st.dataframe(filtered or recs, use_container_width=True)
