# app_streamlit.py
# Streamlit front-end pour api_games_plus.py — thème "arcade neon"
# Lance: streamlit run app_streamlit.py

import os
import time
import json
import requests
import streamlit as st
from typing import Optional, Dict, Any, List

# =========================
# --------- THEME ---------
# =========================
ARCADE_CSS = """
<style>
/* Fond dégradé néon */
.stApp {
  background: radial-gradient(1200px 500px at 10% 0%, #081226 0%, #0b0f1d 35%, #070b14 60%, #05070d 100%) !important;
  color: #e6f3ff;
  font-family: ui-rounded, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif;
}

/* Titres néon */
h1, h2, h3, h4 {
  color: #9be8ff !important;
  text-shadow: 0 0 12px rgba(155,232,255,.5), 0 0 24px rgba(72,163,255,.25);
}

/* Cartes vitrées */
.block-container { padding-top: 1.5rem; }
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#09132a,#0b1733) !important;
  border-right: 1px solid rgba(255,255,255,.08);
}
.stTabs [data-baseweb="tab"] {
  color: #cbe7ff !important;
  font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
  background: linear-gradient(90deg, #00eaff33, #9d00ff33);
}

/* Boutons arcade */
.stButton>button {
  background: linear-gradient(90deg, #00eaff, #9d00ff) !important;
  color: #081226 !important;
  border: 0 !important;
  border-radius: 14px !important;
  box-shadow: 0 8px 24px rgba(0,234,255,.25);
  font-weight: 800 !important;
}
.stButton>button:hover {
  transform: translateY(-1px) scale(1.01);
}

/* Inputs */
input, textarea, .stTextInput, .stTextArea, .stNumberInput {
  color: #e6f3ff !important;
}
.css-1cpxqw2, .css-1x8cf1d, .stSelectbox, .stSlider {
  color: #e6f3ff !important;
}

/* Badges / chips */
.badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: linear-gradient(90deg, #00ffa3, #00d4ff);
  color: #081226;
  font-weight: 800;
  font-size: 12px;
}
.metric-pill {
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 8px 12px;
  border-radius: 14px;
  background: #0c1632;
  border: 1px solid rgba(255,255,255,.08);
  margin: 4px 6px 0 0;
}
.kbd {
  background:#0d1a39; border:1px solid #1e2a4d; border-bottom-width:3px; padding:2px 6px; border-radius:6px;
  font-weight:700; color:#bde0ff;
}
.small-muted { color:#9fb6d1; font-size:12px; }
.card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 16px;
  padding: 14px;
}
hr { border-color: rgba(255,255,255,.1); }
</style>
"""

# =========================
# ---- CONFIG & STATE ----
# =========================
DEFAULT_API_URL = os.getenv("GAMES_API_URL", "http://localhost:8000")
st.set_page_config(page_title="Games Reco — Arcade", page_icon="🎮", layout="wide")
st.markdown(ARCADE_CSS, unsafe_allow_html=True)

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "model_version" not in st.session_state:
    st.session_state.model_version = None

# =========================
# ------- HELPERS --------
# =========================
def api_base() -> str:
    return st.session_state.api_url.rstrip("/")

def bearer_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if st.session_state.token:
        h["Authorization"] = f"Bearer {st.session_state.token}"
    return h

def post(path: str, payload: Dict[str, Any], timeout: int = 25) -> requests.Response:
    url = f"{api_base()}{path}"
    return requests.post(url, headers=bearer_headers(), data=json.dumps(payload), timeout=timeout)

def get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 25) -> requests.Response:
    url = f"{api_base()}{path}"
    return requests.get(url, headers=bearer_headers(), params=params, timeout=timeout)

def login(username: str, password: str) -> bool:
    try:
        r = requests.post(f"{api_base()}/token",
                          data={"username": username, "password": password},
                          timeout=20)
        if r.status_code == 200:
            data = r.json()
            st.session_state.token = data.get("access_token")
            st.session_state.username = username
            return True
        else:
            st.error(f"⛔ Login échoué: {r.status_code} — {r.text}")
            return False
    except Exception as e:
        st.error(f"⚠️ Erreur réseau login: {e}")
        return False

def register(username: str, password: str) -> bool:
    try:
        r = requests.post(f"{api_base()}/register",
                          data={"username": username, "password": password},
                          timeout=20)
        if r.status_code == 200:
            st.success("✅ Compte créé ! Connecte-toi maintenant.")
            return True
        else:
            st.error(f"⛔ Inscription refusée: {r.status_code} — {r.text}")
            return False
    except Exception as e:
        st.error(f"⚠️ Erreur réseau register: {e}")
        return False

def guard_auth():
    if not st.session_state.token:
        st.warning("🔐 Tu dois être connecté pour utiliser l’API.")
        st.stop()

def show_metrics_pills(metrics: Dict[str, Any]):
    cols = st.columns(4)
    cols[0].markdown(f'<div class="metric-pill">🧠 <b>Version</b> {metrics.get("model_version","?")}</div>', unsafe_allow_html=True)
    cols[1].markdown(f'<div class="metric-pill">✅ <b>Trained</b> {metrics.get("is_trained")}</div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="metric-pill">📈 <b>Preds</b> {metrics.get("total_predictions",0)}</div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="metric-pill">⭐ <b>Avg conf</b> {round(metrics.get("avg_confidence",0.0),3)}</div>', unsafe_allow_html=True)
    st.caption(f"🕒 Dernier entraînement: {metrics.get('last_training')}")
    st.caption(f"🎮 Jeux en base: {metrics.get('games_count')} | 🔢 Dim features: {metrics.get('feature_dimension')}")

# =========================
# ------- SIDEBAR --------
# =========================
with st.sidebar:
    st.markdown("## 🎮 Games API — Arcade")
    st.text_input("Base API URL", key="api_url", help="Ex: http://localhost:8000 ou https://ton-domaine")
    st.divider()

    if st.session_state.token:
        st.markdown(f"**Connecté:** `{st.session_state.username}`")
        if st.button("🚪 Se déconnecter"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
    else:
        st.markdown("### 🔐 Authentification")
        auth_tab = st.tabs(["Se connecter", "Créer un compte"])
        with auth_tab[0]:
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            demo_hint = st.toggle("Mode démo ?", value=False,
                                  help="Si la BDD est indisponible et l’API est en DEMO_LOGIN_ENABLED, utilise les identifiants démo.")
            if st.button("🎯 Login"):
                if login(u, p):
                    st.success("✅ Connecté !")
                    st.rerun()
        with auth_tab[1]:
            u2 = st.text_input("Username (new)", key="reg_user")
            p2 = st.text_input("Password (new)", type="password", key="reg_pass")
            if st.button("🆕 Register"):
                register(u2, p2)

    st.divider()
    if st.session_state.token:
        try:
            resp = requests.get(f"{api_base()}/healthz", timeout=10)
            if resp.ok:
                hz = resp.json()
                st.markdown("### 🩺 Healthcheck")
                ok = hz.get("status","?")
                db = hz.get("db_ready")
                mv = hz.get("model_version")
                st.write(f"Status: **{ok}**")
                st.write(f"DB ready: **{db}**")
                st.write(f"Model: **{mv}**")
        except Exception:
            st.warning("Healthcheck indisponible.")

# =========================
# --------- TABS ----------
# =========================
st.title("🕹️ Games Recommender — **Arcade Neon**")

tabs = st.tabs([
    "🔎 Recommandations",
    "🧩 Clusters",
    "📊 Modèle & Monitoring",
    "🛠️ Admin rapide",
])

# =========================
# ---- TAB 1: RECOs ------
# =========================
with tabs[0]:
    guard_auth()
    st.subheader("🔎 Reco ML (texte libre)")
    with st.form("form_reco_ml", clear_on_submit=False):
        query = st.text_input("Décris ton envie (ex: 'RPG open world dark fantasy')", value="")
        k = st.slider("Nombre de jeux", 1, 30, 10)
        min_conf = st.slider("Confiance minimum", 0.0, 1.0, 0.10, 0.01)
        submitted = st.form_submit_button("⚡ Recommander")
    if submitted:
        with st.spinner("Calcul des recos..."):
            r = post("/recommend/ml", {"query": query, "k": k, "min_confidence": min_conf})
            if r.ok:
                data = r.json()
                st.session_state.model_version = data.get("model_version")
                st.success(f"✅ {len(data.get('recommendations',[]))} résultats — {round(data.get('latency_ms',0),1)} ms")
                for idx, g in enumerate(data.get("recommendations", []), start=1):
                    with st.container(border=True):
                        st.markdown(f"### {idx}. {g.get('title','?')}")
                        cols = st.columns(4)
                        cols[0].markdown(f"<span class='badge'>score {round(g.get('score',0.0),3)}</span>", unsafe_allow_html=True)
                        if g.get("confidence") is not None:
                            cols[1].markdown(f"<div class='metric-pill'>⭐ conf {round(g['confidence'],3)}</div>", unsafe_allow_html=True)
                        if g.get("genre"):
                            cols[2].markdown(f"<div class='metric-pill'>🏷️ {g['genre']}</div>", unsafe_allow_html=True)
                        if g.get("platforms"):
                            plat = ", ".join(g["platforms"]) if isinstance(g["platforms"], list) else g["platforms"]
                            cols[3].markdown(f"<div class='metric-pill'>🖥️ {plat}</div>", unsafe_allow_html=True)
                        st.caption(g.get("explanation") or "")
            else:
                st.error(f"⛔ {r.status_code} — {r.text}")

    st.divider()
    st.subheader("🎯 Similar Game")
    c1, c2 = st.columns(2)
    with c1:
        game_id = st.number_input("game_id (si connu)", min_value=0, value=0, step=1)
    with c2:
        title_like = st.text_input("...ou titre proche")
    k2 = st.slider("Nombre de jeux similaires", 1, 30, 10, key="k_sim")
    if st.button("🧭 Trouver similaires"):
        payload = {"game_id": int(game_id) if game_id else None, "title": title_like or None, "k": k2}
        r = post("/recommend/similar-game", payload)
        if r.ok:
            d = r.json()
            st.success(f"OK — modèle {d.get('model_version')}")
            for i, g in enumerate(d.get("recommendations", []), start=1):
                st.markdown(f"- **{i}. {g.get('title','?')}** — score: `{round(g.get('score',0.0),3)}`")
        else:
            st.error(f"⛔ {r.status_code} — {r.text}")

    st.divider()
    st.subheader("🧠 Par Titre / Par Genre")
    c3, c4 = st.columns(2)
    with c3:
        t = st.text_input("Titre", key="title_search")
        kk = st.slider("Top K (titre)", 1, 50, 10, key="k_title")
        if st.button("🔤 Reco par titre"):
            r = get(f"/recommend/by-title/{t}", {"k": kk})
            if r.ok:
                recs = r.json().get("recommendations", [])
                for i, g in enumerate(recs, start=1):
                    st.markdown(f"- **{i}. {g.get('title','?')}** — score: `{round(g.get('score',0.0),3)}`")
            else:
                st.error(f"{r.status_code} — {r.text}")
    with c4:
        gname = st.text_input("Genre (ex: RPG, Action, Indie)")
        kkk = st.slider("Top K (genre)", 1, 50, 10, key="k_genre")
        if st.button("🏷️ Reco par genre"):
            r = get(f"/recommend/by-genre/{gname}", {"k": kkk})
            if r.ok:
                recs = r.json().get("recommendations", [])
                for i, g in enumerate(recs, start=1):
                    st.markdown(f"- **{i}. {g.get('title','?')}** — score: `{round(g.get('score',0.0),3)}`")
            else:
                st.error(f"{r.status_code} — {r.text}")

# =========================
# ---- TAB 2: CLUSTERS ----
# =========================
with tabs[1]:
    guard_auth()
    st.subheader("🧩 Explorer les clusters")
    c1, c2, c3 = st.columns(3)
    with c1:
        cluster_id = st.number_input("Cluster ID", min_value=0, value=0, step=1)
    with c2:
        sample = st.slider("Taille d’échantillon", 1, 200, 50)
    with c3:
        top_terms = st.slider("Top termes (analyse)", 3, 30, 10)

    cc1, cc2, cc3 = st.columns(3)
    if cc1.button("📚 Jeux du cluster"):
        r = get(f"/recommend/cluster/{int(cluster_id)}", {"sample": sample})
        if r.ok:
            d = r.json()
            st.success(f"Cluster {d.get('cluster')} — {d.get('count')} jeux")
            for i, g in enumerate(d.get("games", []), start=1):
                st.markdown(f"- {i}. **{g.get('title','?')}**  _{g.get('genres','')}_")
        else:
            st.error(f"{r.status_code} — {r.text}")

    if cc2.button("🧭 Cluster aléatoire"):
        r = get("/recommend/random-cluster", {"sample": 12})
        if r.ok:
            d = r.json()
            st.info("Extrait aléatoire :")
            for i, g in enumerate(d.get("games", []), start=1):
                st.markdown(f"- {i}. **{g.get('title','?')}**  _{g.get('genres','')}_")
        else:
            st.error(f"{r.status_code} — {r.text}")

    if cc3.button("🔍 Cluster Explore (top termes)"):
        r = get("/recommend/cluster-explore", {"top_terms": top_terms})
        if r.ok:
            d = r.json()
            st.json(d)
        else:
            st.error(f"{r.status_code} — {r.text}")

# =========================
# --- TAB 3: MODEL/MON ---
# =========================
with tabs[2]:
    guard_auth()
    st.subheader("📊 Métriques modèle")
    r = get("/model/metrics")
    if r.ok:
        metrics = r.json()
        show_metrics_pills(metrics)
    else:
        st.error(f"{r.status_code} — {r.text}")

    st.divider()
    st.subheader("🧪 Évaluer rapidement")
    default_tests = ["RPG", "Action", "Indie", "Simulation"]
    tests = st.text_input("Queries (séparées par virgule)", value=", ".join(default_tests))
    if st.button("🧪 Lancer l’évaluation"):
        # /model/evaluate utilise Query param test_queries=list -> on passe ?test_queries=a&test_queries=b...
        try:
            params = [("test_queries", q.strip()) for q in tests.split(",") if q.strip()]
            url = f"{api_base()}/model/evaluate"
            r = requests.get(url, headers=bearer_headers(), params=params, timeout=30)
            if r.ok:
                st.json(r.json())
            else:
                st.error(f"{r.status_code} — {r.text}")
        except Exception as e:
            st.error(f"Erreur réseau: {e}")

    st.divider()
    st.subheader("📈 Prometheus endpoint")
    st.caption("L’API expose /metrics (via Instrumentator). Tu peux l’intégrer à Prometheus/Grafana.")

# =========================
# ---- TAB 4: ADMIN ------- 
# =========================
with tabs[3]:
    guard_auth()
    st.subheader("🛠️ Entraîner / Recharger")
    c1, c2 = st.columns(2)
    with c1:
        version = st.text_input("Version (optionnel)", placeholder="api-YYYYMMDD-HHMMSS")
        force = st.toggle("Forcer l’entraînement sur appel ML (fallback interne)", value=False,
                          help="Note: l’API gère déjà un ensure train côté serveur si besoin.")
    with c2:
        st.markdown(" ")
        if st.button("🚀 Entraîner maintenant"):
            payload = {"version": version or None, "force_retrain": False}
            r = post("/model/train", payload)
            if r.ok:
                d = r.json()
                st.success(f"✅ Train OK — v{d.get('version')} — {round(d.get('duration',0),2)}s")
                st.json(d.get("result"))
            else:
                st.error(f"{r.status_code} — {r.text}")

    st.divider()
    st.subheader("🩺 Health & debug")
    if st.button("🔎 /healthz"):
        try:
            hr = requests.get(f"{api_base()}/healthz", timeout=15)
            if hr.ok:
                st.json(hr.json())
            else:
                st.error(f"{hr.status_code} — {hr.text}")
        except Exception as e:
            st.error(f"Erreur: {e}")

st.caption("Astuce: appuie sur <span class='kbd'>R</span> pour relancer (ou le bouton Rerun).", unsafe_allow_html=True)
