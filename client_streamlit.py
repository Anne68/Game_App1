import os
import json
import time
from typing import Dict, Any, Optional, List

import requests
import streamlit as st
import pandas as pd

# =============================
# Configuration
# =============================
# Point this to your deployed FastAPI base URL (without trailing slash)
# Examples:
#   - Local:        http://127.0.0.1:8000
#   - Render:       https://game-app-y8be.onrender.com
#   - AlwaysData:   https://your-domain.alwaysdata.net
API_BASE_URL = os.getenv("API_BASE_URL", "https://game-app-y8be.onrender.com")

# Optional demo credentials if your API is in demo mode when DB is unavailable
DEMO_USERNAME = os.getenv("DEMO_USERNAME", "demo")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo")

st.set_page_config(
    page_title="Games Recommender UI",
    page_icon="🎮",
    layout="wide",
)

# =============== Helpers ===============

def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def api_get(path: str, params: Optional[Dict[str, Any]] = None, require_auth: bool = False):
    url = f"{API_BASE_URL}{path}"
    try:
        headers = _headers() if require_auth else {"Content-Type": "application/json"}
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 401:
            raise PermissionError("Unauthorized. Please login first.")
        resp.raise_for_status()
        return resp.json()
    except PermissionError:
        raise
    except Exception as e:
        raise RuntimeError(f"GET {path} failed: {e}")


def api_post(path: str, payload: Optional[Dict[str, Any]] = None, form: Optional[Dict[str, Any]] = None, require_auth: bool = True):
    url = f"{API_BASE_URL}{path}"
    try:
        if form is not None:
            # form-encoded (for /token, /auth/token, /register)
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            resp = requests.post(url, headers=headers, data=form, timeout=30)
        else:
            headers = _headers() if require_auth else {"Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json=payload or {}, timeout=60)
        if resp.status_code == 401:
            raise PermissionError("Unauthorized. Please login first.")
        # Let FastAPI error surfaces show nicely
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"POST {path} -> HTTP {resp.status_code}: {detail}")
        return resp.json()
    except PermissionError:
        raise
    except Exception as e:
        raise RuntimeError(f"POST {path} failed: {e}")


# =============== UI Pieces ===============

def show_header():
    st.markdown(
        """
        <style>
        .headline {font-size: 2rem; font-weight: 800; margin-bottom: .2rem}
        .subtle {opacity:.7}
        .token-pill {background:#f0f2f6; padding:6px 10px; border-radius:999px; font-size:.85rem}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='headline'>🎮 Games API ML – Streamlit</div>", unsafe_allow_html=True)
    st.caption("Frontend minimaliste pour l'API de recommandations de jeux (C9→C13).")


def sidebar_auth():
    with st.sidebar:
        st.subheader("Connexion")
        logged = st.session_state.get("logged", False)
        if not logged:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Nom d'utilisateur", key="login_username", value=DEMO_USERNAME)
                password = st.text_input("Mot de passe", type="password", key="login_password", value=DEMO_PASSWORD)
                col1, col2 = st.columns(2)
                do_login = col1.form_submit_button("Se connecter")
                do_register = col2.form_submit_button("Créer le compte")
            if do_login:
                try:
                    data = api_post("/auth/token", form={"username": username, "password": password}, require_auth=False)
                    st.session_state.token = data.get("access_token")
                    st.session_state.username = username
                    st.session_state.logged = True
                    st.success("Connecté ✅")
                except Exception as e:
                    st.error(str(e))
            if do_register:
                try:
                    out = api_post("/register", form={"username": username, "password": password}, require_auth=False)
                    st.success(f"Compte créé (user_id={out.get('user_id')}). Vous pouvez vous connecter.")
                except Exception as e:
                    st.error(str(e))
        else:
            st.success(f"Connecté en tant que {st.session_state.get('username')}")
            if st.button("Se déconnecter"):
                st.session_state.clear()
                st.experimental_rerun()

        st.markdown("---")
        st.caption("Base API :")
        st.markdown(f"<span class='token-pill'>{API_BASE_URL}</span>", unsafe_allow_html=True)


# =============== Pages ===============

def page_status():
    st.subheader("Statut & Monitoring")
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Actualiser le statut API"):
            st.session_state["_health"] = None  # erase cache
        try:
            if st.session_state.get("_health") is None:
                st.session_state["_health"] = api_get("/healthz")
            st.json(st.session_state["_health"])
        except Exception as e:
            st.error(str(e))
    with colB:
        try:
            if st.session_state.get("logged"):
                metrics = api_get("/model/metrics", require_auth=True)
                st.metric("Version modèle", metrics.get("model_version", "?"))
                c1, c2, c3 = st.columns(3)
                c1.metric("Entraîné ?", "✅" if metrics.get("is_trained") else "❌")
                c2.metric("Predictions totales", metrics.get("total_predictions", 0))
                c3.metric("Confiance moyenne", f"{metrics.get('avg_confidence', 0.0):.3f}")
                st.caption("Détails complets :")
                st.json(metrics)
            else:
                st.info("Connectez-vous pour voir les métriques du modèle.")
        except Exception as e:
            st.error(str(e))


def page_recommendations():
    st.subheader("🔎 Recommandations (ML)")
    if not st.session_state.get("logged"):
        st.warning("Veuillez vous connecter pour utiliser le modèle.")
        return

    with st.form("ml_reco_form"):
        query = st.text_input("Votre requête (mots-clés / description / genres)", value="RPG open world")
        k = st.slider("Nombre de résultats (k)", 1, 50, 10)
        min_conf = st.slider("Confiance minimale", 0.0, 1.0, 0.10, step=0.01)
        submitted = st.form_submit_button("Recommander")

    if submitted:
        try:
            payload = {"query": query, "k": k, "min_confidence": min_conf}
            t0 = time.time()
            data = api_post("/recommend/ml", payload)
            dt_ms = (time.time() - t0) * 1000
            recs = data.get("recommendations", [])
            if not recs:
                st.info("Aucune recommandation.")
                return
            # Normalize to DataFrame
            df = pd.json_normalize(recs)
            # Friendly columns if present
            rename_map = {"score": "score", "confidence": "confiance", "id": "game_id", "title": "titre", "genres": "genres", "rating": "rating", "metacritic": "metacritic"}
            cols = [c for c in rename_map if c in df.columns]
            df = df[cols].rename(columns=rename_map)
            st.caption(f"{len(df)} résultats – {dt_ms:.1f} ms")
            st.dataframe(df, use_container_width=True)
            with st.expander("Réponse brute"):
                st.json(data)
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.subheader("🧭 Recommandations avancées")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.caption("Par Jeu (ID ou Titre)")
        with st.form("similar_form"):
            game_id = st.number_input("game_id", min_value=0, step=1, value=0)
            title = st.text_input("title", value="")
            k2 = st.slider("k", 1, 50, 10, key="k_sim")
            go_sim = st.form_submit_button("Trouver similaires")
        if go_sim:
            try:
                payload = {"game_id": int(game_id) if game_id else None, "title": title or None, "k": k2}
                data = api_post("/recommend/similar-game", payload)
                df = pd.json_normalize(data.get("recommendations", []))
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Aucun résultat.")
                with st.expander("Réponse brute"):
                    st.json(data)
            except Exception as e:
                st.error(str(e))

    with c2:
        st.caption("Par Titre (similarité)")
        title_q = st.text_input("Titre exact/partiel", value="Hades")
        k_title = st.slider("k", 1, 50, 10, key="k_title")
        if st.button("Recommander par titre"):
            try:
                data = api_get(f"/recommend/by-title/{title_q}", params={"k": k_title}, require_auth=True)
                df = pd.json_normalize(data.get("recommendations", []))
                st.dataframe(df, use_container_width=True)
                with st.expander("Réponse brute"):
                    st.json(data)
            except Exception as e:
                st.error(str(e))

    with c3:
        st.caption("Par Genre")
        genre = st.text_input("Genre", value="Action")
        k_gen = st.slider("k", 1, 50, 10, key="k_gen")
        if st.button("Recommander par genre"):
            try:
                data = api_get(f"/recommend/by-genre/{genre}", params={"k": k_gen}, require_auth=True)
                df = pd.json_normalize(data.get("recommendations", []))
                st.dataframe(df, use_container_width=True)
                with st.expander("Réponse brute"):
                    st.json(data)
            except Exception as e:
                st.error(str(e))

    with c4:
        st.caption("Clusters")
        top_terms = st.slider("Top terms", 3, 30, 10)
        sample = st.slider("Sample", 1, 200, 12)
        colX, colY = st.columns(2)
        if colX.button("Explorer clusters"):
            try:
                data = api_get("/recommend/cluster-explore", params={"top_terms": top_terms}, require_auth=True)
                st.json(data)
            except Exception as e:
                st.error(str(e))
        if colY.button("Cluster aléatoire"):
            try:
                data = api_get("/recommend/random-cluster", params={"sample": sample}, require_auth=True)
                games = data if isinstance(data, list) else data.get("games", data)
                df = pd.json_normalize(games)
                st.dataframe(df, use_container_width=True)
                with st.expander("Réponse brute"):
                    st.json(data)
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.subheader("Jeux d'un cluster donné")
    cluster_id = st.number_input("cluster_id", min_value=0, step=1, value=0)
    sample2 = st.slider("Sample", 1, 500, 50, key="sample_cluster")
    if st.button("Charger le cluster"):
        try:
            data = api_get(f"/recommend/cluster/{int(cluster_id)}", params={"sample": int(sample2)}, require_auth=True)
            df = pd.json_normalize(data.get("games", []))
            st.dataframe(df, use_container_width=True)
            with st.expander("Réponse brute"):
                st.json(data)
        except Exception as e:
            st.error(str(e))


def page_training():
    st.subheader("⚙️ Entraînement & Évaluation")
    if not st.session_state.get("logged"):
        st.warning("Veuillez vous connecter.")
        return

    with st.expander("Entraîner le modèle"):
        version = st.text_input("Version (optionnel)", value="")
        force = st.checkbox("Forcer le réentraînement à la prochaine requête (côté API)", value=False, help="Utile si vous avez modifié la BDD.")
        if st.button("Lancer l'entraînement maintenant"):
            try:
                payload = {"version": version or None, "force_retrain": bool(force)}
                data = api_post("/model/train", payload)
                st.success("Entraînement lancé / réalisé ✔️")
                st.json(data)
            except Exception as e:
                st.error(str(e))

    with st.expander("Évaluer (smoke-test)"):
        default_q = ["RPG", "Action", "Indie", "Simulation"]
        q = st.text_area("Liste de requêtes (séparées par des virgules)", value=", ".join(default_q))
        if st.button("Évaluer"):
            try:
                lst = [x.strip() for x in q.split(",") if x.strip()]
                # /model/evaluate expects a query param list named test_queries
                params = []
                for item in lst:
                    params.append(("test_queries", item))
                url = f"{API_BASE_URL}/model/evaluate"
                resp = requests.post(url, headers=_headers(), params=params, timeout=60)
                if not resp.ok:
                    raise RuntimeError(resp.text)
                st.json(resp.json())
            except Exception as e:
                st.error(str(e))


# =============== Main Layout ===============

def main():
    show_header()
    sidebar_auth()

    tabs = st.tabs(["Statut", "Recommandations", "Entraînement / Évaluation"])  # light, focused UI

    with tabs[0]:
        page_status()
    with tabs[1]:
        page_recommendations()
    with tabs[2]:
        page_training()

    st.markdown("---")
    st.caption("💡 Astuces : Assurez-vous que la base MySQL est prête côté API. En mode démo (DB indisponible), utilisez les identifiants par défaut si activés.")


if __name__ == "__main__":
    main()
