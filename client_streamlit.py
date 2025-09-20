import json
import requests
import streamlit as st
from typing import Any, Dict, List, Optional

############################################
# Streamlit UI — Frontend for Games API ML #
############################################
# API docs: https://game-app-y8be.onrender.com/docs#/
# This app lets you:
# - Register or Login to obtain a JWT
# - Query ML recommendations (/recommend/ml)
# - Find similar games by id or title (/recommend/similar-game)
# - Explore clusters, genres, titles
# - View /model/metrics, trigger /model/train, run /model/evaluate
#
# How to run locally:
#   1) pip install streamlit requests
#   2) streamlit run streamlit_app_games_frontend.py
#
# Tip: You can change the API base URL in the left sidebar.

DEFAULT_API_BASE = "https://game-app-y8be.onrender.com"

##########################
# Helpers & Session State #
##########################

if "api_base" not in st.session_state:
    st.session_state.api_base = DEFAULT_API_BASE
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "history" not in st.session_state:
    st.session_state.history = []  # store last queries


def api_url(path: str) -> str:
    base = st.session_state.get("api_base", DEFAULT_API_BASE) or DEFAULT_API_BASE
    return base.rstrip("/") + path


def _auth_headers() -> Dict[str, str]:
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def post_form_token(username: str, password: str) -> Dict[str, Any]:
    # FastAPI OAuth2 token endpoint expects application/x-www-form-urlencoded
    url = api_url("/token")
    try:
        resp = requests.post(url, data={"username": username, "password": password}, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"{resp.status_code}: {resp.text}"}
    except requests.RequestException as e:
        return {"error": str(e)}


def register_user(username: str, password: str) -> Dict[str, Any]:
    url = api_url("/register")
    try:
        resp = requests.post(url, data={"username": username, "password": password}, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"{resp.status_code}: {resp.text}"}
    except requests.RequestException as e:
        return {"error": str(e)}


def get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = api_url(path)
    try:
        resp = requests.get(url, params=params, headers=_auth_headers(), timeout=30)
        if resp.headers.get("content-type", "").startswith("application/json"):
            payload = resp.json()
        else:
            payload = {"raw": resp.text}
        if resp.ok:
            return payload
        return {"error": f"{resp.status_code}", "detail": payload}
    except requests.RequestException as e:
        return {"error": str(e)}


def post_json(path: str, payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = api_url(path)
    try:
        resp = requests.post(url, params=params, json=payload or {}, headers=_auth_headers(), timeout=60)
        if resp.headers.get("content-type", "").startswith("application/json"):
            data = resp.json()
        else:
            data = {"raw": resp.text}
        if resp.ok:
            return data
        return {"error": f"{resp.status_code}", "detail": data}
    except requests.RequestException as e:
        return {"error": str(e)}


########################
# Sidebar Configuration #
########################

st.sidebar.title("⚙️ Configuration")
api_base_input = st.sidebar.text_input("API base URL", value=st.session_state.api_base)
if api_base_input != st.session_state.api_base:
    st.session_state.api_base = api_base_input

# Health check
if st.sidebar.button("🔍 Test /healthz"):
    res = get_json("/healthz")
    if "error" in res:
        st.sidebar.error(f"Health check failed: {res}")
    else:
        st.sidebar.success("OK ✅")
        st.sidebar.json(res)

############################
# Auth: Login / Registration #
############################

st.title("🎮 Games Reco — Frontend ML")
st.caption("Client Streamlit pour l'API avec JWT + endpoints de recommandation et de monitoring.")

with st.expander("🔐 Connexion / Inscription", expanded=st.session_state.token is None):
    tab_login, tab_register = st.tabs(["Se connecter", "S'inscrire"])

    with tab_login:
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Nom d'utilisateur", key="login_user")
        with col2:
            password = st.text_input("Mot de passe", type="password", key="login_pass")
        if st.button("Obtenir un token (JWT)"):
            if not username or not password:
                st.warning("Veuillez saisir un identifiant et un mot de passe.")
            else:
                out = post_form_token(username, password)
                if "access_token" in out:
                    st.session_state.token = out["access_token"]
                    st.session_state.username = username
                    mode = out.get("mode")
                    if mode == "demo":
                        st.info("Connecté en mode DEMO (DB indisponible).")
                    st.success("Connecté ! Le token a été stocké en session.")
                else:
                    st.error(f"Impossible d'obtenir un token: {out}")

    with tab_register:
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            new_user = st.text_input("Nouvel identifiant", key="reg_user")
        with rcol2:
            new_pass = st.text_input("Mot de passe", type="password", key="reg_pass")
        if st.button("Créer un compte"):
            if not new_user or not new_pass:
                st.warning("Renseignez un identifiant et un mot de passe.")
            else:
                res = register_user(new_user, new_pass)
                if res.get("ok"):
                    st.success("Utilisateur créé. Vous pouvez maintenant vous connecter.")
                else:
                    st.error(f"Inscription échouée: {res}")

if st.session_state.token:
    st.success(f"Connecté en tant que **{st.session_state.username}**")
    if st.button("Se déconnecter"):
        st.session_state.token = None
        st.session_state.username = None
        st.experimental_rerun()
else:
    st.info("Astuce: si la base n'est pas disponible côté API, un *demo login* peut être activé par l'API.")

#########################
# Tabs for API Features  #
#########################

if st.session_state.token:
    tabs = st.tabs([
        "🔎 Recommandations (ML)",
        "🎯 Jeu similaire",
        "🔤 Par titre",
        "🏷️ Par genre",
        "🧩 Cluster ID",
        "🗺️ Explorer clusters",
        "🎲 Cluster aléatoire",
        "📊 Modèle / Monitoring",
    ])

    # 1) /recommend/ml
    with tabs[0]:
        st.subheader("🔎 Recommandations (ML)")
        q = st.text_input("Votre requête (ex: 'RPG open world', 'indie platformer', 'space strategy')", key="ml_q")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            k = st.number_input("k", min_value=1, max_value=50, value=10, step=1)
        with c2:
            min_conf = st.slider("Confiance minimale", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        with c3:
            run_ml = st.button("Obtenir des recos")
        if run_ml:
            payload = {"query": q or "", "k": int(k), "min_confidence": float(min_conf)}
            res = post_json("/recommend/ml", payload)
            if "error" in res:
                st.error(res)
            else:
                st.session_state.history.append({"type": "ml", "query": q, "response": res})
                st.write(f"**Version modèle:** {res.get('model_version', 'N/A')} — **latence:** {res.get('latency_ms', 0):.1f} ms")
                recs = res.get("recommendations", [])
                if recs:
                    st.dataframe(recs, use_container_width=True)
                else:
                    st.info("Aucune recommandation retournée.")

    # 2) /recommend/similar-game
    with tabs[1]:
        st.subheader("🎯 Recommandations similaires à un jeu")
        mode = st.radio("Choisir le mode", ["Par ID", "Par titre"], horizontal=True)
        payload: Dict[str, Any] = {"k": st.number_input("k", 1, 50, 10, key="sim_k")}
        if mode == "Par ID":
            gid = st.number_input("game_id (RAWG id)", min_value=1, step=1, value=3498)
            payload["game_id"] = int(gid)
        else:
            title = st.text_input("Titre du jeu", value="The Witcher 3")
            if title:
                payload["title"] = title
        if st.button("Trouver des jeux similaires"):
            res = post_json("/recommend/similar-game", payload)
            if "error" in res:
                st.error(res)
            else:
                st.write(f"**Version modèle:** {res.get('model_version', 'N/A')}")
                st.json({"source_id": res.get("source_id"), "count": len(res.get("recommendations", []))})
                st.dataframe(res.get("recommendations", []), use_container_width=True)

    # 3) /recommend/by-title/{title}
    with tabs[2]:
        st.subheader("🔤 Recos par similarité de titre")
        title = st.text_input("Titre à rapprocher", value="Hades", key="rt_title")
        rk = st.number_input("k", 1, 50, 10, key="rt_k")
        if st.button("Recommander par titre"):
            res = get_json(f"/recommend/by-title/{title}", params={"k": int(rk)})
            if "error" in res:
                st.error(res)
            else:
                st.dataframe(res.get("recommendations", []), use_container_width=True)

    # 4) /recommend/by-genre/{genre}
    with tabs[3]:
        st.subheader("🏷️ Recos par genre")
        genre = st.text_input("Genre (ex: Action, RPG, Indie)", value="Action", key="rg_genre")
        gk = st.number_input("k", 1, 50, 10, key="rg_k")
        if st.button("Recommander par genre"):
            res = get_json(f"/recommend/by-genre/{genre}", params={"k": int(gk)})
            if "error" in res:
                st.error(res)
            else:
                st.dataframe(res.get("recommendations", []), use_container_width=True)

    # 5) /recommend/cluster/{cluster_id}
    with tabs[4]:
        st.subheader("🧩 Jeux d'un cluster")
        cid = st.number_input("cluster_id", min_value=0, step=1, value=0)
        sample = st.number_input("Taille d'échantillon", min_value=1, max_value=500, value=50)
        if st.button("Charger le cluster"):
            res = get_json(f"/recommend/cluster/{int(cid)}", params={"sample": int(sample)})
            if "error" in res:
                st.error(res)
            else:
                st.write(f"Cluster {res.get('cluster')} — {res.get('count')} jeux")
                st.dataframe(res.get("games", []), use_container_width=True)

    # 6) /recommend/cluster-explore
    with tabs[5]:
        st.subheader("🗺️ Exploration des clusters (termes)")
        topn = st.slider("Nombre de termes par cluster", 3, 30, 10)
        if st.button("Explorer"):
            res = get_json("/recommend/cluster-explore", params={"top_terms": int(topn)})
            if "error" in res:
                st.error(res)
            else:
                st.json(res)

    # 7) /recommend/random-cluster
    with tabs[6]:
        st.subheader("🎲 Cluster aléatoire")
        sample = st.number_input("Taille d'échantillon", 1, 200, 12, key="rand_sample")
        if st.button("Obtenir un cluster aléatoire"):
            res = get_json("/recommend/random-cluster", params={"sample": int(sample)})
            if "error" in res:
                st.error(res)
            else:
                st.json({"count": len(res.get("games", []))})
                st.dataframe(res.get("games", []), use_container_width=True)

    # 8) Model / Monitoring
    with tabs[7]:
        st.subheader("📊 État du modèle & Monitoring")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Voir /model/metrics"):
                metrics = get_json("/model/metrics")
                if "error" in metrics:
                    st.error(metrics)
                else:
                    st.json(metrics)
        with c2:
            if st.button("Évaluer le modèle (/model/evaluate)"):
                # The FastAPI function expects a POST with query params like ?test_queries=RPG&test_queries=Action
                params = [("test_queries", v) for v in ["RPG", "Action", "Indie", "Simulation"]]
                # requests can't pass duplicate keys via dict, so we build the query string manually via 'params' list of tuples
                try:
                    url = api_url("/model/evaluate")
                    resp = requests.post(url, headers=_auth_headers(), params=params, timeout=60)
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
                    if resp.ok:
                        st.json(data)
                    else:
                        st.error({"status": resp.status_code, "detail": data})
                except requests.RequestException as e:
                    st.error(str(e))
        with c3:
            version = st.text_input("Nouvelle version (optionnel)", value="")
            force = st.checkbox("Forcer l'entraînement (si applicable)")
            if st.button("(Re)entraîner le modèle"):
                payload = {"version": version or None, "force_retrain": bool(force)}
                res = post_json("/model/train", payload)
                if "error" in res:
                    st.error(res)
                else:
                    st.success("Entraînement demandé avec succès")
                    st.json(res)

    st.divider()
    with st.expander("🧾 Historique des requêtes (session)"):
        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history[-20:]), start=1):
                st.markdown(f"**{i}.** `{item['type']}` — `{item.get('query','')}`")
        else:
            st.caption("Aucun historique pour le moment.")

else:
    st.info("Connectez-vous pour accéder aux recommandations et fonctionnalités du modèle.")
