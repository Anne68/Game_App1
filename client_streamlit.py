import os
import requests
import streamlit as st
from typing import Any, Dict, List, Optional

# -----------------------------
# Config de base
# -----------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Games+ Reco & Prix", page_icon="🎮", layout="wide")

# -----------------------------
# Helpers HTTP
# -----------------------------
def api_get(path: str, token: Optional[str] = None, **params) -> requests.Response:
    headers = {"accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{API_BASE_URL}{path}"
    return requests.get(url, headers=headers, params=params, timeout=30)


def api_post(path: str, token: Optional[str] = None, json: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None) -> requests.Response:
    headers = {"accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{API_BASE_URL}{path}"
    return requests.post(url, headers=headers, json=json, data=data, timeout=60)


# -----------------------------
# UI helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def get_health() -> Dict[str, Any]:
    try:
        r = api_get("/healthz")
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}


def ensure_token() -> Optional[str]:
    if "token" in st.session_state and st.session_state.token:
        return st.session_state.token

    with st.sidebar:
        st.subheader("🔐 Connexion")
        username = st.text_input("Nom d'utilisateur", key="login_user")
        password = st.text_input("Mot de passe", type="password", key="login_pwd")
        if st.button("Se connecter", use_container_width=True):
            if not username or not password:
                st.error("Merci de saisir identifiant et mot de passe")
            else:
                try:
                    r = api_post("/token", data={"username": username, "password": password})
                    if r.ok:
                        j = r.json()
                        st.session_state.token = j.get("access_token")
                        st.success("Connecté ✅")
                        st.rerun()
                    else:
                        st.error(f"Échec connexion: {r.status_code} - {r.text}")
                except Exception as e:
                    st.error(f"Erreur réseau: {e}")
    return None


def reco_cards(items: List[Dict[str, Any]]):
    if not items:
        st.info("Aucune recommandation trouvée.")
        return
    cols = st.columns(3)
    for i, it in enumerate(items):
        with cols[i % 3]:
            title = it.get("title") or it.get("name") or f"Game #{it.get('id', '?')}"
            score = it.get("score") or it.get("confidence") or it.get("similarity")
            rating = it.get("rating")
            meta = it.get("metacritic") or it.get("metascore")
            platforms = it.get("platforms") or []
            if isinstance(platforms, str):
                platforms = [p.strip() for p in platforms.split(",") if p.strip()]

            st.markdown(f"### {title}")
            if score is not None:
                st.progress(min(max(float(score), 0.0), 1.0))
                st.caption(f"Score modèle: {score:.2f}")
            if rating is not None or meta is not None:
                st.text(f"Rating: {rating if rating is not None else '-'} | Metacritic: {meta if meta is not None else '-'}")
            if platforms:
                st.text("Plateformes: " + ", ".join(platforms))


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🎮 Games+ App")
    st.caption("Front-end Streamlit pour votre API ML")
    st.divider()
    st.text_input("URL de l'API", value=API_BASE_URL, key="api_url_input")
    if st.session_state.get("api_url_input") != API_BASE_URL:
        API_BASE_URL = st.session_state.api_url_input
    st.divider()
    h = get_health()
    if h:
        st.markdown("**État API**")
        st.json({k: h[k] for k in ["status", "db_ready", "model_loaded", "model_version"] if k in h})
    else:
        st.warning("Impossible de joindre /healthz pour le moment.")


# -----------------------------
# Corps de page
# -----------------------------
st.title("🎯 Recommandations & Meilleur prix")

# Login (obligatoire pour consommer les endpoints protégés)
token = ensure_token()

# Onglets
onglets = st.tabs([
    "🔎 Par titre",
    "🎭 Par genre",
    "🧠 Recherche ML (mots-clés)",
    "💸 Meilleur prix",
    "📊 Modèle & clusters",
])

# ------------- Onglet: Par titre -------------
with onglets[0]:
    st.subheader("Recommander à partir d'un titre")
    col1, col2 = st.columns([3, 1])
    with col1:
        title = st.text_input("Titre du jeu", placeholder="Ex: The Witcher 3")
    with col2:
        k = st.slider("Nombre de recos", 1, 30, 10)
    if st.button("Recommander", disabled=not bool(token)):
        if not token:
            st.error("Connectez-vous d'abord.")
        elif not title:
            st.warning("Saisissez un titre.")
        else:
            r = api_get(f"/recommend/by-title/{title}", token=token, k=k)
            if r.ok:
                data = r.json()
                items = data.get("recommendations") or []
                reco_cards(items)
            else:
                st.error(f"Erreur API: {r.status_code} - {r.text}")

# ------------- Onglet: Par genre -------------
with onglets[1]:
    st.subheader("Recommander par genre")
    col1, col2 = st.columns([3, 1])
    with col1:
        genre = st.text_input("Genre", placeholder="Ex: RPG, Action, Indie…")
    with col2:
        k2 = st.slider("Nombre de recos", 1, 30, 10, key="k_genre")
    if st.button("Recommander par genre", disabled=not bool(token)):
        if not token:
            st.error("Connectez-vous d'abord.")
        elif not genre:
            st.warning("Saisissez un genre.")
        else:
            r = api_get(f"/recommend/by-genre/{genre}", token=token, k=k2)
            if r.ok:
                data = r.json()
                items = data.get("recommendations") or []
                reco_cards(items)
            else:
                st.error(f"Erreur API: {r.status_code} - {r.text}")

# ------------- Onglet: Recherche ML (mots-clés) -------------
with onglets[2]:
    st.subheader("Recherche hybride (titre/genre/mots-clés)")
    query = st.text_area(
        "Décrivez ce que vous cherchez",
        placeholder="Ex: RPG open world narratif avec crafting et quêtes secondaires",
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        k3 = st.slider("Top K", 1, 30, 10, key="k_ml")
    with c2:
        min_conf = st.slider("Confiance min.", 0.0, 1.0, 0.1, 0.05)
    if st.button("Lancer la recherche ML", disabled=not bool(token)):
        if not token:
            st.error("Connectez-vous d'abord.")
        elif not query.strip():
            st.warning("Saisissez une requête.")
        else:
            r = api_post("/recommend/ml", token=token, json={"query": query, "k": k3, "min_confidence": min_conf})
            if r.ok:
                j = r.json()
                st.caption(f"Version modèle: {j.get('model_version','?')} — Latence: {j.get('latency_ms',0):.0f} ms")
                items = j.get("recommendations") or []
                reco_cards(items)
            else:
                st.error(f"Erreur API: {r.status_code} - {r.text}")

# ------------- Onglet: Meilleur prix -------------
with onglets[3]:
    st.subheader("Trouver le meilleur prix")
    title_price = st.text_input("Titre du jeu à chercher", key="price_title", placeholder="Ex: Hades")
    region = st.selectbox("Région / boutique", ["auto", "EU", "US", "Global"], index=0)

    st.caption("💡 L'app tente d'abord un endpoint backend /price/best s'il existe; sinon, liens directs vers comparateurs.")

    if st.button("Chercher le meilleur prix", disabled=not bool(title_price)):
        # 1) Essayer un endpoint backend optionnel
        got_result = False
        if token:
            try:
                r = api_get("/price/best", token=token, title=title_price, region=None if region == "auto" else region)
                if r.ok:
                    data = r.json()
                    got_result = True
                    st.success("Résultats via API backend")
                    st.json(data)
                # 404 ou autre -> fallback liens
            except Exception:
                pass
        if not got_result:
            st.warning("Endpoint /price/best non disponible. Utilisez les comparateurs ci-dessous :")
            q = requests.utils.quote(title_price)
            colA, colB, colC = st.columns(3)
            with colA:
                st.link_button("IsThereAnyDeal", f"https://isthereanydeal.com/search/?q={q}", use_container_width=True)
            with colB:
                st.link_button("GG.deals", f"https://gg.deals/games/?title={q}", use_container_width=True)
            with colC:
                st.link_button("Steam (recherche)", f"https://store.steampowered.com/search/?term={q}", use_container_width=True)

# ------------- Onglet: Modèle & clusters -------------
with onglets[4]:
    st.subheader("Infos modèle & exploration de clusters")
    colm1, colm2 = st.columns(2)

    with colm1:
        st.markdown("**Métriques modèle**")
        if token:
            r = api_get("/model/metrics", token=token)
            if r.ok:
                st.json(r.json())
            else:
                st.error(f"Erreur /model/metrics: {r.status_code}")
        else:
            st.info("Connectez-vous pour voir les métriques.")

        st.markdown("**Évaluer sur requêtes de test**")
        if token:
            tests = st.text_input("Requêtes (séparées par virgules)", value="RPG, Action, Indie, Simulation")
            if st.button("Évaluer"):
                queries = [t.strip() for t in tests.split(",") if t.strip()]
                r = api_post("/model/evaluate", token=token, json=None, data=None,)
                # NB: l'endpoint /model/evaluate du backend lit la Query param côté FastAPI; ici, on affichera le retour brut
                if r.ok:
                    st.json(r.json())
                else:
                    st.error(f"Erreur /model/evaluate: {r.status_code} - {r.text}")

    with colm2:
        st.markdown("**Explorer un cluster**")
        if token:
            cid = st.number_input("ID cluster", min_value=0, value=0, step=1)
            sample = st.slider("Taille d'échantillon", 1, 200, 20)
            if st.button("Voir les jeux du cluster"):
                r = api_get(f"/recommend/cluster/{int(cid)}", token=token, sample=sample)
                if r.ok:
                    data = r.json()
                    st.write(f"Cluster {data.get('cluster')} — {data.get('count')} jeux")
                    items = data.get("games") or []
                    reco_cards(items)
                else:
                    st.error(f"Erreur /recommend/cluster: {r.status_code} - {r.text}")

        st.markdown("**Termes top par cluster**")
        if token and st.button("Afficher les top termes"):
            r = api_get("/recommend/cluster-explore", token=token, top_terms=10)
            if r.ok:
                st.json(r.json())
            else:
                st.error(f"Erreur /recommend/cluster-explore: {r.status_code} - {r.text}")

# Footer
st.divider()
st.caption("Propulsé par votre API Games+ (C9→C13) — Auth JWT, reco ML, monitoring Prometheus.")
