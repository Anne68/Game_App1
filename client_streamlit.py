import os
import requests
import streamlit as st

# ---------------- Config ----------------
API_BASE_URL = (
    st.secrets.get("API_URL")
    or os.getenv("API_URL")
    or "https://game-app-y8be.onrender.com"
)

st.set_page_config(page_title="Games UI", page_icon="🎮", layout="wide")

# ---------------- Session ----------------
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

def api_headers():
    h = {"accept": "application/json"}
    if st.session_state.token:
        h["Authorization"] = f"Bearer {st.session_state.token}"
    return h

def api_get(path, params=None):
    r = requests.get(f"{API_BASE_URL}{path}", headers=api_headers(), params=params, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()

def api_post(path, json=None, data=None):
    r = requests.post(f"{API_BASE_URL}{path}", headers=api_headers(), json=json, data=data, timeout=60)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text
        raise RuntimeError(detail or "Erreur API")
    return r.json()

# ---------------- Styles ----------------
st.markdown("""
<style>
.badge {display:inline-block;padding:.2rem .45rem;border-radius:999px;border:1px solid rgba(0,0,0,.1);margin-right:.25rem;margin-bottom:.25rem;font-size:.78rem;opacity:.9}
.card {border:1px solid rgba(0,0,0,.08);border-radius:14px;padding:14px;margin-bottom:12px;background:rgba(255,255,255,.6)}
.card h4{margin:0 0 6px 0}
.price {font-weight:600}
.meta {opacity:.8;font-size:.9rem}
.actions a {text-decoration:none}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("## 🎮 **Games UI**")

tab_reg, tab_auth, tab_search, tab_recos = st.tabs(["🆕 Inscription", "🔑 Auth", "🔎 Recherche", "✨ Recos"])

# ---------------- Register ----------------
with tab_reg:
    st.subheader("Créer un compte")
    with st.form("reg"):
        u = st.text_input("Nom d'utilisateur")
        p = st.text_input("Mot de passe", type="password")
        ok = st.form_submit_button("S'inscrire")
    if ok:
        try:
            api_post("/register", data={"username": u, "password": p})
            st.success("Compte créé. Connectez-vous dans l’onglet **Auth**.")
        except Exception as e:
            st.error(f"Inscription impossible : {e}")

# ---------------- Auth ----------------
with tab_auth:
    if st.session_state.token:
        st.success(f"Connecté en tant que **{st.session_state.username}**")
        if st.button("Se déconnecter"):
            st.session_state.token = None
            st.session_state.username = None
            st.experimental_rerun()
    else:
        st.subheader("Connexion")
        with st.form("login"):
            u = st.text_input("Nom d'utilisateur", value=st.session_state.username or "")
            p = st.text_input("Mot de passe", type="password")
            ok = st.form_submit_button("Se connecter")
        if ok:
            try:
                r = api_post("/token", data={"username": u, "password": p})
                st.session_state.token = r["access_token"]
                st.session_state.username = u
                st.success("Connecté ✅")
            except Exception as e:
                st.error(f"Connexion impossible : {e}")

# ---------------- Recherche (cartes) ----------------
with tab_search:
    st.subheader("🔎 Recherche par titre")
    q = st.text_input("Titre contient…", placeholder="Mario, Zelda, ...")
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Chercher", type="primary"):
            st.session_state["__do_search"] = True
    results_area = st.container()
    if st.session_state.get("__do_search") and q.strip():
        try:
            data = api_get(f"/games/by-title/{q.strip()}")
            items = data.get("results", [])
            if not items:
                results_area.info("Aucun résultat.")
            else:
                with results_area:
                    for r in items:
                        price = r.get("best_price")
                        shop = r.get("best_shop")
                        site = r.get("site_url")
                        plats = r.get("platforms") or []
                        rating = r.get("rating")
                        meta = r.get("metacritic")

                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown(f"<h4>{r['title']}</h4>", unsafe_allow_html=True)
                        cols = st.columns([3,2,2,2])
                        with cols[0]:
                            if plats:
                                st.markdown(" ".join([f'<span class="badge">{p}</span>' for p in plats]), unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="meta">Plateformes inconnues</span>', unsafe_allow_html=True)
                        with cols[1]:
                            st.markdown(f'<span class="meta">⭐ Note: {rating if rating is not None else "—"}</span>', unsafe_allow_html=True)
                        with cols[2]:
                            st.markdown(f'<span class="meta">🧪 Metacritic: {meta if meta is not None else "—"}</span>', unsafe_allow_html=True)
                        with cols[3]:
                            if price:
                                txt = f'<span class="price">{price}</span>'
                                if shop: txt += f' @ {shop}'
                                if site:
                                    st.markdown(f'<div class="actions"><a href="{site}" target="_blank">{txt}</a></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(txt, unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="meta">💸 Prix inconnu</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            results_area.error(f"Erreur recherche : {e}")
    elif st.session_state.get("__do_search"):
        results_area.warning("Saisissez un titre à rechercher.")

# ---------------- Recos ----------------
with tab_recos:
    st.subheader("✨ Recommandations")
    prompt = st.text_input("Prompt (genre, mot-clé…)", value="RPG")
    k = st.slider("Nombre de propositions", min_value=1, max_value=50, value=10)
    if st.button("Lancer la reco"):
        try:
            res = api_post("/recommend/ml", json={"query": prompt, "k": k, "min_confidence": 0.0})
            recs = res.get("recommendations", [])
            if not recs:
                st.info("Pas de recommandations.")
            else:
                for r in recs:
                    title = r.get("title") or r.get("name") or f"#{r.get('id')}"
                    score = r.get("score") or r.get("similarity") or r.get("confidence")
                    st.markdown(f"- **{title}** — score: `{score}`")
        except Exception as e:
            st.error(f"{e}")
