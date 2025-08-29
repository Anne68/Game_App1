# client_streamlit.py (version complète)

import os
import time
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🎮 Games UI", page_icon="🎮", layout="wide")
API_URL = os.getenv("ST_API_URL", "https://game-app-y8be.onrender.com").rstrip("/")

# ---------- CSS pour les cartes ----------
st.markdown("""
<style>
.card {
  border: 1px solid rgba(200,200,200,.18);
  background: rgba(255,255,255,.03);
  border-radius: 18px;
  padding: 14px 16px;
  height: 100%;
}
.card h3 {
  margin: 0 0 6px 0; font-size: 1.05rem;
}
.rowmeta { display:flex; gap:10px; margin:.35rem 0 .6rem 0; flex-wrap:wrap; }
.badge {
  font-size: .78rem; padding: 4px 8px; border-radius: 999px;
  background: rgba(125,125,255,.15); border:1px solid rgba(125,125,255,.25);
}
.badge.price { background: rgba(120,200,120,.15); border-color: rgba(120,200,120,.25);}
.badge.gray  { background: rgba(255,255,255,.07); border-color: rgba(255,255,255,.15);}
.platforms { margin:.25rem 0 .6rem 0; }
.platforms .chip {
  display:inline-block; margin:2px 6px 2px 0; padding:3px 8px; border-radius:999px;
  font-size:.78rem; background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.15);
}
.small { opacity:.7; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers HTTP ----------
def api_headers():
    tok = st.session_state.get("token")
    return {"Authorization": f"Bearer {tok}"} if tok else {}

def api_get(path, params=None):
    r = requests.get(f"{API_URL}{path}", headers=api_headers(), params=params or {}, timeout=30)
    if r.status_code == 401:
        st.error("Non autorisé — connecte-toi d’abord.")
        return {}
    if not r.ok:
        try:
            st.error(r.json())
        except Exception:
            st.error(r.text)
        return {}
    return r.json()

def login(username, password):
    r = requests.post(f"{API_URL}/token", data={"username": username, "password": password}, timeout=30)
    if r.ok:
        st.session_state["token"] = r.json()["access_token"]
        st.success("✅ Connecté")
    else:
        st.error("Échec de connexion")

def register(username, password):
    r = requests.post(f"{API_URL}/register", data={"username": username, "password": password}, timeout=30)
    if r.ok:
        st.success("Compte créé, tu peux te connecter.")
    else:
        try:
            st.error(r.json())
        except Exception:
            st.error(r.text)

# ---------- UI ----------
st.title("🎮 Games UI")

with st.sidebar:
    st.subheader("🔑 Auth")
    if "token" not in st.session_state:
        st.session_state["token"] = None

    if st.session_state["token"]:
        st.success("✅ Connecté")
        if st.button("Se déconnecter"):
            st.session_state["token"] = None
            st.rerun()
    else:
        mode = st.radio("Mode", ["Connexion", "Inscription"], horizontal=True)
        u = st.text_input("Utilisateur")
        p = st.text_input("Mot de passe", type="password")
        if st.button("Valider"):
            if mode == "Connexion":
                login(u, p)
            else:
                register(u, p)

tabs = st.tabs(["🆕 Inscription", "🔑 Auth", "🔎 Recherche", "✨ Recos"])

with tabs[0]:
    st.caption("Tu peux aussi créer un compte dans la barre latérale.")

with tabs[1]:
    st.caption("Connecte-toi via la barre latérale pour débloquer les appels API protégés.")

# ---------- Composant carte ----------
def render_game_card(g: dict):
    title = g.get("title") or "Sans titre"
    price = g.get("best_price") or "N/A"
    rating = g.get("rating")
    metac = g.get("metacritic")
    plats = g.get("platforms") or []

    # texte pour badges
    rating_txt = f"{rating:.2f}" if isinstance(rating, (int, float)) else "N/A"
    metac_txt  = f"{int(metac)}" if isinstance(metac, (int, float)) else "N/A"

    # chips plateformes
    chips = " ".join([f"<span class='chip'>{p}</span>" for p in plats])

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)

        st.markdown(f"<div class='platforms'>{chips}</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='rowmeta'>"
            f"<span class='badge price'>💶 {price}</span>"
            f"<span class='badge'>⭐ {rating_txt}</span>"
            f"<span class='badge gray'>📰 Metacritic: {metac_txt}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        link = g.get("site_url")
        if link:
            st.link_button("Voir l’offre", link, use_container_width=True)
        else:
            st.caption("Aucune offre disponible")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Onglet Recherche ----------
with tabs[2]:
    st.subheader("🔎 Recherche par titre")
    col1, col2 = st.columns([3, 1])
    with col1:
        q = st.text_input("Titre contient…", value="Mario")
    with col2:
        st.write("")
        st.write("")
        do = st.button("Chercher", type="primary")

    if do:
        if not st.session_state.get("token"):
            st.warning("Connecte-toi d’abord.")
        else:
            data = api_get(f"/games/by-title/{requests.utils.quote(q)}")
            rows = data.get("results", [])
            if not rows:
                st.info("Aucun résultat.")
            else:
                cols_per_row = 3
                for i in range(0, len(rows), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, game in enumerate(rows[i:i+cols_per_row]):
                        with cols[j]:
                            render_game_card(game)

# ---------- Onglet Recos ----------
with tabs[3]:
    st.subheader("✨ Recommandations")
    q = st.text_input("Prompt (genre, mot-clé…)", value="RPG")
    k = st.slider("Nombre de propositions", 1, 20, 10)
    if st.button("Lancer la reco"):
        if not st.session_state.get("token"):
            st.warning("Connecte-toi d’abord.")
        else:
            payload = {"query": q, "k": k, "min_confidence": 0.0}
            r = requests.post(f"{API_URL}/recommend/ml", headers=api_headers(), json=payload, timeout=60)
            if not r.ok:
                try:
                    st.error(r.json())
                except Exception:
                    st.error(r.text)
            else:
                out = r.json()
                st.success(f"OK en {out.get('latency_ms', 0):.0f} ms — modèle {out.get('model_version', 'n/a')}")
                recs = out.get("recommendations", [])
                if not recs:
                    st.info("Aucune reco.")
                else:
                    st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
