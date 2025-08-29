# client_streamlit.py
import os
import time
import requests
import pandas as pd
import streamlit as st

# -------------------- Config --------------------
st.set_page_config(page_title="🎮 Games UI", page_icon="🎮", layout="wide")
API_URL = os.getenv("ST_API_URL", "https://game-app-y8be.onrender.com").rstrip("/")

# -------------------- Helpers HTTP --------------------
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

def api_post(path, data=None):
    r = requests.post(f"{API_URL}{path}", headers=api_headers(), data=data or {}, timeout=30)
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

# -------------------- UI --------------------
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

# ----- Onglet Inscription (raccourci) -----
with tabs[0]:
    st.caption("Tu peux aussi créer un compte dans la barre latérale.")

# ----- Onglet Auth (raccourci) -----
with tabs[1]:
    st.caption("Connecte-toi via la barre latérale pour débloquer les appels API protégés.")

# ----- Onglet Recherche -----
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
            df = pd.DataFrame(rows)

            if df.empty:
                st.info("Aucun résultat.")
            else:
                # Garder uniquement les colonnes souhaitées
                desired = ["title", "best_price", "rating", "metacritic", "site_url"]
                for c in desired:
                    if c not in df.columns:
                        df[c] = None
                df_view = df[desired].copy()
                df_view.rename(columns={
                    "title": "Titre",
                    "best_price": "Prix",
                    "rating": "Note",
                    "metacritic": "Metacritic",
                    "site_url": "Lien"
                }, inplace=True)

                st.dataframe(
                    df_view,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Titre": st.column_config.TextColumn("Titre"),
                        "Prix": st.column_config.TextColumn("Prix"),
                        "Note": st.column_config.NumberColumn("Note", format="%.2f"),
                        "Metacritic": st.column_config.NumberColumn("Metacritic", format="%d"),
                        "Lien": st.column_config.LinkColumn("Lien offre", display_text="Voir"),
                    },
                )
                st.caption("Astuce : clique « Voir » pour ouvrir l’offre du meilleur prix (si disponible).")

# ----- Onglet Recos -----
with tabs[3]:
    st.subheader("✨ Recommandations")
    q = st.text_input("Prompt (genre, mot-clé…)", value="RPG")
    k = st.slider("Nombre de propositions", 1, 20, 10)
    if st.button("Lancer la reco"):
        if not st.session_state.get("token"):
            st.warning("Connecte-toi d’abord.")
        else:
            payload = {"query": q, "k": k, "min_confidence": 0.0}
            t0 = time.time()
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
