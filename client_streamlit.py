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
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
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


def api_post(path: str, payload: Optional[Dict[str, Any]] = None, form: Optional[Dict[str, Any]] = None, require_auth: bool = True):
    url = f"{API_BASE_URL}{path}"
    if form is not None:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        resp = requests.post(url, headers=headers, data=form, timeout=30)
    else:
        headers = _headers() if require_auth else {"Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json=payload or {}, timeout=60)
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        st.error(f"POST {path} -> HTTP {resp.status_code}: {detail}")
        return {}
    return resp.json()


def show_header():
    st.markdown("<h2>🎮 Games API ML – Streamlit</h2>", unsafe_allow_html=True)
    st.caption("Interface simple pour tester l'API de recommandations.")


def sidebar_auth():
    with st.sidebar:
        st.subheader("Connexion")
        if not st.session_state.get("logged", False):
            with st.form("login_form"):
                username = st.text_input("Nom d'utilisateur", value=DEMO_USERNAME)
                password = st.text_input("Mot de passe", type="password", value=DEMO_PASSWORD)
                login_btn = st.form_submit_button("Se connecter")
            if login_btn:
                data = api_post("/auth/token", form={"username": username, "password": password}, require_auth=False)
                if data.get("access_token"):
                    st.session_state.token = data["access_token"]
                    st.session_state.logged = True
                    st.session_state.username = username
                    st.success("Connecté ✅")
        else:
            st.success(f"Connecté en tant que {st.session_state.get('username')}")
            if st.button("Se déconnecter"):
                st.session_state.clear()
                st.experimental_rerun()

# =============== Pages ===============

def page_recommendations():
    st.subheader("🔎 Recommandations ML")
    if not st.session_state.get("logged"):
        st.warning("Veuillez vous connecter.")
        return

    query = st.text_input("Votre requête (ex: RPG open world)", value="RPG open world")
    k = st.slider("Nombre de résultats (k)", 1, 20, 5)
    if st.button("Recommander"):
        payload = {"query": query, "k": k, "min_confidence": 0.1}
        data = api_post("/recommend/ml", payload)
        recs = data.get("recommendations", [])

        if not recs:
            st.info("Aucune recommandation.")
            return

        # Construire DataFrame simplifié (titre, prix, url vers plateforme si dispo)
        df = pd.DataFrame(recs)

        # Détection des colonnes possibles
        url_col = None
        for c in ["url", "link", "store_url", "purchase_url"]:
            if c in df.columns:
                url_col = c
                break

        price_col = None
        for c in ["price", "price_eur", "price_usd", "msrp"]:
            if c in df.columns:
                price_col = c
                break

        # Colonnes finales
        cols = {"title": "Titre"}
        if price_col:
            cols[price_col] = "Prix"
        if url_col:
            cols[url_col] = "Lien"

        if not cols:
            st.dataframe(df)
        else:
            df = df[list(cols.keys())].rename(columns=cols)
            if url_col:
                # Rendre les liens cliquables
                df["Lien"] = df["Lien"].apply(lambda x: f"[Acheter ici]({x})" if isinstance(x, str) and x.startswith("http") else "")
                st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
            else:
                st.dataframe(df)

        with st.expander("Réponse brute"):
            st.json(data)


def main():
    show_header()
    sidebar_auth()
    page_recommendations()

if __name__ == "__main__":
    main()
