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

COMMON_PLATFORMS = [
    "PC", "Steam", "Epic", "GOG", "PS5", "PS4", "Xbox", "Xbox One",
    "Xbox Series", "Switch", "Nintendo Switch", "Mobile", "iOS", "Android"
]

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
    try:
        if form is not None:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            resp = requests.post(url, headers=headers, data=form, timeout=30)
        else:
            headers = _headers() if require_auth else {"Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json=payload or {}, timeout=60)
        if resp.status_code == 401:
            st.error("Non autorisé : connectez-vous.")
            return {}
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            st.error(f"POST {path} -> HTTP {resp.status_code}: {detail}")
            return {}
        return resp.json()
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return {}


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


def _detect_url_col(df: pd.DataFrame) -> Optional[str]:
    url_candidates = ["url", "store_url", "purchase_url", "buy_url", "link", "shop_url", "steam_url", "epic_url", "gog_url"]
    for c in url_candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if "url" in lc or "link" in lc:
            return c
    return None


def page_recommendations():
    st.subheader("🔎 Recommandations ML")
    if not st.session_state.get("logged"):
        st.warning("Veuillez vous connecter.")
        return

    with st.form("reco_form"):
        query = st.text_input("Votre requête (ex: RPG open world)", value="RPG open world")
        k = st.slider("Nombre de résultats (k)", 1, 50, 10)
        plats = st.multiselect("Filtrer par plateformes (côté API)", COMMON_PLATFORMS, default=[])
        min_price = st.number_input("Prix minimum (≥)", min_value=0.0, step=1.0, value=0.0)
        submit = st.form_submit_button("Recommander")

    if submit:
        payload = {
            "query": query,
            "k": int(k),
            "min_confidence": 0.1,
            "platforms": plats or None,
            "min_price": float(min_price) if min_price else None
        }
        data = api_post("/recommend/ml", payload)
        recs = data.get("recommendations", [])
        if not recs:
            st.info("Aucune recommandation.")
            with st.expander("Réponse brute"):
                st.json(data)
            return

        df = pd.DataFrame(recs)

        title_col = "title" if "title" in df.columns else None
        price_col = "best_price_PC" if "best_price_PC" in df.columns else ("price" if "price" in df.columns else None)
        store_col = "store" if "store" in df.columns else None
        url_col = _detect_url_col(df)

        display_cols = []
        rename_map = {}
        if title_col:
            display_cols.append(title_col); rename_map[title_col] = "Titre"
        if price_col:
            display_cols.append(price_col); rename_map[price_col] = "Prix"
        if store_col:
            display_cols.append(store_col); rename_map[store_col] = "Store"
        if url_col:
            display_cols.append(url_col); rename_map[url_col] = "Lien"

        if display_cols:
            df_disp = df[display_cols].rename(columns=rename_map).copy()
            if "Lien" in df_disp.columns:
                def mk_link(row):
                    u = str(row.get("Lien") or "").strip()
                    t = str(row.get("Titre") or "").strip() or "Voir"
                    return f"[{t}]({u})" if u.startswith("http") else t
                df_disp["Titre"] = df_disp.apply(mk_link, axis=1)
            st.dataframe(df_disp, use_container_width=True)
        else:
            base_cols = [c for c in ["confidence","id","title","genres","rating","metacritic"] if c in df.columns]
            st.dataframe(df[base_cols].rename(columns={"id":"game_id","title":"Titre"}), use_container_width=True)
            st.info("Astuce: ajoutez `best_price_PC`/`store`/`url` côté API pour afficher Titre · Prix · Store · Lien.")

        with st.expander("Réponse brute"):
            st.json(data)


def main():
    show_header()
    sidebar_auth()
    page_recommendations()

if __name__ == "__main__":
    main()
