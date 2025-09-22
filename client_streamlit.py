import os
import time
from typing import Dict, Any, Optional, List

import requests
import streamlit as st
import pandas as pd

# =============================
# Streamlit config
# =============================
st.set_page_config(page_title="Games Reco – Client", page_icon="🎮", layout="wide")

# =============================
# Helpers
# =============================

def get_api_base() -> str:
    # Priority: session > sidebar text input > env
    if "api_base_url" in st.session_state and st.session_state["api_base_url"]:
        return st.session_state["api_base_url"].rstrip("/")
    env = os.getenv("API_BASE_URL", "")
    return env.rstrip("/") if env else ""


def set_api_base(new_url: str):
    st.session_state["api_base_url"] = (new_url or "").rstrip("/")


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    tok = st.session_state.get("token")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def api_post(path: str, payload: Optional[Dict[str, Any]] = None, form: Optional[Dict[str, Any]] = None, require_auth: bool = True):
    base = get_api_base()
    if not base:
        st.error("⚠️ Renseignez l'URL de l'API dans la barre latérale.")
        return {}
    url = f"{base}{path}"
    try:
        if form is not None:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            resp = requests.post(url, headers=headers, data=form, timeout=60)
        else:
            headers = _headers() if require_auth else {"Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json=payload or {}, timeout=60)
        if resp.status_code == 401:
            st.error("Non autorisé. Connectez-vous.")
            return {}
        if not resp.ok:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            st.error(f"POST {path} → HTTP {resp.status_code}: {err}")
            return {}
        return resp.json()
    except Exception as e:
        st.error(f"POST {path} failed: {e}")
        return {}


def api_get(path: str, params: Optional[Dict[str, Any]] = None, require_auth: bool = False):
    base = get_api_base()
    if not base:
        st.error("⚠️ Renseignez l'URL de l'API dans la barre latérale.")
        return {}
    url = f"{base}{path}"
    try:
        headers = _headers() if require_auth else {"Content-Type": "application/json"}
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 401:
            st.error("Non autorisé. Connectez-vous.")
            return {}
        if not resp.ok:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            st.error(f"GET {path} → HTTP {resp.status_code}: {err}")
            return {}
        return resp.json()
    except Exception as e:
        st.error(f"GET {path} failed: {e}")
        return {}

# Preferred column names from API
PRICE_COLS = ["best_price_PC", "price", "price_eur", "price_usd", "msrp", "sale_price", "current_price"]
URL_COLS   = ["url", "store_url", "purchase_url", "buy_url", "link", "shop_url"]
STORE_COLS = ["store", "shop", "purchase_platform", "store_name", "vendor"]

COMMON_PLATFORMS = [
    "PC", "Steam", "Epic", "GOG", "PS5", "PS4", "Xbox", "Xbox One",
    "Xbox Series", "Switch", "Nintendo Switch", "Mobile", "iOS", "Android"
]

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API Base URL", value=get_api_base() or "https://", help="Ex: https://game-app-xxxx.onrender.com")
    if api_url and api_url != get_api_base():
        set_api_base(api_url)
    if st.button("Tester /healthz"):
        st.session_state["health"] = api_get("/healthz")
    if st.session_state.get("health"):
        st.caption("/healthz")
        st.json(st.session_state["health"])

    st.markdown("---")
    st.subheader("Connexion")
    if not st.session_state.get("token"):
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            ok = st.form_submit_button("Se connecter")
        if ok:
            data = api_post("/auth/token", form={"username": u, "password": p}, require_auth=False)
            if data.get("access_token"):
                st.session_state["token"] = data["access_token"]
                st.success("Connecté ✅")
    else:
        st.success("Token présent ✅")
        if st.button("Se déconnecter"):
            st.session_state.pop("token", None)
            st.experimental_rerun()

# =============================
# Main – Reco ML
# =============================
st.title("🎮 Games Recommender – Client")

with st.form("reco_form"):
    query = st.text_input("Votre requête", value="RPG open world")
    k = st.slider("k", 1, 50, 10)
    min_conf = st.slider("min_confidence", 0.0, 1.0, 0.1, step=0.01)
    plats = st.multiselect("Plateformes (filtrage serveur)", COMMON_PLATFORMS)
    min_price = st.number_input("Prix minimum (≥)", min_value=0.0, value=0.0, step=1.0)
    submit = st.form_submit_button("Recommander")

if submit:
    payload = {
        "query": query,
        "k": int(k),
        "min_confidence": float(min_conf),
        "platforms": plats or None,
        "min_price": float(min_price) if min_price else None,
    }
    t0 = time.time()
    data = api_post("/recommend/ml", payload)
    dt = (time.time() - t0) * 1000

    recs = (data or {}).get("recommendations", [])
    st.caption(f"{len(recs)} résultats • {dt:.1f} ms")

    if recs:
        df = pd.DataFrame(recs)
        # pick columns
        title_col = "title" if "title" in df.columns else None
        price_col = next((c for c in PRICE_COLS if c in df.columns), None)
        url_col   = next((c for c in URL_COLS if c in df.columns), None)
        store_col = next((c for c in STORE_COLS if c in df.columns), None)

        cols = {}
        if title_col: cols[title_col] = "Titre"
        if price_col: cols[price_col] = "Prix"
        if store_col: cols[store_col] = "Store"
        if url_col:   cols[url_col]   = "Lien"

        if cols:
            dfv = df[list(cols.keys())].rename(columns=cols).copy()
            if "Lien" in dfv.columns:
                def mk_link(row):
                    u = str(row.get("Lien") or "").strip()
                    t = str(row.get("Titre") or "").strip() or "Voir"
                    return f"[{t}]({u})" if u.startswith("http") else t
                dfv["Titre"] = dfv.apply(mk_link, axis=1)
                # garder la colonne lien visible? on peut la retirer si on veut uniquement le titre cliquable
                # dfv = dfv.drop(columns=["Lien"])
            st.dataframe(dfv, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    with st.expander("Réponse brute"):
        st.json(data if data else {})

st.markdown("---")
st.caption("Astuce: définissez API_BASE_URL dans l'environnement ou utilisez le champ 'API Base URL' de la barre latérale.")

