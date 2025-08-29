# client_streamlit.py — UI Streamlit <-> API FastAPI
from __future__ import annotations

import os
from urllib.parse import quote

import requests
import pandas as pd
import streamlit as st

# =======================
# Config & Session State
# =======================
st.set_page_config(page_title="Games UI", page_icon="🎮", layout="wide")

API_URL_DEFAULT = "https://game-app-y8be.onrender.com"  # ton service Render
API_URL = (os.getenv("API_URL") or API_URL_DEFAULT).rstrip("/")

if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None


# =======================
# HTTP helpers
# =======================
def _headers(with_auth: bool = True) -> dict:
    h = {"Accept": "application/json"}
    if with_auth and st.session_state["token"]:
        h["Authorization"] = f"Bearer {st.session_state['token']}"
    return h

def _parse_json(resp: requests.Response) -> dict:
    ct = resp.headers.get("content-type", "")
    return resp.json() if ct.startswith("application/json") else {"detail": resp.text}

def api_get(path: str, params: dict | None = None, with_auth: bool = True):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, headers=_headers(with_auth), timeout=25)
        return r.status_code, _parse_json(r)
    except Exception as e:
        return 0, {"detail": str(e)}

def api_post_form(path: str, data: dict, with_auth: bool = False):
    try:
        r = requests.post(f"{API_URL}{path}", data=data, headers=_headers(with_auth), timeout=25)
        return r.status_code, _parse_json(r)
    except Exception as e:
        return 0, {"detail": str(e)}

def api_post_json(path: str, json: dict, with_auth: bool = True):
    try:
        r = requests.post(f"{API_URL}{path}", json=json, headers=_headers(with_auth), timeout=60)
        return r.status_code, _parse_json(r)
    except Exception as e:
        return 0, {"detail": str(e)}

def handle_401(code: int, payload: dict) -> bool:
    if code == 401:
        st.error("⛔ Session expirée / non autorisée. Merci de vous reconnecter.")
        st.session_state["token"] = None
        st.session_state["username"] = None
        return True
    return False

def show_not_found(code: int, payload: dict, msg: str) -> bool:
    if code == 404:
        st.info(msg)
        return True
    return False


# =======================
# Header + Sidebar
# =======================
left, right = st.columns([1, 1], vertical_alignment="center")
with left:
    st.title("🎮 Games UI")
with right:
    status = "✅ Connecté" if st.session_state["token"] else "🔒 Non connecté"
    st.info(status)

st.sidebar.subheader("Backend")
api_input = st.sidebar.text_input(
    "API URL",
    value=API_URL,
    help="Ex: https://game-app-y8be.onrender.com (peut aussi être défini via la variable d'env API_URL).",
)
if api_input.strip():
    API_URL = api_input.strip().rstrip("/")

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Test /healthz"):
    code, payload = api_get("/healthz", with_auth=False)
    (st.sidebar.success if code == 200 else st.sidebar.error)(f"{code} → {payload}")

if col_b.button("Métriques modèle"):
    if not st.session_state["token"]:
        st.sidebar.warning("Connecte-toi d'abord.")
    else:
        c, p = api_get("/model/metrics")
        if handle_401(c, p): ...
        elif c == 200:
            st.sidebar.json(p)
        else:
            st.sidebar.error(p.get("detail", p))

with st.sidebar.expander("⚙️ Entraîner le modèle", expanded=False):
    force = st.checkbox("Forcer ré-entraînement", value=False)
    if st.button("Train now"):
        if not st.session_state["token"]:
            st.warning("Connecte-toi d'abord.")
        else:
            payload = {"force_retrain": bool(force)}
            c, p = api_post_json("/model/train", payload, with_auth=True)
            if handle_401(c, p): ...
            elif c == 200:
                st.success(f"✅ Entraîné — version {p.get('version')}, {round(p.get('duration', 0), 3)} s")
            else:
                st.error(p.get("detail", p))


# =======================
# Auth UI
# =======================
def login_box(suffix: str = "main"):
    st.subheader("🔑 Auth")
    with st.form(f"login_form_{suffix}", clear_on_submit=False):
        u = st.text_input("Username", key=f"login_u_{suffix}")
        p = st.text_input("Password", type="password", key=f"login_p_{suffix}")
        sub = st.form_submit_button("Login")
    if sub:
        code, payload = api_post_form("/token", {"username": u, "password": p}, with_auth=False)
        if code == 200 and "access_token" in payload:
            st.session_state["token"] = payload["access_token"]
            st.session_state["username"] = u
            st.success("Connecté !")
        else:
            st.error(payload.get("detail", payload))

def signup_box(suffix: str = "main"):
    st.subheader("🆕 Créer un compte")
    with st.form(f"signup_form_{suffix}", clear_on_submit=False):
        u = st.text_input("Username", key=f"signup_u_{suffix}")
        p = st.text_input("Password", type="password", key=f"signup_p_{suffix}")
        p2 = st.text_input("Confirmer mot de passe", type="password", key=f"signup_p2_{suffix}")
        sub = st.form_submit_button("S'inscrire")
    if sub:
        if not u.strip() or not p:
            st.warning("Renseigne un username + mot de passe.")
            return
        if p != p2:
            st.error("Les mots de passe ne correspondent pas.")
            return
        c1, r1 = api_post_form("/register", {"username": u.strip(), "password": p}, with_auth=False)
        if c1 == 200 and r1.get("ok"):
            st.success("Compte créé ✅")
            c2, r2 = api_post_form("/token", {"username": u.strip(), "password": p}, with_auth=False)
            if c2 == 200 and "access_token" in r2:
                st.session_state["token"] = r2["access_token"]
                st.session_state["username"] = u.strip()
                st.success("Connecté automatiquement 🎉")
            else:
                st.info("Compte créé. Tu peux te connecter.")
        else:
            st.error(r1.get("detail", r1))


# =======================
# UI Sections
# =======================
tab_signup, tab_auth, tab_search, tab_reco = st.tabs(
    ["🆕 Inscription", "🔑 Auth", "🔎 Recherche", "✨ Recos"]
)

with tab_signup:
    signup_box()

with tab_auth:
    login_box()

with tab_search:
    st.subheader("🔎 Recherche par titre")
    q = st.text_input("Titre contient…", key="search_title")
    if st.button("Chercher", key="btn_search_title"):
        if not q.strip():
            st.warning("Saisis un titre.")
        else:
            code, payload = api_get(f"/games/by-title/{quote(q.strip())}")
            if handle_401(code, payload):
                st.stop()
            if code == 200:
                df = pd.DataFrame(payload.get("results", []))
                st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("Aucun résultat.")
            elif show_not_found(code, payload, "Aucun résultat."):
                pass
            else:
                st.error(payload.get("detail", payload))

with tab_reco:
    st.subheader("✨ Recommandations")

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Par **titre** (DB Alwaysdata)")
        ti = st.text_input("Titre", key="reco_title")
        k1 = st.number_input("k", min_value=1, max_value=50, value=10, step=1, key="k_title")
        if st.button("Get recommendations (title)", key="btn_recos_title"):
            if not ti.strip():
                st.warning("Saisis un titre.")
            else:
                code, payload = api_get(f"/recommend/by-title/{quote(ti.strip())}", params={"k": int(k1)})
                if handle_401(code, payload):
                    st.stop()
                if code == 200:
                    df = pd.DataFrame(payload.get("recommendations", []))
                    st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("Aucune reco.")
                elif show_not_found(code, payload, "Aucune reco."):
                    pass
                else:
                    st.error(payload.get("detail", payload))

    with col2:
        st.caption("Par **genre** (ex: Action, RPG)")
        ge = st.text_input("Genre", key="reco_genre")
        k2 = st.number_input("k ", min_value=1, max_value=50, value=10, step=1, key="k_genre")
        if st.button("Get recommendations (genre)", key="btn_recos_genre"):
            if not ge.strip():
                st.warning("Saisis un genre.")
            else:
                code, payload = api_get(f"/recommend/by-genre/{quote(ge.strip())}", params={"k": int(k2)})
                if handle_401(code, payload):
                    st.stop()
                if code == 200:
                    df = pd.DataFrame(payload.get("recommendations", []))
                    st.dataframe(df, use_container_width=True, hide_index=True) if not df.empty else st.info("Aucune reco.")
                elif show_not_found(code, payload, "Aucune reco."):
                    pass
                else:
                    st.error(payload.get("detail", payload))
