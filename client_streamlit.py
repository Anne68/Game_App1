# client_streamlit.py — UI de recherche + recommandations + inscription (corrigé)
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

API_URL_DEFAULT = "https://game-app-y8be.onrender.com"  # Render
API_URL = (os.getenv("API_URL") or API_URL_DEFAULT).rstrip("/")

if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None


# =======================
# Helpers HTTP
# =======================
def _headers(with_auth: bool = True) -> dict:
    h = {"Accept": "application/json"}
    if with_auth and st.session_state.get("token"):
        h["Authorization"] = f"Bearer {st.session_state['token']}"
    return h


def api_get(path: str, params: dict | None = None, with_auth: bool = True):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, headers=_headers(with_auth), timeout=20)
        # On essaye de parser JSON si présent
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status_code, r.json()
        return r.status_code, {"detail": r.text}
    except Exception as e:
        return 0, {"detail": str(e)}


def api_post_form(path: str, form: dict, with_auth: bool = False):
    """POST x-www-form-urlencoded (register/token)."""
    try:
        r = requests.post(f"{API_URL}{path}", data=form, headers=_headers(with_auth), timeout=20)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status_code, r.json()
        return r.status_code, {"detail": r.text}
    except Exception as e:
        return 0, {"detail": str(e)}


def api_post_json(path: str, payload: dict, with_auth: bool = True):
    """POST JSON (recommend/ml)."""
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, headers=_headers(with_auth), timeout=30)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status_code, r.json()
        return r.status_code, {"detail": r.text}
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
api_input = st.sidebar.text_input("API URL", value=API_URL, help="Peut aussi être défini via la variable d'env API_URL.")
if api_input.strip():
    API_URL = api_input.strip().rstrip("/")

if st.sidebar.button("Test API"):
    code, payload = api_get("/healthz", with_auth=False)  # public
    if code == 200:
        st.sidebar.success(f"/healthz → {code} {payload}")
    else:
        st.sidebar.error(f"Échec: {code} {payload}")


# ========== Auth UI ==========
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
        p2 = st.text_input("Confirmer le mot de passe", type="password", key=f"signup_p2_{suffix}")
        sub = st.form_submit_button("S'inscrire")
    if sub:
        if not u.strip() or not p:
            st.warning("Veuillez renseigner un username et un mot de passe.")
            return
        if p != p2:
            st.error("Les mots de passe ne correspondent pas.")
            return
        # 1) Register
        code, payload = api_post_form("/register", {"username": u.strip(), "password": p}, with_auth=False)
        if code == 200 and payload.get("ok"):
            st.success("Compte créé avec succès ✅")
            # 2) Auto-login
            code2, payload2 = api_post_form("/token", {"username": u.strip(), "password": p}, with_auth=False)
            if code2 == 200 and "access_token" in payload2:
                st.session_state["token"] = payload2["access_token"]
                st.session_state["username"] = u.strip()
                st.success("Connecté automatiquement 🎉")
            else:
                st.info("Compte créé. Vous pouvez maintenant vous connecter.")
        else:
            st.error(payload.get("detail", payload))


# =======================
# UI Components
# =======================
def platform_pills(items):
    if not items:
        st.caption("🎮 Aucune plateforme trouvée")
        return
    items = list(items)
    cols = st.columns(min(6, len(items)) or 1)
    for i, p in enumerate(items):
        with cols[i % len(cols)]:
            st.markdown(
                "<div style='padding:6px 10px;border-radius:999px;background:#f0f2f6;display:inline-block;margin:4px'>"
                f"{p}</div>",
                unsafe_allow_html=True,
            )


# ========== MAIN TABS ==========
tab_signup, tab_auth, tab_search, tab_reco = st.tabs(["🆕 Inscription", "🔑 Auth", "🔎 Recherche", "✨ Recos"])

with tab_signup:
    signup_box()

with tab_auth:
    login_box()

with tab_search:
    st.subheader("🔎 Recherche par titre (catalogue)")
    q = st.text_input("Titre contient…", key="search_title")
    if st.button("Chercher", key="btn_search_title"):
        if not q.strip():
            st.warning("Saisissez un titre.")
        else:
            code, payload = api_get(f"/games/by-title/{quote(q.strip())}")  # protégé → header Bearer envoyé
            if handle_401(code, payload):
                st.stop()
            if code == 200:
                df = pd.DataFrame(payload.get("results", []))
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucun résultat.")
            elif show_not_found(code, payload, "Aucun résultat."):
                pass
            else:
                st.error(payload.get("detail", payload))

with tab_reco:
    st.subheader("✨ Recommandations")

    col1, col2 = st.columns(2)

    # ---------- Par titre (via /games/by-title -> /recommend/by-game/{id})
    with col1:
        st.caption("Par titre → trouve le jeu puis recommande des jeux similaires")
        ti = st.text_input("Titre", key="reco_title")
        k1 = st.number_input("k", min_value=1, max_value=50, value=10, step=1, key="k_title")
        if st.button("Get recommendations (title)", key="btn_recos_title"):
            if not ti.strip():
                st.warning("Saisissez un titre.")
            else:
                # 1) chercher le(s) jeu(x) correspondant(s)
                code, payload = api_get(f"/games/by-title/{quote(ti.strip())}")
                if handle_401(code, payload):
                    st.stop()
                if code != 200:
                    st.error(payload.get("detail", payload))
                else:
                    items = payload.get("results", [])
                    if not items:
                        st.info("Aucun jeu trouvé pour ce titre.")
                    else:
                        # on prend le 1er résultat (ou tu peux ajouter un selectbox si tu veux)
                        game_id = items[0]["id"]
                        code2, payload2 = api_get(f"/recommend/by-game/{int(game_id)}", params={"k": int(k1)})
                        if handle_401(code2, payload2):
                            st.stop()
                        if code2 == 200:
                            st.dataframe(pd.DataFrame(payload2.get("recommendations", [])),
                                         use_container_width=True, hide_index=True)
                        else:
                            st.error(payload2.get("detail", payload2))

    # ---------- Par genre (via /recommend/ml en JSON)
    with col2:
        st.caption("Par genre → utilise le modèle ML (POST JSON)")
        ge = st.text_input("Genre (ex: Action)", key="reco_genre")
        k2 = st.number_input("k ", min_value=1, max_value=50, value=10, step=1, key="k_genre")
        if st.button("Get recommendations (genre)", key="btn_recos_genre"):
            if not ge.strip():
                st.warning("Saisissez un genre.")
            else:
                code, payload = api_post_json(
                    "/recommend/ml",
                    {"query": ge.strip(), "k": int(k2), "min_confidence": 0.1},
                    with_auth=True,
                )
                if handle_401(code, payload):
                    st.stop()
                if code == 200:
                    st.dataframe(pd.DataFrame(payload.get("recommendations", [])),
                                 use_container_width=True, hide_index=True)
                else:
                    st.error(payload.get("detail", payload))
