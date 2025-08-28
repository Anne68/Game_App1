# client_streamlit.py (Search + AI Recos seulement)
from __future__ import annotations

import os
import re
from urllib.parse import quote

import requests
import pandas as pd
import streamlit as st

# =======================
# Config & Session State
# =======================
st.set_page_config(page_title="Games UI", page_icon="🎮", layout="wide")

def _resolve_api_base() -> str:
    """
    Priorité :
    1) st.secrets["API_BASE"] (Streamlit Cloud)
    2) env API_BASE
    3) env API_URL (compat)
    4) fallback (modifie-le si besoin)
    """
    url = (
        st.secrets.get("API_BASE", None)
        or os.getenv("API_BASE")
        or os.getenv("API_URL")
        or "https://game-app1.onrender.com"   # <-- fallback prod ; mets localhost si dev
    )
    return url.rstrip("/")

API_URL = _resolve_api_base()

# Auth/session
st.session_state.setdefault("token", "")
st.session_state.setdefault("username", "")
st.session_state["login_drawn_this_run"] = False  # reset each run
st.session_state.setdefault("active_tab", "Search")

# HTTP session (garde le keep-alive & headers)
_session = requests.Session()
_BASE_HEADERS = {"Accept": "application/json"}

# ==============
# HTTP helpers
# ==============
def _headers() -> dict:
    h = dict(_BASE_HEADERS)
    if st.session_state.token:
        h["Authorization"] = f"Bearer {st.session_state.token}"
    return h

def api_get(path: str, params: dict | None = None):
    try:
        r = _session.get(API_URL + path, headers=_headers(), params=params, timeout=30)
        data = r.json() if r.content else {}
        return r.status_code, data
    except Exception as e:
        return 599, {"detail": str(e)}

def api_post_form(path: str, form: dict):
    # /token attend x-www-form-urlencoded -> data= (pas json=)
    try:
        r = _session.post(API_URL + path, headers=_headers(), data=form, timeout=30)
        data = r.json() if r.content else {}
        return r.status_code, data
    except Exception as e:
        return 599, {"detail": str(e)}

# ==========
# Auth UI
# ==========
def login_box(suffix: str = "main"):
    if st.session_state.get("login_drawn_this_run", False):
        return
    st.session_state["login_drawn_this_run"] = True

    st.subheader("Auth")
    st.caption(f"API: {API_URL}")
    with st.form(f"login_form_{suffix}", clear_on_submit=False):
        u = st.text_input("Username", key=f"login_u_{suffix}")
        p = st.text_input("Password", type="password", key=f"login_p_{suffix}")
        sub = st.form_submit_button("Login")
    if sub:
        code, payload = api_post_form("/token", {"username": u, "password": p})
        if code == 200 and "access_token" in payload:
            st.session_state.token = payload["access_token"]
            st.session_state.username = u
            st.success("Authentifié ✅")
            st.rerun()
        else:
            st.error(payload.get("detail", payload))

def handle_401(code: int, payload: dict) -> bool:
    if code == 401:
        st.warning("🔒 Session expirée. Veuillez vous reconnecter.")
        st.session_state.token = ""
        st.session_state["login_drawn_this_run"] = False
        login_box(suffix="401")
        return True
    return False

def show_not_found(code: int, payload: dict, fallback_msg: str) -> bool:
    if code == 404:
        st.info(payload.get("detail", fallback_msg))
        return True
    return False

# =================
# UI helper blocks
# =================
def nice_game_title(g: dict) -> str:
    t = g.get("title") or g.get("name") or "Untitled"
    r, m = g.get("rating"), g.get("metacritic")
    bits = []
    if r is not None:
        try:
            bits.append(f"⭐ {float(r):.1f}")
        except:
            pass
    if m is not None:
        try:
            bits.append(f"MC {int(m)}")
        except:
            pass
    return t + ((" · " + " · ".join(bits)) if bits else "")

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
                f"{p}</div>", unsafe_allow_html=True
            )

def _extract_min_price(rows: list[dict]) -> tuple[str, str] | None:
    """
    Retourne (price_text, shop_text) pour le prix minimum si identifiable.
    Sinon None.
    """
    if not rows:
        return None
    df = pd.DataFrame(rows)

    def pick(*cands):
        if df.empty:
            return None
        lower = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in df.columns:
                return c
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    c_price = pick("price", "best_price", "best_price_pc")
    c_shop  = pick("shop", "best_shop", "best_shop_pc")
    if not c_price:
        return None

    # convert to numeric when possible
    def to_num(s):
        if s is None:
            return None
        s = str(s)
        s_norm = (
            s.replace("\xa0", " ")
             .replace("€", "")
             .replace("EUR", "")
             .replace(",", ".")
        )
        m = re.search(r"-?\d+(?:\.\d+)?", s_norm)
        return float(m.group(0)) if m else None

    df["_pnum"] = df[c_price].map(to_num)
    row = df.loc[df["_pnum"].idxmin()] if df["_pnum"].notna().any() else df.iloc[0]
    price_text = f"€ {row['_pnum']:.2f}" if pd.notna(row.get("_pnum")) else str(row[c_price])
    shop_text  = str(row[c_shop]) if c_shop in df.columns else ""
    return price_text, shop_text

# ==========================
# Game card (Search results)
# ==========================
def game_card(g: dict, idx: int):
    """
    Carte jeu qui AFFICHE directement :
      - prix mini via /games/title/{title}/prices ou /games/{id}/prices
      - plateformes via /games/by-title/{title}/platforms ou /games/{id}/platforms
    """
    with st.container():
        st.markdown(f"### {nice_game_title(g)}")
        c1, c2 = st.columns([2, 2])

        # --- Genres
        with c1:
            genres = g.get("genres", "")
            if genres:
                st.caption(genres)

        # Résolution de l'identité
        title = (g.get("title") or g.get("name") or "").strip()
        gid   = g.get("id") or g.get("game_id_rawg")

        # --- Prix (mini) ---
        prices_endpoint = None
        if title:
            prices_endpoint = f"/games/title/{quote(title)}/prices"
        elif gid:
            prices_endpoint = f"/games/{gid}/prices"

        min_price_text = None
        min_price_shop = None
        if prices_endpoint:
            code, payload = api_get(prices_endpoint)
            if not handle_401(code, payload):
                if code == 200:
                    rows = payload.get("prices", [])
                    mp = _extract_min_price(rows)
                    if mp:
                        min_price_text, min_price_shop = mp

        # --- Plateformes ---
        plats_endpoint = None
        if title:
            plats_endpoint = f"/games/by-title/{quote(title)}/platforms"
        elif gid:
            plats_endpoint = f"/games/{gid}/platforms"

        plats = []
        if plats_endpoint:
            code, payload = api_get(plats_endpoint)
            if not handle_401(code, payload):
                if code == 200:
                    plats = payload.get("platforms") or payload.get("platform_ids") or []

        # --- Rendu : prix mini + pastilles plateformes
        with c2:
            if min_price_text:
                if min_price_shop:
                    st.markdown(f"**💶 From {min_price_text}**  ·  *{min_price_shop}*")
                else:
                    st.markdown(f"**💶 From {min_price_text}**")
            else:
                st.caption("💶 Prix indisponibles")

            platform_pills(plats)

# =========
# Header
# =========
with st.container():
    a, b = st.columns([1, 1])
    with a:
        st.title("🎮 Games UI")
    with b:
        with st.expander("Compte", expanded=True):
            if st.session_state.token:
                st.write(f"Connecté en tant que **{st.session_state.username}**")
                if st.button("Se déconnecter"):
                    st.session_state.token = ""
                    st.session_state.username = ""
                    st.rerun()
            else:
                st.caption("Non connecté")

# Exige une session authentifiée
if not st.session_state.token:
    login_box(suffix="main")
    st.stop()

# =========================
# Navigation (2 sections)
# =========================
nav_order = ["Search", "AI Recos"]
nav_idx = nav_order.index(st.session_state.active_tab) if st.session_state.active_tab in nav_order else 0
active = st.radio("Navigation", nav_order, index=nav_idx, horizontal=True)
st.session_state.active_tab = active

# =========
# SEARCH
# =========
if st.session_state.active_tab == "Search":
    st.markdown("### Search games by title ↪")
    q = st.text_input("Title contains…", key="q_search", placeholder="ex: mario, battle royal…")
    if st.button("Search", key="btn_search"):
        if not q.strip():
            st.warning("Saisissez un texte de recherche.")
        else:
            # L’API accepte /games/by-title/{title} ET /games/title/{title}
            code, payload = api_get(f"/games/by-title/{quote(q.strip())}")
            if handle_401(code, payload):
                st.stop()
            if code == 200:
                games = payload.get("games", [])
                if not games:
                    st.info("Aucun résultat.")
                else:
                    for i, g in enumerate(games):
                        game_card(g, i)
            elif show_not_found(code, payload, "Aucun résultat."):
                pass
            else:
                st.error(payload.get("detail", payload))

# =========
# AI RECOS
# =========
else:
    st.markdown("### Recommendations")
    s1, s2 = st.tabs(["By title", "By genre"])

    # --- by title ---
    with s1:
        ti = st.text_input("Title", key="q_reco_title", placeholder="ex: The Witcher 3")
        k1 = st.number_input("Top-N", value=5, min_value=1, max_value=20, step=1, key="k_title")
        if st.button("Get recommendations (title)", key="btn_recos_title"):
            if not ti.strip():
                st.warning("Saisissez un titre.")
            else:
                code, payload = api_get(f"/recommend/by-title/{quote(ti.strip())}", params={"k": int(k1)})
                if handle_401(code, payload):
                    st.stop()
                if code == 200:
                    recs = payload.get("recommendations", [])
                    st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
                elif show_not_found(code, payload, "Aucune reco."):
                    pass
                else:
                    st.error(payload.get("detail", payload))

    # --- by genre ---
    with s2:
        ge = st.text_input("Genre", key="q_reco_genre", placeholder="ex: Action, RPG…")
        k2 = st.number_input("Top-N ", value=5, min_value=1, max_value=20, step=1, key="k_genre")
        if st.button("Get recommendations (genre)", key="btn_recos_genre"):
            if not ge.strip():
                st.warning("Saisissez un genre.")
            else:
                code, payload = api_get(f"/recommend/by-genre/{quote(ge.strip())}", params={"k": int(k2)})
                if handle_401(code, payload):
                    st.stop()
                if code == 200:
                    recs = payload.get("recommendations", [])
                    st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
                elif show_not_found(code, payload, "Aucune reco."):
                    pass
                else:
                    st.error(payload.get("detail", payload))
