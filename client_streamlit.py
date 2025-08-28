# client_streamlit.py (Search + AI Recos seulement)
from __future__ import annotations

import os
import re
from urllib.parse import quote, urlparse

import requests
import pandas as pd
import streamlit as st

# =======================
# Config & Session State
# =======================
st.set_page_config(page_title="Games UI", page_icon="🎮", layout="wide")

# ---- Defaults / candidates (tu peux en ajouter) ----
DEFAULT_API_CANDIDATES = [
    # Ajoute ici tes services Render valides
    "https://game-app-y8be.onrender.com",
    "https://game-app1.onrender.com",
]

def _secrets_get(key: str, default: str = "") -> str:
    try:
        val = st.secrets.get(key, default)
        return str(val) if val is not None else default
    except Exception:
        return default

def _resolve_api_base() -> str:
    # 1) session override
    if st.session_state.get("api_base"):
        return st.session_state["api_base"].rstrip("/")
    # 2) secrets/env
    base = (
        _secrets_get("API_BASE")
        or os.getenv("API_BASE")
        or os.getenv("API_URL")
        or ""
    ).strip()
    if base:
        return base.rstrip("/")
    # 3) fallback candidats
    return DEFAULT_API_CANDIDATES[0]

def _resolve_api_prefix() -> str:
    if st.session_state.get("api_prefix") is not None:
        return st.session_state["api_prefix"]
    pref = (_secrets_get("API_PREFIX") or os.getenv("API_PREFIX", "") or "").strip()
    if pref in ["/", "."]:
        return ""
    return ("/" + pref.strip("/")) if pref else ""

def _resolve_token_path_override() -> str:
    if st.session_state.get("token_path_override") is not None:
        return st.session_state["token_path_override"]
    p = (_secrets_get("TOKEN_PATH") or os.getenv("TOKEN_PATH", "") or "").strip()
    if not p:
        return ""
    return p if p.startswith("/") else ("/" + p)

# Initial session
st.session_state.setdefault("token", "")
st.session_state.setdefault("username", "")
st.session_state.setdefault("active_tab", "Search")
st.session_state.setdefault("login_drawn_this_run", False)

# API target in session (can be edited from UI)
st.session_state.setdefault("api_base", _resolve_api_base())
st.session_state.setdefault("api_prefix", _resolve_api_prefix())
st.session_state.setdefault("token_path_override", _resolve_token_path_override())
st.session_state.setdefault("discovered_token_path", "")

def API_BASE() -> str:
    return (st.session_state.get("api_base") or _resolve_api_base()).rstrip("/")

def API_PREFIX() -> str:
    pref = st.session_state.get("api_prefix") or ""
    if pref in ["/", "."]:
        return ""
    return ("/" + pref.strip("/")) if pref else ""

def TOKEN_PATH_OVERRIDE() -> str:
    return st.session_state.get("token_path_override") or ""

def build_url(path: str) -> str:
    path = path if path.startswith("/") else ("/" + path)
    return f"{API_BASE()}{API_PREFIX()}{path}"

# HTTP session
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

def _parse_json_or_text(r):
    try:
        return r.json()
    except Exception:
        return {"detail": (r.text or "").strip() or f"HTTP {r.status_code}"}

def api_get(path: str, params: dict | None = None):
    url = build_url(path)
    try:
        r = _session.get(url, headers=_headers(), params=params, timeout=30)
        return r.status_code, _parse_json_or_text(r)
    except Exception as e:
        return 599, {"detail": f"{e} (GET {url})"}

def api_post_form(path: str, form: dict):
    url = build_url(path)
    try:
        r = _session.post(
            url,
            headers={**_headers(), "Content-Type": "application/x-www-form-urlencoded"},
            data=form,
            timeout=30,
        )
        return r.status_code, _parse_json_or_text(r)
    except Exception as e:
        return 599, {"detail": f"{e} (POST {url})"}

# ==============
# Discovery
# ==============
def _to_relative(p: str) -> str:
    if not p:
        return "/token"
    if p.startswith("http://") or p.startswith("https://"):
        return urlparse(p).path or "/token"
    return p if p.startswith("/") else ("/" + p)

@st.cache_data(show_spinner=False)
def discover_token_path_for(base: str, pref: str) -> str:
    """
    Essaie de lire le tokenUrl de l'OpenAPI pour une base donnée.
    """
    candidate_specs = [
        f"{base.rstrip('/')}{pref}/openapi.json",
        f"{base.rstrip('/')}/openapi.json",
    ]
    for spec_url in candidate_specs:
        try:
            r = _session.get(spec_url, headers={"Accept": "application/json"}, timeout=10)
            if r.status_code != 200:
                continue
            spec = r.json()
            comps = (spec or {}).get("components", {})
            schemes = comps.get("securitySchemes", {})
            for s in schemes.values():
                flows = (s or {}).get("flows", {})
                pwd = flows.get("password") or {}
                token_url = _to_relative(pwd.get("tokenUrl"))
                if token_url:
                    return token_url
        except Exception:
            pass
    return "/token"

# ==============
# Auth helpers
# ==============
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

# ==========
# Auth UI
# ==========
def login_box(suffix: str = "main"):
    if st.session_state.get("login_drawn_this_run", False):
        return
    st.session_state["login_drawn_this_run"] = True

    st.subheader("Auth")
    st.caption(f"API: {API_BASE()}{API_PREFIX() or ''}")
    with st.form(f"login_form_{suffix}", clear_on_submit=False):
        u = st.text_input("Username", key=f"login_u_{suffix}")
        p = st.text_input("Password", type="password", key=f"login_p_{suffix}")
        sub = st.form_submit_button("Login")

    if not sub:
        return

    # Token path discovery (or override)
    token_path = TOKEN_PATH_OVERRIDE() or st.session_state.get("discovered_token_path")
    if not token_path:
        token_path = discover_token_path_for(API_BASE(), API_PREFIX())
        st.session_state["discovered_token_path"] = token_path

    candidates: list[str] = []
    if TOKEN_PATH_OVERRIDE():
        candidates.append(TOKEN_PATH_OVERRIDE())
    candidates += [
        token_path or "/token",
        "/token",
        "/api/token",
        "/auth/token",
        "/login",
        "/login/access-token",
    ]

    seen = set()
    tried_msgs = []
    for path in candidates:
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        if not path.startswith("/"):
            path = "/" + path
        code, payload = api_post_form(path, {"username": u, "password": p})
        tried_msgs.append(f"{build_url(path)} → {code}")
        if code == 200 and "access_token" in payload:
            st.session_state.token = payload["access_token"]
            st.session_state.username = u
            st.success(f"Authentifié ✅ (via {path})")
            st.rerun()
            return
        if code == 404:
            continue
        st.error(f"{payload.get('detail', payload)} (status {code}, URL {build_url(path)})")
        return

    st.error(
        "Endpoint token introuvable. "
        + " — ".join(tried_msgs)
        + "\n➡ Vérifie API_BASE/API_PREFIX ou définis TOKEN_PATH (ex: '/auth/token')."
    )

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
    with st.container():
        st.markdown(f"### {nice_game_title(g)}")
        c1, c2 = st.columns([2, 2])

        # Genres
        with c1:
            genres = g.get("genres", "")
            if genres:
                st.caption(genres)

        # Identité
        title = (g.get("title") or g.get("name") or "").strip()
        gid   = g.get("id") or g.get("game_id_rawg")

        # Prix
        prices_endpoint = f"/games/title/{quote(title)}/prices" if title else (f"/games/{gid}/prices" if gid else None)
        min_price_text = min_price_shop = None
        if prices_endpoint:
            code, payload = api_get(prices_endpoint)
            if not handle_401(code, payload) and code == 200:
                rows = payload.get("prices", [])
                mp = _extract_min_price(rows)
                if mp:
                    min_price_text, min_price_shop = mp

        # Plateformes
        plats_endpoint = f"/games/by-title/{quote(title)}/platforms" if title else (f"/games/{gid}/platforms" if gid else None)
        plats = []
        if plats_endpoint:
            code, payload = api_get(plats_endpoint)
            if not handle_401(code, payload) and code == 200:
                plats = payload.get("platforms") or payload.get("platform_ids") or []

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
# Header + API config
# =========
with st.container():
    a, b = st.columns([1, 1])
    with a:
        st.title("🎮 Games UI")
    with b:
        with st.expander("⚙️ API config", expanded=False):
            api_base_in = st.text_input("API_BASE", API_BASE(), placeholder="https://<ton-service>.onrender.com")
            api_prefix_in = st.text_input("API_PREFIX (optionnel)", API_PREFIX())
            token_path_in = st.text_input("TOKEN_PATH forcé (optionnel)", TOKEN_PATH_OVERRIDE())

            if st.button("💾 Enregistrer"):
    st.session_state["api_base"] = api_base_in.rstrip("/")
    st.session_state["api_prefix"] = api_prefix_in
    st.session_state["token_path_override"] = token_path_in
    st.session_state["discovered_token_path"] = ""  # reset
    st.success("Paramètres API enregistrés. Rechargement…")
    st.rerun()   # <- au lieu de st.experimental_rerun()


            if st.button("🔎 Tester l'API"):
                url = build_url("/__paths")
                try:
                    r = _session.get(url, timeout=10)
                    st.write(f"GET {url} → {r.status_code}")
                    try:
                        st.json(r.json())
                    except Exception:
                        st.code(r.text)
                except Exception as e:
                    st.error(str(e))

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
