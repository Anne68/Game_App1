# client_streamlit.py — UI Streamlit (Auth, Recherche, Recos, Observabilité)
from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any, List
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

# =======================
# Config & Session State
# =======================
st.set_page_config(page_title="Games UI", page_icon="🎮", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")

if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "login_drawn_this_run" not in st.session_state:
    st.session_state["login_drawn_this_run"] = False

# =======================
# Helpers HTTP (API)
# =======================
def _auth_headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state['token']}"
    return headers

def api_post_form(path: str, data: Dict[str, Any], timeout: int = 10) -> tuple[int, Any]:
    url = f"{API_URL}{path}"
    try:
        r = requests.post(url, data=data, headers={"Accept": "application/json"}, timeout=timeout)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status_code, r.json()
        return r.status_code, {"detail": r.text}
    except Exception as e:
        return 0, {"detail": f"API POST error: {e}"}

def api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> tuple[int, Any]:
    url = f"{API_URL}{path}"
    try:
        r = requests.get(url, params=params, headers=_auth_headers(), timeout=timeout)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.status_code, r.json()
        return r.status_code, {"detail": r.text}
    except Exception as e:
        return 0, {"detail": f"API GET error: {e}"}

# =======================
# Auth UI
# =======================
def handle_401(code: int, payload: Any) -> bool:
    """Retourne True si 401 et affiche un message, sinon False."""
    if code == 401:
        st.warning("Session expirée ou non authentifiée. Merci de vous reconnecter.")
        st.session_state["token"] = None
        return True
    return False

def login_box(suffix: str = "main"):
    if st.session_state.get("token"):
        with st.expander("Connecté", expanded=False):
            st.write(f"Utilisateur : **{st.session_state.get('username', 'inconnu')}**")
            if st.button("Se déconnecter", key=f"logout_{suffix}"):
                st.session_state["token"] = None
                st.session_state["username"] = None
                st.experimental_rerun()
        return

    st.subheader("Authentification")
    with st.form(f"login_form_{suffix}", clear_on_submit=False):
        u = st.text_input("Username", key=f"login_u_{suffix}")
        p = st.text_input("Password", type="password", key=f"login_p_{suffix}")
        sub = st.form_submit_button("Login")
    if sub:
        code, payload = api_post_form("/token", data={"username": u.strip(), "password": p})
        if code == 200 and "access_token" in payload:
            st.session_state["token"] = payload["access_token"]
            st.session_state["username"] = u.strip()
            st.success("Authentifié ✅")
            st.experimental_rerun()
        else:
            st.error(payload.get("detail", "Échec d’authentification"))

# =======================
# UI: Recherche / Jeux
# =======================
def ui_games_search():
    st.subheader("Recherche de jeux par titre")
    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        q = st.text_input("Titre à rechercher", placeholder="ex: Halo", key="games_q")
    with col_b:
        k_page = st.number_input("Page", min_value=1, value=1, step=1, key="games_page")
    with col_c:
        k_size = st.number_input("Taille page", min_value=1, max_value=200, value=50, step=1, key="games_size")

    cta = st.button("Rechercher", type="primary", key="btn_games_search")
    if not cta:
        return

    title = (q or "").strip()
    if not title:
        st.warning("Saisissez un titre.")
        return

    # Recherche directe par titre (endpoint by-title)
    code, payload = api_get(f"/games/by-title/{quote(title)}")
    if handle_401(code, payload):
        st.stop()

    if code == 200:
        games = payload.get("games") or payload.get("items") or payload  # tolérance
        if not isinstance(games, list):
            st.info("Aucun résultat.")
            return
        df = pd.DataFrame(games)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Zone d'exploration prix/plateformes
        st.markdown("### Prix & Plateformes")
        sel_title = st.selectbox(
            "Choisir un jeu pour voir les prix et plateformes",
            options=[g.get("title") for g in games if g.get("title")],
            index=0 if games else None,
            key="sel_title_prices",
        )
        if sel_title:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("💰 Prix")
                c_code, c_payload = api_get(f"/games/title/{quote(sel_title)}/prices")
                if handle_401(c_code, c_payload):
                    st.stop()
                if c_code == 200:
                    prices = c_payload.get("prices", [])
                    if prices:
                        st.dataframe(pd.DataFrame(prices), use_container_width=True, hide_index=True)
                    else:
                        st.info("Pas de prix disponibles.")
                else:
                    st.warning(c_payload.get("detail", c_payload))

            with c2:
                st.caption("🕹️ Plateformes")
                p_code, p_payload = api_get(f"/games/title/{quote(sel_title)}/platforms")
                if handle_401(p_code, p_payload):
                    st.stop()
                if p_code == 200:
                    plats = p_payload.get("platforms", [])
                    if plats:
                        st.dataframe(pd.DataFrame(plats), use_container_width=True, hide_index=True)
                    else:
                        st.info("Pas de plateformes disponibles.")
                else:
                    st.warning(p_payload.get("detail", p_payload))
    elif code == 404:
        st.info("Aucun jeu trouvé.")
    else:
        st.error(payload.get("detail", payload))

# =======================
# UI: Recommandations
# =======================
def ui_recommendations():
    st.subheader("Recommandations IA")

    tab1, tab2 = st.tabs(["Par titre", "Par genre"])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            title = st.text_input("Titre", placeholder="ex: Halo Infinite", key="rec_title")
        with col2:
            k = st.slider("k", 1, 50, 5, key="rec_k_title")
        if st.button("Obtenir des recommandations (titre)", key="btn_recos_title"):
            if not (title or "").strip():
                st.warning("Saisissez un titre.")
            else:
                code, payload = api_get(f"/recommend/by-title/{quote(title.strip())}", params={"k": int(k)})
                if handle_401(code, payload):
                    st.stop()
                if code == 200:
                    recs = payload.get("recommendations") or payload.get("items") or []
                    st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
                elif code == 404:
                    st.info("Aucune recommandation.")
                else:
                    st.error(payload.get("detail", payload))

    with tab2:
        col1, col2 = st.columns([3, 1])
        with col1:
            genre = st.text_input("Genre", placeholder="ex: Action", key="rec_genre")
        with col2:
            k2 = st.slider("k", 1, 50, 5, key="rec_k_genre")
        if st.button("Obtenir des recommandations (genre)", key="btn_recos_genre"):
            if not (genre or "").strip():
                st.warning("Saisissez un genre.")
            else:
                code, payload = api_get(f"/recommend/by-genre/{quote(genre.strip())}", params={"k": int(k2)})
                if handle_401(code, payload):
                    st.stop()
                if code == 200:
                    recs = payload.get("recommendations") or payload.get("items") or []
                    st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
                elif code == 404:
                    st.info("Aucune recommandation.")
                else:
                    st.error(payload.get("detail", payload))

# =======================
# Observabilité (Prometheus)
# =======================
# Requêtes PromQL alignées sur tes règles d’alertes
PROMQL_ERROR_RATE_5XX_5M = 'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))'
PROMQL_LATENCY_P95_5M = 'histogram_quantile(0.95, sum(rate(api_request_latency_seconds_bucket[5m])) by (le))'
PROMQL_RATE_401_5M = 'sum(rate(http_requests_total{status="401"}[5m]))'

@st.cache_data(ttl=10)
def prom_instant_query(query: str) -> Optional[float]:
    """Retourne la valeur float d'une requête instant PromQL, ou None si pas de résultat."""
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=5)
        r.raise_for_status()
        data = r.json()
        res = data.get("data", {}).get("result", [])
        if not res:
            return None
        value = float(res[0]["value"][1])
        return value
    except Exception:
        return None

@st.cache_data(ttl=10)
def prom_active_alerts() -> List[Dict[str, Any]]:
    """Retourne la liste des alertes actives depuis Prometheus (API /api/v1/alerts)."""
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/alerts", timeout=5)
        r.raise_for_status()
        data = r.json()
        alerts = data.get("data", {}).get("alerts", [])
        return alerts or []
    except Exception:
        return []

def ui_observability():
    st.subheader("Observabilité (Prometheus)")

    col1, col2, col3 = st.columns(3)
    err_rate = prom_instant_query(PROMQL_ERROR_RATE_5XX_5M)  # ratio (0..1)
    p95 = prom_instant_query(PROMQL_LATENCY_P95_5M)          # secondes
    rate_401 = prom_instant_query(PROMQL_RATE_401_5M)        # req/s

    # Taux d'erreurs 5xx
    if err_rate is None:
        col1.metric("Taux d’erreurs 5xx (5m)", "n/a")
    else:
        col1.metric("Taux d’erreurs 5xx (5m)", f"{err_rate*100:.2f} %")
        if err_rate > 0.05:
            col1.error("Alerte: > 5%")
        else:
            col1.success("OK")

    # Latence P95
    if p95 is None:
        col2.metric("Latence P95 (5m)", "n/a")
    else:
        col2.metric("Latence P95 (5m)", f"{p95:.3f} s")
        if p95 > 1.0:
            col2.error("Alerte: > 1s")
        else:
            col2.success("OK")

    # 401/s
    if rate_401 is None:
        col3.metric("401/s (5m)", "n/a")
    else:
        col3.metric("401/s (5m)", f"{rate_401:.2f}")
        if rate_401 > 5:
            col3.warning("Pic de 401")
        else:
            col3.success("OK")

    st.divider()
    st.caption("Seuils alignés sur les règles d’alertes Prometheus.")

    # Alertes actives
    st.markdown("### Alertes actives")
    alerts = prom_active_alerts()
    if not alerts:
        st.success("Aucune alerte active 👍")
    else:
        for a in alerts:
            name = a.get("labels", {}).get("alertname", "unknown")
            sev = a.get("labels", {}).get("severity", "unknown")
            summary = (a.get("annotations", {}) or {}).get("summary", "")
            state = a.get("state", "firing")
            starts_at = a.get("activeAt", a.get("startsAt", ""))

            box = st.container(border=True)
            with box:
                st.write(f"**{name}** — état: `{state}` — sévérité: `{sev}`")
                if summary:
                    st.write(summary)
                if starts_at:
                    st.caption(f"Active depuis: {starts_at}")

# =======================
# Layout principal
# =======================
with st.sidebar:
    st.markdown("### Configuration")
    st.text_input("API_URL", value=API_URL, key="cfg_api_url", help="Ex: http://localhost:8000")
    st.text_input("PROMETHEUS_URL", value=PROMETHEUS_URL, key="cfg_prom_url", help="Ex: http://localhost:9090")
    st.caption("Modifiez les variables d’environnement pour persister ces URLs.")

st.title("🎮 Games — Recherche, Recos & Observabilité")

# Auth
login_box()

# Si non authentifié, arrêter ici (les endpoints /games/* et /recommend/* requièrent le token)
if not st.session_state.get("token"):
    st.stop()

# Tabs
tab_search, tab_recos, tab_obs = st.tabs(["Recherche", "Recommandations", "Observabilité"])

with tab_search:
    ui_games_search()

with tab_recos:
    ui_recommendations()

with tab_obs:
    ui_observability()

# (Option) Auto-refresh Observabilité — simple bouton
with tab_obs:
    colr1, colr2 = st.columns([1, 9])
    if colr1.button("🔄 Rafraîchir les métriques", key="btn_refresh_metrics"):
        # Invalider le cache Prometheus et relancer le rendering
        prom_instant_query.clear()
        prom_active_alerts.clear()
        time.sleep(0.2)
        st.experimental_rerun()
