# client_streamlit.py
from __future__ import annotations
import os
import requests
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="🎮 Games UI", page_icon="🎮", layout="wide")

API_BASE = (
    st.secrets.get("API_BASE")
    or os.getenv("API_BASE")
    or "https://game-app-y8be.onrender.com"
)

DEFAULT_K = int(os.getenv("DEFAULT_K", "10"))

# -----------------------------------------------------------------------------
# Utils API
# -----------------------------------------------------------------------------
def _auth_headers() -> dict:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}

def api_get(path: str, timeout: float = 25.0):
    url = f"{API_BASE}{path}"
    r = requests.get(url, headers=_auth_headers(), timeout=timeout)
    if r.status_code == 401:
        raise PermissionError("Non autorisé. Connecte-toi d’abord.")
    r.raise_for_status()
    return r.json()

def api_post_form(path: str, data: dict, timeout: float = 25.0):
    url = f"{API_BASE}{path}"
    r = requests.post(url, data=data, timeout=timeout)
    r.raise_for_status()
    return r.json()

# -----------------------------------------------------------------------------
# Auth helpers
# -----------------------------------------------------------------------------
def do_login(username: str, password: str) -> bool:
    try:
        data = api_post_form("/token", {"username": username, "password": password})
        st.session_state["token"] = data["access_token"]
        st.session_state["username"] = username
        return True
    except requests.HTTPError as e:
        msg = e.response.json().get("detail") if e.response is not None else str(e)
        st.error(f"Échec de connexion : {msg}")
        return False

def do_register(username: str, password: str) -> bool:
    try:
        api_post_form("/register", {"username": username, "password": password})
        st.success("Compte créé ✅ Tu peux te connecter maintenant.")
        return True
    except requests.HTTPError as e:
        msg = e.response.json().get("detail") if e.response is not None else str(e)
        st.error(f"Inscription impossible : {msg}")
        return False

# -----------------------------------------------------------------------------
# UI header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .tag { padding:4px 10px; border-radius:999px; font-size:.80rem; margin-right:.35rem; }
      .tag-green { background:#E8FFF3; color:#137C4B; border:1px solid #9CE0C0; }
      .muted { color:#6b7280; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([1, 3])
with col_a:
    st.title("🎮 Games UI")
with col_b:
    st.markdown(
        """
        <div style="margin-top:14px;">
          <span class="tag tag-green">✅ Connecté</span> si tu vois des résultats.
          <span class="tag">🆕 Inscription</span>
          <span class="tag">🔑 Auth</span>
          <span class="tag">🔎 Recherche</span>
          <span class="tag">✨ Recos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Sidebar: Auth
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("🔐 Authentification")

    if "token" not in st.session_state:
        with st.form("login_form", clear_on_submit=False):
            u = st.text_input("Utilisateur", key="login_user")
            p = st.text_input("Mot de passe", type="password", key="login_pass")
            ok = st.form_submit_button("Se connecter")
        if ok:
            if do_login(u.strip(), p):
                st.toast("Connecté ✅", icon="✅")

        with st.expander("🆕 Créer un compte"):
            with st.form("register_form", clear_on_submit=True):
                ru = st.text_input("Utilisateur (nouveau)", key="reg_user")
                rp = st.text_input("Mot de passe (nouveau)", type="password", key="reg_pass")
                rok = st.form_submit_button("S’inscrire")
            if rok:
                if do_register(ru.strip(), rp):
                    st.toast("Compte créé 🎉", icon="🎉")
    else:
        st.success(f"Connecté en tant que **{st.session_state['username']}**")
        if st.button("Se déconnecter"):
            st.session_state.pop("token", None)
            st.session_state.pop("username", None)
            st.rerun()

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_search, tab_reco = st.tabs(["🔎 Recherche", "✨ Recommandations"])

# =============================================================================
# 🔎 Recherche par titre
# =============================================================================
with tab_search:
    st.subheader("🔎 Recherche par titre")
    q = st.text_input("Titre contient…", value="Mario")
    c1, c2 = st.columns([1, 3])
    with c1:
        k_for_quick = st.number_input("k (pour recos rapides)", 1, 50, min(DEFAULT_K, 10))
    with c2:
        if st.button("Chercher", use_container_width=False):
            if not st.session_state.get("token"):
                st.warning("Connecte-toi pour interroger l’API.")
            else:
                with st.spinner("Recherche en cours…"):
                    try:
                        data = api_get(f"/games/by-title/{requests.utils.quote(q)}")
                        rows = data.get("results", [])
                        df = pd.DataFrame(rows)
                        if df.empty:
                            st.info("Aucun résultat.")
                        else:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            st.caption("Clique une ligne pour obtenir des recos par **titre** ci-dessous 👇")

                            # Petites actions en table
                            for r in rows:
                                cols = st.columns([5, 1])
                                with cols[0]:
                                    st.write(f"**{r['title']}** (id source RAWG : {r['id']})")
                                with cols[1]:
                                    if st.button("✨ Recos", key=f"reco_{r['id']}"):
                                        st.session_state["reco_title_preset"] = r["title"]
                                        st.session_state["reco_k_preset"] = int(k_for_quick)
                                        st.switch_page("client_streamlit.py")
                    except PermissionError as e:
                        st.error(str(e))
                    except requests.HTTPError as e:
                        # 404/500 messages propres
                        detail = e.response.json().get("detail") if e.response is not None else str(e)
                        st.error(f"Erreur API: {detail}")
                    except Exception as e:
                        st.exception(e)

# =============================================================================
# ✨ Recommandations
# =============================================================================
with tab_reco:
    sub_tab_title, sub_tab_genre = st.tabs(["Par **titre**", "Par **genre**"])

    # ------------------ Par titre ------------------
    with sub_tab_title:
        st.subheader("✨ Recos — par titre")
        preset_title = st.session_state.pop("reco_title_preset", None)
        preset_k = st.session_state.pop("reco_k_preset", DEFAULT_K)
        title = st.text_input("Titre (ex. Mario)", value=preset_title or "Mario")
        k = st.number_input("Combien de recommandations ?", min_value=1, max_value=50, value=preset_k)

        col_go, _ = st.columns([1, 4])
        if col_go.button("Obtenir des recommandations", type="primary"):
            if not st.session_state.get("token"):
                st.warning("Connecte-toi pour interroger l’API.")
            else:
                with st.spinner("Prédiction du modèle…"):
                    try:
                        # L’API entraîne au besoin, puis rec par titre
                        data = api_get(f"/recommend/by-title/{requests.utils.quote(title)}?k={int(k)}")
                        recs = data.get("recommendations", [])
                        if not recs:
                            st.info("Aucune reco trouvée.")
                        else:
                            # recs items: [{'id':..., 'title':..., 'score':...}, ...] (selon ton modèle)
                            df = pd.DataFrame(recs)
                            # Propreté des colonnes
                            for c in ["id", "game_id", "game_id_rawg"]:
                                if c in df.columns and "id" not in df.columns:
                                    df.rename(columns={c: "id"}, inplace=True)
                            if "score" in df.columns:
                                df["score"] = (df["score"] * 100).round(1)
                                df.rename(columns={"score": "confidence_%"}, inplace=True)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                    except requests.HTTPError as e:
                        if e.response is not None:
                            try:
                                detail = e.response.json().get("detail")
                            except Exception:
                                detail = e.response.text
                        else:
                            detail = str(e)
                        # Message fréquent côté modèle scikit-learn
                        if isinstance(detail, str) and "NaN" in detail:
                            st.error("Le modèle a détecté des NaN dans les features. "
                                     "Essaie d’abord une reco **par genre** (ci-dessous) pour déclencher un entraînement propre, "
                                     "ou relance la recherche.")
                        else:
                            st.error(f"Erreur API: {detail}")
                    except Exception as e:
                        st.exception(e)

    # ------------------ Par genre ------------------
    with sub_tab_genre:
        st.subheader("✨ Recos — par genre")
        genre = st.text_input("Genre (ex. RPG, Action, Indie…)", value="RPG")
        k2 = st.number_input("k", min_value=1, max_value=50, value=DEFAULT_K, key="genre_k")
        if st.button("Recommander par genre"):
            if not st.session_state.get("token"):
                st.warning("Connecte-toi pour interroger l’API.")
            else:
                with st.spinner("Prédiction du modèle…"):
                    try:
                        data = api_get(f"/recommend/by-genre/{requests.utils.quote(genre)}?k={int(k2)}")
                        recs = data.get("recommendations", [])
                        if not recs:
                            st.info("Aucune reco trouvée.")
                        else:
                            df = pd.DataFrame(recs)
                            for c in ["id", "game_id", "game_id_rawg"]:
                                if c in df.columns and "id" not in df.columns:
                                    df.rename(columns={c: "id"}, inplace=True)
                            if "score" in df.columns:
                                df["score"] = (df["score"] * 100).round(1)
                                df.rename(columns={"score": "confidence_%"}, inplace=True)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                    except requests.HTTPError as e:
                        if e.response is not None:
                            try:
                                detail = e.response.json().get("detail")
                            except Exception:
                                detail = e.response.text
                        else:
                            detail = str(e)
                        st.error(f"Erreur API: {detail}")
                    except Exception as e:
                        st.exception(e)

# Footer
st.caption(
    f"API: `{API_BASE}` · Astuce : définis `API_BASE` dans Secrets/ENV pour cibler un autre déploiement."
)
