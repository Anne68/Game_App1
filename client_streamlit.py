import os
import time
import requests
import streamlit as st

# =============================================================================
# Config & session
# =============================================================================
API_BASE_URL = (
    st.secrets.get("API_URL")
    or os.getenv("API_URL")
    or "https://game-app-y8be.onrender.com"  # <- remplace si besoin
)

st.set_page_config(page_title="Games UI", page_icon="🎮", layout="wide")

# Etat de session
ss = st.session_state
ss.setdefault("token", None)
ss.setdefault("username", None)
ss.setdefault("favorites", set())
ss.setdefault("last_search", "")
ss.setdefault("last_results", [])

# =============================================================================
# Helpers API
# =============================================================================
def api_headers():
    h = {"accept": "application/json"}
    if ss.token:
        h["Authorization"] = f"Bearer {ss.token}"
    return h

def api_get(path, params=None, timeout=30):
    url = f"{API_BASE_URL}{path}"
    r = requests.get(url, headers=api_headers(), params=params, timeout=timeout)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text
        raise RuntimeError(detail or f"Erreur API ({r.status_code})")
    return r.json()

def api_post(path, json=None, data=None, timeout=60):
    url = f"{API_BASE_URL}{path}"
    r = requests.post(url, headers=api_headers(), json=json, data=data, timeout=timeout)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text
        raise RuntimeError(detail or f"Erreur API ({r.status_code})")
    return r.json()

# =============================================================================
# Styles (UI/UX)
# =============================================================================
st.markdown("""
<style>
/* Global */
:root { --card-bg: rgba(255,255,255,.65); --soft-border: 1px solid rgba(0,0,0,.08); }
html, body, [class^="css"]  { font-variation-settings: "wght" 520; }

/* Header */
.app-hero { display:flex; gap:12px; align-items:center; margin:8px 0 4px; }
.app-hero .logo { font-size:28px; }
.app-hero .title { font-size:36px; font-weight:800; letter-spacing:.2px; }

/* Pills nav header */
.top-pills { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 12px; }
.pill { background:#f6f7fb; border:var(--soft-border); padding:6px 10px; border-radius:999px; font-size:.92rem; }
.pill .ico { margin-right:6px; }

/* Cards */
.card { border: var(--soft-border); border-radius:14px; padding:14px; background:var(--card-bg); margin-bottom:12px; }
.card h4 { margin: 0 0 4px 0; font-weight:800; }
.meta { opacity:.8; font-size:.9rem; }
.price { font-weight:700; }
.badge { display:inline-block; padding:.22rem .5rem; border-radius:999px; font-size:.78rem; margin:0 .25rem .25rem 0; border:var(--soft-border); }
.badge.pc { background:#e7f1ff; }
.badge.ps { background:#f4e9ff; }
.badge.xbox { background:#eafbf1; }
.badge.switch { background:#ffe9ea; }
.badge.other { background:#f2f2f2; }
.btn-link a { text-decoration:none; font-weight:700; }

/* Sidebar status */
.side-status { padding:10px; border-radius:12px; border:var(--soft-border); background:rgba(0,0,0,.03); }
.small { font-size:.9rem; opacity:.85; }

/* Chips cliquables */
.chips { display:flex; gap:6px; flex-wrap:wrap; }
.chip { border:var(--soft-border); padding:6px 10px; border-radius:999px; cursor:pointer; background:#fff; }
.chip:hover { background:#f6f7fb; }

/* Compact slider */
.stSlider [data-baseweb="slider"] { margin-top:-10px; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header (hero + nav pills)
# =============================================================================
st.markdown(f"""
<div class="app-hero">
  <div class="logo">🎮</div>
  <div class="title">Games UI</div>
</div>
<div class="top-pills">
  <span class="pill"><span class="ico">🆕</span> Inscription</span>
  <span class="pill"><span class="ico">🔑</span> Auth</span>
  <span class="pill"><span class="ico">🔎</span> Recherche</span>
  <span class="pill"><span class="ico">✨</span> Recos</span>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar (status + favoris)
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    st.write(f"**API** : `{API_BASE_URL}`")
    st.markdown("---")
    st.markdown("### 👤 Session")
    box = st.container()
    with box:
        st.markdown('<div class="side-status">', unsafe_allow_html=True)
        if ss.token:
            st.success(f"Connecté : **{ss.username}**")
            if st.button("Se déconnecter", use_container_width=True):
                ss.token = None
                ss.username = None
                st.toast("Déconnecté")
                st.rerun()
        else:
            st.info("Non connecté")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### ⭐ Favoris")
    if ss.favorites:
        for t in sorted(ss.favorites):
            st.write(f"• {t}")
    else:
        st.caption("Aucun favori…")

# =============================================================================
# Onglets
# =============================================================================
tab_reg, tab_auth, tab_search, tab_recos = st.tabs(["🆕 Inscription", "🔑 Auth", "🔎 Recherche", "✨ Recos"])

# ---------------- Inscription
with tab_reg:
    st.subheader("Créer un compte")
    with st.form("reg"):
        u = st.text_input("Nom d'utilisateur")
        p = st.text_input("Mot de passe", type="password")
        colA, colB = st.columns([1,3])
        ok = colA.form_submit_button("S'inscrire", use_container_width=True)
        if ok:
            try:
                api_post("/register", data={"username": u, "password": p})
                st.success("Compte créé ✅ Allez dans l’onglet **Auth**.")
                st.toast("Inscription réussie")
            except Exception as e:
                st.error(f"Inscription impossible : {e}")

# ---------------- Auth
with tab_auth:
    st.subheader("Connexion")
    if ss.token:
        st.success(f"Déjà connecté en tant que **{ss.username}**")
    with st.form("login"):
        u = st.text_input("Nom d'utilisateur", value=ss.username or "")
        p = st.text_input("Mot de passe", type="password")
        col1, col2 = st.columns([1,3])
        ok = col1.form_submit_button("Se connecter", use_container_width=True)
        if ok:
            try:
                r = api_post("/token", data={"username": u, "password": p})
                ss.token = r["access_token"]
                ss.username = u
                st.success("Connecté ✅")
                st.toast("Bienvenue !")
                time.sleep(0.2)
                st.rerun()
            except Exception as e:
                st.error(f"Connexion impossible : {e}")

# ---------------- Recherche (cartes visuelles)
PLAT_COLORS = {
    "PC": "pc", "Windows": "pc",
    "PlayStation": "ps", "PS4": "ps", "PS5": "ps", "PS3": "ps",
    "Xbox": "xbox", "Xbox One": "xbox", "Xbox Series": "xbox", "X360":"xbox",
    "Switch": "switch", "Nintendo Switch": "switch", "Wii":"switch", "Wii U":"switch",
}

def render_platform_badges(platforms):
    if not platforms:
        return '<span class="badge other">Unknown</span>'
    parts = []
    for p in platforms:
        cls = "other"
        for k, v in PLAT_COLORS.items():
            if k.lower() in p.lower():
                cls = v
                break
        parts.append(f'<span class="badge {cls}">{p}</span>')
    return " ".join(parts)

with tab_search:
    st.subheader("🔎 Recherche par titre")

    # suggestions rapides
    st.caption("Suggestions rapides :")
    sug = st.container()
    with sug:
        st.markdown('<div class="chips">', unsafe_allow_html=True)
        for s in ["Mario", "Zelda", "Elden Ring", "The Witcher", "Halo", "Fortnite"]:
            if st.button(s, key=f"chip_{s}", help="Rechercher ce titre", use_container_width=False):
                ss.last_search = s
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with st.form("search_form"):
        title = st.text_input("Titre contient…", value=ss.last_search or "", placeholder="Mario, Zelda, ...")
        submitted = st.form_submit_button("Chercher", type="primary")
        if submitted:
            if not title.strip():
                st.warning("Saisis un titre.")
            else:
                with st.status("Recherche en cours…", expanded=False) as status:
                    try:
                        data = api_get(f"/games/by-title/{title.strip()}")
                        ss.last_search = title.strip()
                        ss.last_results = data.get("results", [])
                        status.update(label="Terminé ✅", state="complete")
                        st.toast("Résultats chargés")
                    except Exception as e:
                        status.update(label="Erreur", state="error")
                        st.error(f"Erreur : {e}")

    # Affichage résultats
    results = ss.last_results
    if results:
        st.caption(f"{len(results)} résultat(s) pour **{ss.last_search}**")
        for r in results:
            price = r.get("best_price")
            shop = r.get("best_shop")
            site = r.get("site_url")
            plats = r.get("platforms") or []
            rating = r.get("rating")
            meta = r.get("metacritic")
            gid = r.get("id")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Header + actions
            top = st.columns([8,2,2])
            with top[0]:
                st.markdown(f"<h4>{r['title']}</h4>", unsafe_allow_html=True)
                st.markdown(render_platform_badges(plats), unsafe_allow_html=True)
            with top[1]:
                st.markdown(f'<span class="meta">⭐ Note&nbsp;: <b>{rating if rating is not None else "—"}</b></span>', unsafe_allow_html=True)
            with top[2]:
                st.markdown(f'<span class="meta">🧪 Metacritic&nbsp;: <b>{meta if meta is not None else "—"}</b></span>', unsafe_allow_html=True)

            # Price / CTA
            cols = st.columns([6,3,3])
            with cols[0]:
                st.caption(f"ID RAWG: {gid}")
            with cols[1]:
                if price:
                    if site:
                        st.markdown(f'<div class="btn-link"><a href="{site}" target="_blank">💸 {price} — {shop or "Store"}</a></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"💸 **{price}** {('— ' + shop) if shop else ''}")
                else:
                    st.markdown('<span class="meta">💸 Prix inconnu</span>', unsafe_allow_html=True)
            with cols[2]:
                fav_key = f"fav_{gid}"
                add = st.button("⭐ Favori", key=fav_key)
                if add:
                    ss.favorites.add(r["title"])
                    st.toast("Ajouté aux favoris")

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Tape un titre ci-dessus pour lancer une recherche.")

# ---------------- Recommandations (UX plus robuste)
with tab_recos:
    st.subheader("✨ Recommandations")
    st.caption("Saisis un genre ou un mot-clé. Ex: *RPG*, *indie*, *souls-like*, *space*…")

    prompt = st.text_input("Prompt", value="RPG")
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        k = st.slider("Nombre de propositions", min_value=1, max_value=50, value=10)
    with c2:
        min_conf = st.number_input("Confiance min", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    if st.button("Lancer la reco", type="primary"):
        if not ss.token:
            st.error("Connecte-toi d'abord (onglet **Auth**).")
        else:
            with st.status("Calcul des recos…", expanded=False) as status:
                try:
                    res = api_post("/recommend/ml", json={"query": prompt, "k": int(k), "min_confidence": float(min_conf)})
                    recs = res.get("recommendations", [])
                    status.update(label="Terminé ✅", state="complete")
                    if not recs:
                        st.info("Pas de recommandations.")
                    else:
                        for r in recs:
                            title = r.get("title") or r.get("name") or f"#{r.get('id')}"
                            score = r.get("score") or r.get("similarity") or r.get("confidence")
                            st.markdown(f"- **{title}** — score: `{round(score, 3) if isinstance(score,(int,float)) else score}`")
                        st.toast("Reco terminée")
                except Exception as e:
                    status.update(label="Erreur", state="error")
                    st.error(f"{e}")
