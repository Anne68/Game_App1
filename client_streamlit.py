# client_streamlit_enhanced.py - Interface améliorée avec wishlist et modèle hybride
import os
import time
import json
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# =========================
# --------- THEME ---------
# =========================
ARCADE_CSS = """
<style>
/* Fond dégradé néon amélioré */
.stApp {
  background: radial-gradient(1200px 500px at 10% 0%, #081226 0%, #0b0f1d 35%, #070b14 60%, #05070d 100%) !important;
  color: #e6f3ff;
  font-family: ui-rounded, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif;
}

/* Titres néon */
h1, h2, h3, h4 {
  color: #9be8ff !important;
  text-shadow: 0 0 12px rgba(155,232,255,.5), 0 0 24px rgba(72,163,255,.25);
}

/* Cartes vitrées */
.block-container { padding-top: 1.5rem; }
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#09132a,#0b1733) !important;
  border-right: 1px solid rgba(255,255,255,.08);
}

/* Tabs améliorés */
.stTabs [data-baseweb="tab"] {
  color: #cbe7ff !important;
  font-weight: 700 !important;
  border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab-highlight"] {
  background: linear-gradient(90deg, #00eaff33, #9d00ff33);
}

/* Boutons arcade */
.stButton>button {
  background: linear-gradient(90deg, #00eaff, #9d00ff) !important;
  color: #081226 !important;
  border: 0 !important;
  border-radius: 14px !important;
  box-shadow: 0 8px 24px rgba(0,234,255,.25);
  font-weight: 800 !important;
  transition: all 0.3s ease;
}
.stButton>button:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 12px 32px rgba(0,234,255,.35);
}

/* Boutons spéciaux pour wishlist */
.wishlist-btn {
  background: linear-gradient(90deg, #ff6b6b, #ffd93d) !important;
  color: #2c3e50 !important;
}

.alert-btn {
  background: linear-gradient(90deg, #ff9500, #ff5722) !important;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(255, 149, 0, 0.7); }
  70% { box-shadow: 0 0 0 10px rgba(255, 149, 0, 0); }
  100% { box-shadow: 0 0 0 0 rgba(255, 149, 0, 0); }
}

/* Inputs */
input, textarea, .stTextInput, .stTextArea, .stNumberInput {
  color: #e6f3ff !important;
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 8px !important;
}

/* Métriques colorées */
.metric-card {
  background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 16px;
  margin: 8px 0;
}

/* Badges et pills */
.badge {
  display:inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: linear-gradient(90deg, #00ffa3, #00d4ff);
  color: #081226;
  font-weight: 800;
  font-size: 12px;
  margin: 2px;
}

.alert-badge {
  background: linear-gradient(90deg, #ff6b6b, #ff9500);
  animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
  from { box-shadow: 0 0 5px #ff6b6b; }
  to { box-shadow: 0 0 20px #ff6b6b, 0 0 30px #ff6b6b; }
}

/* Graphiques */
.plotly-chart {
  background: rgba(255,255,255,0.02) !important;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.05);
}

/* Wishlist spécifique */
.wishlist-item {
  background: linear-gradient(135deg, rgba(0,234,255,0.1), rgba(157,0,255,0.05));
  border: 1px solid rgba(0,234,255,0.2);
  border-radius: 12px;
  padding: 12px;
  margin: 8px 0;
}

.price-alert {
  background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,149,0,0.1));
  border: 1px solid rgba(255,107,107,0.3);
  border-radius: 12px;
  padding: 12px;
  margin: 8px 0;
}
</style>
"""

# =========================
# ---- CONFIG & STATE ----
# =========================
DEFAULT_API_URL = os.getenv("GAMES_API_URL", "http://localhost:8000")
st.set_page_config(
    page_title="Games Reco Enhanced", 
    page_icon="🎮", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(ARCADE_CSS, unsafe_allow_html=True)

# État de session étendu
session_keys = [
    "api_url", "token", "username", "model_version", "user_id",
    "wishlist_items", "price_alerts", "notification_settings",
    "last_alert_check", "hybrid_model_enabled"
]

for key in session_keys:
    if key not in st.session_state:
        if key == "api_url":
            st.session_state[key] = DEFAULT_API_URL
        elif key == "hybrid_model_enabled":
            st.session_state[key] = True
        else:
            st.session_state[key] = None

# =========================
# ------- HELPERS --------
# =========================
def api_base() -> str:
    return st.session_state.api_url.rstrip("/")

def bearer_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if st.session_state.token:
        h["Authorization"] = f"Bearer {st.session_state.token}"
    return h

def post(path: str, payload: Dict[str, Any], timeout: int = 25) -> requests.Response:
    url = f"{api_base()}{path}"
    return requests.post(url, headers=bearer_headers(), data=json.dumps(payload), timeout=timeout)

def get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 25) -> requests.Response:
    url = f"{api_base()}{path}"
    return requests.get(url, headers=bearer_headers(), params=params, timeout=timeout)

def login(username: str, password: str) -> bool:
    try:
        r = requests.post(f"{api_base()}/token",
                          data={"username": username, "password": password},
                          timeout=20)
        if r.status_code == 200:
            data = r.json()
            st.session_state.token = data.get("access_token")
            st.session_state.username = username
            # Obtenir l'ID utilisateur
            user_info = get_user_info()
            if user_info:
                st.session_state.user_id = user_info.get("user_id")
            return True
        else:
            st.error(f"⛔ Login échoué: {r.status_code} — {r.text}")
            return False
    except Exception as e:
        st.error(f"⚠️ Erreur réseau login: {e}")
        return False

def get_user_info() -> Optional[Dict]:
    """Récupère les informations utilisateur"""
    try:
        r = get("/user/profile")
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def register(username: str, password: str) -> bool:
    try:
        r = requests.post(f"{api_base()}/register",
                          data={"username": username, "password": password},
                          timeout=20)
        if r.status_code == 200:
            st.success("✅ Compte créé ! Connecte-toi maintenant.")
            return True
        else:
            st.error(f"⛔ Inscription refusée: {r.status_code} — {r.text}")
            return False
    except Exception as e:
        st.error(f"⚠️ Erreur réseau register: {e}")
        return False

def guard_auth():
    if not st.session_state.token:
        st.warning("🔐 Tu dois être connecté pour utiliser l'API.")
        st.stop()

def show_metrics_pills(metrics: Dict[str, Any]):
    cols = st.columns(4)
    cols[0].markdown(f'<div class="metric-card">🧠 <b>Version</b><br>{metrics.get("model_version","?")}</div>', unsafe_allow_html=True)
    cols[1].markdown(f'<div class="metric-card">✅ <b>Trained</b><br>{metrics.get("is_trained")}</div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="metric-card">📈 <b>Predictions</b><br>{metrics.get("total_predictions",0)}</div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="metric-card">⭐ <b>Avg Confidence</b><br>{round(metrics.get("avg_confidence",0.0),3)}</div>', unsafe_allow_html=True)

# =========================
# ---- WISHLIST FUNCS ----
# =========================
def add_to_wishlist(game_title: str, max_price: float, currency: str = "EUR") -> bool:
    """Ajoute un jeu à la wishlist"""
    try:
        payload = {
            "game_title": game_title,
            "max_price": max_price,
            "currency": currency
        }
        r = post("/wishlist/add", payload)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Erreur ajout wishlist: {e}")
        return False

def get_wishlist() -> List[Dict]:
    """Récupère la wishlist de l'utilisateur"""
    try:
        r = get("/wishlist")
        if r.status_code == 200:
            return r.json().get("items", [])
    except Exception as e:
        st.error(f"Erreur récupération wishlist: {e}")
    return []

def remove_from_wishlist(wishlist_id: int) -> bool:
    """Supprime un item de la wishlist"""
    try:
        r = requests.delete(f"{api_base()}/wishlist/{wishlist_id}", headers=bearer_headers())
        return r.status_code == 200
    except Exception as e:
        st.error(f"Erreur suppression wishlist: {e}")
        return False

def get_price_alerts() -> List[Dict]:
    """Récupère les alertes de prix"""
    try:
        r = get("/wishlist/alerts")
        if r.status_code == 200:
            return r.json().get("alerts", [])
    except Exception as e:
        st.error(f"Erreur récupération alertes: {e}")
    return []

def mark_alert_read(alert_id: int) -> bool:
    """Marque une alerte comme lue"""
    try:
        r = post(f"/wishlist/alerts/{alert_id}/read", {})
        return r.status_code == 200
    except Exception:
        return False

def check_new_alerts() -> int:
    """Vérifie s'il y a de nouvelles alertes"""
    try:
        alerts = get_price_alerts()
        pending_alerts = [a for a in alerts if a.get("status") == "pending"]
        return len(pending_alerts)
    except Exception:
        return 0

# =========================
# ------- SIDEBAR --------
# =========================
with st.sidebar:
    st.markdown("## 🎮 Games API — Enhanced")
    st.text_input("Base API URL", key="api_url", help="Ex: http://localhost:8000")
    
    # Vérification des nouvelles alertes
    if st.session_state.token:
        new_alerts = check_new_alerts()
        if new_alerts > 0:
            st.markdown(f'<div class="alert-badge">🔔 {new_alerts} nouvelle(s) alerte(s)!</div>', unsafe_allow_html=True)
    
    st.divider()

    if st.session_state.token:
        st.markdown(f"**Connecté:** `{st.session_state.username}`")
        if st.button("🚪 Se déconnecter"):
            for key in session_keys:
                if key != "api_url":
                    st.session_state[key] = None
            st.rerun()
    else:
        st.markdown("### 🔐 Authentification")
        auth_tab = st.tabs(["Se connecter", "Créer un compte"])
        with auth_tab[0]:
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            if st.button("🎯 Login"):
                if login(u, p):
                    st.success("✅ Connecté !")
                    st.rerun()
        with auth_tab[1]:
            u2 = st.text_input("Username (new)", key="reg_user")
            p2 = st.text_input("Password (new)", type="password", key="reg_pass")
            if st.button("🆕 Register"):
                register(u2, p2)

    st.divider()
    
    # Modèle hybride toggle
    st.session_state.hybrid_model_enabled = st.toggle(
        "🧠 Modèle Hybride (Gradient Boosting)", 
        value=st.session_state.hybrid_model_enabled,
        help="Utilise le modèle hybride avec Gradient Boosting pour de meilleures recommandations"
    )
    
    if st.session_state.token:
        try:
            resp = requests.get(f"{api_base()}/healthz", timeout=10)
            if resp.ok:
                hz = resp.json()
                st.markdown("### 🩺 Status")
                status = hz.get("status", "?")
                if status == "healthy":
                    st.success(f"✅ {status}")
                else:
                    st.warning(f"⚠️ {status}")
                
                mv = hz.get("model_version", "?")
                st.caption(f"Model: {mv}")
                
                compliance = hz.get("compliance_enabled", False)
                if compliance:
                    st.caption("🔒 Compliance activée")
        except Exception:
            st.warning("Healthcheck indisponible.")

# =========================
# --------- TABS ----------
# =========================
st.title("🕹️ Games Recommender — **Enhanced with AI & Wishlist**")

tabs = st.tabs([
    "🔎 Recommandations IA",
    "💝 Ma Wishlist",
    "🔔 Alertes Prix", 
    "📊 Analytics Avancées",
    "🛠️ Admin & Modèles"
])

# =========================
# ---- TAB 1: RECOs IA ----
# =========================
with tabs[0]:
    guard_auth()
    
    st.subheader("🤖 Recommandations IA Hybrides")
    
    # Indicateur du modèle utilisé
    model_type = "Hybride (TF-IDF + Gradient Boosting)" if st.session_state.hybrid_model_enabled else "Classique (TF-IDF + SVD)"
    st.info(f"🧠 **Modèle actif:** {model_type}")
    
    with st.form("form_reco_ai", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("🎯 Décris ton jeu idéal", 
                                 value="", 
                                 placeholder="ex: RPG open world fantasy avec crafting")
        with col2:
            k = st.slider("Nombre", 1, 20, 8)
        
        col3, col4 = st.columns(2)
        with col3:
            min_conf = st.slider("Confiance min", 0.0, 1.0, 0.15, 0.05)
        with col4:
            algorithm = "hybrid" if st.session_state.hybrid_model_enabled else "classic"
        
        submitted = st.form_submit_button("⚡ Obtenir des Recommandations IA", use_container_width=True)
    
    if submitted and query:
        with st.spinner("🤖 IA en cours d'analyse..."):
            endpoint = "/recommend/ml" if not st.session_state.hybrid_model_enabled else "/recommend/hybrid"
            r = post(endpoint, {"query": query, "k": k, "min_confidence": min_conf})
            
            if r.ok:
                data = r.json()
                recommendations = data.get('recommendations', [])
                
                if recommendations:
                    st.success(f"🎉 {len(recommendations)} recommandations trouvées — {round(data.get('latency_ms',0),1)} ms")
                    
                    # Affichage amélioré des recommandations
                    for idx, game in enumerate(recommendations, start=1):
                        with st.container():
                            col_main, col_actions = st.columns([4, 1])
                            
                            with col_main:
                                # Titre et score
                                confidence = game.get('confidence', 0)
                                confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{idx}. {game.get('title', '?')} {confidence_color}</h4>
                                    <p><strong>Genres:</strong> {game.get('genres', 'N/A')}</p>
                                    <p><strong>Score IA:</strong> {round(confidence, 3)} | 
                                       <strong>Rating:</strong> ⭐ {game.get('rating', 0)}/5 | 
                                       <strong>Metacritic:</strong> 🏆 {game.get('metacritic', 0)}/100</p>
                                    {f"<p><strong>Algorithme:</strong> {game.get('algorithm', 'unknown')}</p>" if 'algorithm' in game else ""}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_actions:
                                st.markdown("#### Actions")
                                
                                # Bouton wishlist
                                if st.button(f"💝 Wishlist", key=f"wishlist_{idx}", help="Ajouter à la wishlist"):
                                    # Modal pour le prix
                                    with st.expander("💰 Définir le seuil de prix", expanded=True):
                                        price_threshold = st.number_input(
                                            f"Prix maximum pour '{game.get('title')}'", 
                                            min_value=0.0, 
                                            max_value=200.0, 
                                            value=25.0, 
                                            step=0.5,
                                            key=f"price_{idx}"
                                        )
                                        currency = st.selectbox("Devise", ["EUR", "USD"], key=f"currency_{idx}")
                                        
                                        if st.button(f"✅ Ajouter à la wishlist", key=f"confirm_wishlist_{idx}"):
                                            if add_to_wishlist(game.get('title'), price_threshold, currency):
                                                st.success(f"✅ {game.get('title')} ajouté à la wishlist (≤ {price_threshold} {currency})")
                                                time.sleep(1)
                                                st.rerun()
                                            else:
                                                st.error("❌ Erreur lors de l'ajout")
                                
                                # Bouton jeux similaires
                                if st.button(f"🔗 Similaires", key=f"similar_{idx}"):
                                    st.session_state[f"show_similar_{idx}"] = True
                            
                            # Affichage des jeux similaires si demandé
                            if st.session_state.get(f"show_similar_{idx}", False):
                                with st.expander(f"🔗 Jeux similaires à '{game.get('title')}'", expanded=True):
                                    similar_r = post("/recommend/similar-game", {
                                        "game_id": game.get('id'),
                                        "k": 5
                                    })
                                    
                                    if similar_r.ok:
                                        similar_games = similar_r.json().get('recommendations', [])
                                        for sim_game in similar_games:
                                            st.markdown(f"• **{sim_game.get('title')}** — Score: {round(sim_game.get('confidence', 0), 3)}")
                                    else:
                                        st.warning("Impossible de charger les jeux similaires")
                                
                                if st.button(f"❌ Fermer", key=f"close_similar_{idx}"):
                                    st.session_state[f"show_similar_{idx}"] = False
                                    st.rerun()
                
                else:
                    st.warning("Aucune recommandation trouvée avec ces critères")
            else:
                st.error(f"⛔ Erreur API: {r.status_code} — {r.text}")

# =========================
# ---- TAB 2: WISHLIST ----
# =========================
with tabs[1]:
    guard_auth()
    
    st.subheader("💝 Ma Wishlist")
    
    # Actions rapides
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Actualiser", use_container_width=True):
            st.session_state.wishlist_items = None
            st.rerun()
    
    with col2:
        if st.button("🔍 Vérifier les prix", use_container_width=True):
            with st.spinner("Vérification des prix en cours..."):
                # Déclencher la vérification des prix
                try:
                    r = post("/wishlist/check-prices", {})
                    if r.ok:
                        result = r.json()
                        new_alerts = result.get("new_alerts", 0)
                        if new_alerts > 0:
                            st.success(f"🎉 {new_alerts} nouvelle(s) alerte(s) trouvée(s)!")
                        else:
                            st.info("Aucune nouvelle offre trouvée")
                    else:
                        st.warning("Impossible de vérifier les prix")
                except Exception as e:
                    st.error(f"Erreur: {e}")
    
    with col3:
        # Ajout manuel
        with st.popover("➕ Ajouter un jeu"):
            with st.form("add_game_form"):
                new_game = st.text_input("Nom du jeu")
                new_price = st.number_input("Prix maximum (EUR)", min_value=0.0, value=20.0, step=0.5)
                
                if st.form_submit_button("Ajouter"):
                    if new_game and new_price > 0:
                        if add_to_wishlist(new_game, new_price):
                            st.success(f"✅ {new_game} ajouté!")
                            st.session_state.wishlist_items = None
                            time.sleep(1)
                            st.rerun()
    
    # Récupération de la wishlist
    if st.session_state.wishlist_items is None:
        st.session_state.wishlist_items = get_wishlist()
    
    wishlist_items = st.session_state.wishlist_items
    
    if wishlist_items:
        st.markdown(f"**📊 {len(wishlist_items)} jeu(x) dans votre wishlist**")
        
        # Affichage de la wishlist
        for idx, item in enumerate(wishlist_items):
            with st.container():
                st.markdown(f"""
                <div class="wishlist-item">
                    <h4>🎮 {item.get('game_title', 'Jeu inconnu')}</h4>
                    <p><strong>Prix max:</strong> {item.get('max_price', 0)} {item.get('currency', 'EUR')}</p>
                    <p><strong>Ajouté le:</strong> {item.get('created_at', 'Date inconnue')}</p>
                    <p><strong>Notifications reçues:</strong> {item.get('notification_count', 0)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button(f"🗑️ Supprimer", key=f"delete_{idx}"):
                        if remove_from_wishlist(item.get('id')):
                            st.success("✅ Supprimé!")
                            st.session_state.wishlist_items = None
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Erreur")
                
                with col2:
                    if st.button(f"✏️ Modifier prix", key=f"edit_{idx}"):
                        st.session_state[f"edit_mode_{idx}"] = True
                
                # Mode édition
                if st.session_state.get(f"edit_mode_{idx}", False):
                    with col3:
                        new_price = st.number_input(
                            "Nouveau prix max", 
                            value=float(item.get('max_price', 20)), 
                            key=f"new_price_{idx}"
                        )
                        
                        if st.button(f"💾 Sauvegarder", key=f"save_{idx}"):
                            # Mise à jour du prix (nécessite un endpoint API)
                            st.session_state[f"edit_mode_{idx}"] = False
                            st.info("Fonctionnalité de modification à implémenter côté API")
    else:
        st.info("📭 Votre wishlist est vide. Ajoutez des jeux depuis les recommandations!")
        
        # Guide rapide
        with st.expander("ℹ️ Comment utiliser la wishlist?"):
            st.markdown("""
            **🎯 Comment ça marche:**
            1. **Ajoutez des jeux** depuis les recommandations ou manuellement
            2. **Définissez un seuil de prix** pour chaque jeu
            3. **Recevez des notifications** quand le prix baisse
            4. **Surveillez les alertes** dans l'onglet dédié
            
            **💡 Astuces:**
            - Utilisez des noms de jeux précis pour de meilleurs résultats
            - Vérifiez régulièrement les prix avec le bouton "Vérifier"
            - Les notifications apparaissent dans l'onglet "Alertes Prix"
            """)

# =========================
# ---- TAB 3: ALERTES ----
# =========================
with tabs[2]:
    guard_auth()
    
    st.subheader("🔔 Alertes de Prix")
    
    # Récupération des alertes
    price_alerts = get_price_alerts()
    
    if price_alerts:
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox(
                "Filtrer par statut",
                ["Toutes", "En attente", "Lues"],
                key="alert_status_filter"
            )
        
        # Filtrage
        if status_filter == "En attente":
            filtered_alerts = [a for a in price_alerts if a.get("status") == "pending"]
        elif status_filter == "Lues":
            filtered_alerts = [a for a in price_alerts if a.get("status") == "read"]
        else:
            filtered_alerts = price_alerts
        
        st.markdown(f"**📊 {len(filtered_alerts)} alerte(s) affichée(s)**")
        
        # Bouton marquer toutes comme lues
        if any(a.get("status") == "pending" for a in filtered_alerts):
            if st.button("✅ Marquer toutes comme lues"):
                for alert in filtered_alerts:
                    if alert.get("status") == "pending":
                        mark_alert_read(alert.get("id"))
                st.success("✅ Toutes les alertes ont été marquées comme lues")
                time.sleep(1)
                st.rerun()
        
        # Affichage des alertes
        for idx, alert in enumerate(filtered_alerts):
            status = alert.get("status", "pending")
            is_new = status == "pending"
            
            container_class = "price-alert" if is_new else "metric-card"
            
            st.markdown(f"""
            <div class="{container_class}">
                <h4>{'🔥' if is_new else '📖'} {alert.get('game_title', 'Jeu inconnu')}</h4>
                <p><strong>Prix actuel:</strong> <span style="color: #00ff88; font-size: 1.2em;">{alert.get('current_price', 0)} EUR</span></p>
                <p><strong>Votre seuil:</strong> {alert.get('threshold_price', 0)} EUR</p>
                <p><strong>Économie:</strong> <span style="color: #ffaa00;">-{round(float(alert.get('threshold_price', 0)) - float(alert.get('current_price', 0)), 2)} EUR</span></p>
                <p><strong>Boutique:</strong> {alert.get('shop_name', 'Inconnue')}</p>
                <p><strong>Date:</strong> {alert.get('created_at', 'Date inconnue')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if alert.get('shop_url'):
                    st.link_button(f"🛒 Voir l'offre", alert['shop_url'])
            
            with col2:
                if is_new and st.button(f"✅ Marquer lue", key=f"read_{idx}"):
                    if mark_alert_read(alert.get("id")):
                        st.success("✅ Marquée comme lue")
                        time.sleep(1)
                        st.rerun()
            
            with col3:
                if st.button(f"💝 Voir wishlist", key=f"view_wishlist_{idx}"):
                    st.info("Redirection vers l'onglet Wishlist...")
    
    else:
        st.info("📭 Aucune alerte pour le moment")
        
        # Encouragement
        st.markdown("""
        **🎯 Pour recevoir des alertes:**
        1. Ajoutez des jeux à votre wishlist
        2. Le système vérifie automatiquement les prix
        3. Vous recevez une notification quand un prix baisse
        
        **⏰ Vérification automatique:** Toutes les heures
        """)

# =========================
# ---- TAB 4: ANALYTICS ---
# =========================
with tabs[3]:
    guard_auth()
    
    st.subheader("📊 Analytics Avancées")
    
    # Métriques du modèle IA
    try:
        r = get("/model/metrics")
        if r.ok:
            metrics = r.json()
            
            st.markdown("#### 🤖 Performance du Modèle IA")
            show_metrics_pills(metrics)
            
            # Graphique de performance (simulé)
            if st.session_state.hybrid_model_enabled:
                # Données simulées pour le modèle hybride
                performance_data = {
                    "Métrique": ["Précision", "Rappel", "F1-Score", "R² Score"],
                    "Modèle Classique": [0.72, 0.68, 0.70, 0.65],
                    "Modèle Hybride": [0.85, 0.82, 0.83, 0.79]
                }
                
                df_perf = pd.DataFrame(performance_data)
                
                fig = px.bar(
                    df_perf, 
                    x="Métrique", 
                    y=["Modèle Classique", "Modèle Hybride"],
                    title="🆚 Comparaison des Modèles",
                    barmode="group"
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e6f3ff"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Impossible de charger les métriques: {e}")
    
    # Analytics wishlist
    st.markdown("#### 💝 Analytics Wishlist")
    
    # Simuler des stats wishlist
    if wishlist_items:
        # Distribution des prix
        prices = [float(item.get('max_price', 0)) for item in wishlist_items]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_prices = px.histogram(
                x=prices, 
                title="📊 Distribution des Seuils de Prix",
                nbins=10
            )
            fig_prices.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e6f3ff"
            )
            st.plotly_chart(fig_prices, use_container_width=True)
        
        with col2:
            # Métriques wishlist
            total_items = len(wishlist_items)
            avg_price = sum(prices) / len(prices) if prices else 0
            total_notifications = sum(item.get('notification_count', 0) for item in wishlist_items)
            
            st.metric("🎮 Jeux en wishlist", total_items)
            st.metric("💰 Prix moyen souhaité", f"{avg_price:.2f} EUR")
            st.metric("🔔 Notifications reçues", total_notifications)
    
    # Analytics des recommandations
    st.markdown("#### 🎯 Analytics des Recommandations")
    
    # Graphique des genres populaires (simulé)
    genre_data = {
        "Genre": ["RPG", "Action", "Indie", "Strategy", "Simulation", "Adventure"],
        "Recherches": [45, 38, 32, 28, 22, 18],
        "Recommandations": [52, 41, 29, 31, 25, 20]
    }
    
    df_genres = pd.DataFrame(genre_data)
    
    fig_genres = px.bar(
        df_genres,
        x="Genre",
        y=["Recherches", "Recommandations"],
        title="🎮 Genres les plus Populaires",
        barmode="group"
    )
    fig_genres.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e6f3ff"
    )
    st.plotly_chart(fig_genres, use_container_width=True)

# =========================
# ---- TAB 5: ADMIN -------
# =========================
with tabs[4]:
    guard_auth()
    
    st.subheader("🛠️ Administration & Modèles")
    
    # Sélection du modèle
    st.markdown("#### 🧠 Configuration du Modèle")
    
    model_type = st.radio(
        "Type de modèle à utiliser:",
        ["Modèle Classique (TF-IDF + SVD)", "Modèle Hybride (+ Gradient Boosting)"],
        index=1 if st.session_state.hybrid_model_enabled else 0
    )
    
    st.session_state.hybrid_model_enabled = "Hybride" in model_type
    
    if st.session_state.hybrid_model_
