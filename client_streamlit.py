# enhanced_streamlit_client.py - Client Streamlit avec Wishlist et modèle hybride
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
# --------- CONFIG --------
# =========================
DEFAULT_API_URL = os.getenv("GAMES_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Games Reco Pro — Wishlist & Hybrid AI", 
    page_icon="🎮", 
    layout="wide"
)

# CSS Theme Arcade Neon amélioré
ENHANCED_CSS = """
<style>
/* Fond dégradé néon amélioré */
.stApp {
  background: radial-gradient(1200px 500px at 10% 0%, #081226 0%, #0b0f1d 35%, #070b14 60%, #05070d 100%) !important;
  color: #e6f3ff;
  font-family: ui-rounded, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, sans-serif;
}

/* Titres néon */
h1, h2, h3, h4 {
  color: #9be8ff !important;
  text-shadow: 0 0 12px rgba(155,232,255,.5), 0 0 24px rgba(72,163,255,.25);
}

/* Sidebar néon */
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#09132a,#0b1733) !important;
  border-right: 1px solid rgba(255,255,255,.08);
}

/* Onglets futuristes */
.stTabs [data-baseweb="tab"] {
  color: #cbe7ff !important;
  font-weight: 700 !important;
  background: rgba(255,255,255,0.02) !important;
  border-radius: 8px 8px 0 0 !important;
  margin-right: 4px !important;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(155,232,255,0.1) !important;
  transform: translateY(-1px);
}

.stTabs [data-baseweb="tab-highlight"] {
  background: linear-gradient(90deg, #00eaff33, #9d00ff33) !important;
}

/* Boutons arcade améliorés */
.stButton>button {
  background: linear-gradient(90deg, #00eaff, #9d00ff) !important;
  color: #081226 !important;
  border: 0 !important;
  border-radius: 14px !important;
  box-shadow: 0 8px 24px rgba(0,234,255,.25);
  font-weight: 800 !important;
  transition: all 0.3s ease !important;
}

.stButton>button:hover {
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 12px 32px rgba(0,234,255,.4) !important;
}

/* Cards wishlist */
.wishlist-card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(155,232,255,.15);
  border-radius: 16px;
  padding: 20px;
  margin: 12px 0;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
}

.wishlist-card:hover {
  border-color: rgba(155,232,255,.3);
  background: rgba(255,255,255,0.08);
  transform: translateY(-2px);
}

/* Badges prix */
.price-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 20px;
  font-weight: 800;
  font-size: 14px;
  margin: 4px;
}

.price-target {
  background: linear-gradient(90deg, #00ffa3, #00d4ff);
  color: #081226;
}

.price-current {
  background: linear-gradient(90deg, #ff6b6b, #feca57);
  color: #081226;
}

.price-alert {
  background: linear-gradient(90deg, #00ff87, #60efff);
  color: #081226;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.7; }
  100% { opacity: 1; }
}

/* Notifications */
.notification-card {
  background: rgba(0,255,135,0.1);
  border: 1px solid rgba(0,255,135,0.3);
  border-radius: 12px;
  padding: 16px;
  margin: 8px 0;
  position: relative;
}

.notification-card::before {
  content: "🔔";
  position: absolute;
  top: 8px;
  right: 12px;
  font-size: 16px;
}

/* Métriques en temps réel */
.metric-realtime {
  background: rgba(157,0,255,0.1);
  border: 1px solid rgba(157,0,255,0.3);
  border-radius: 12px;
  padding: 12px;
  text-align: center;
  min-height: 80px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

/* Inputs futuristes */
.stTextInput input, .stNumberInput input, .stSelectbox select {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(155,232,255,0.2) !important;
  border-radius: 8px !important;
  color: #e6f3ff !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
  border-color: rgba(155,232,255,0.5) !important;
  box-shadow: 0 0 12px rgba(155,232,255,0.3) !important;
}

/* Status indicators */
.status-online {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: #00ff87;
  border-radius: 50%;
  margin-right: 8px;
  animation: pulse 2s infinite;
}

.status-error {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: #ff6b6b;
  border-radius: 50%;
  margin-right: 8px;
}
</style>
"""

st.markdown(ENHANCED_CSS, unsafe_allow_html=True)

# =========================
# -------- STATE ----------
# =========================
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "wishlist_items" not in st.session_state:
    st.session_state.wishlist_items = []

# =========================
# ------- HELPERS ---------
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

def delete(path: str, timeout: int = 25) -> requests.Response:
    url = f"{api_base()}{path}"
    return requests.delete(url, headers=bearer_headers(), timeout=timeout)

def put(path: str, payload: Dict[str, Any], timeout: int = 25) -> requests.Response:
    url = f"{api_base()}{path}"
    return requests.put(url, headers=bearer_headers(), data=json.dumps(payload), timeout=timeout)

def login(username: str, password: str) -> bool:
    try:
        r = requests.post(f"{api_base()}/token",
                          data={"username": username, "password": password},
                          timeout=20)
        if r.status_code == 200:
            data = r.json()
            st.session_state.token = data.get("access_token")
            st.session_state.username = username
            return True
        else:
            st.error(f"⛔ Login échoué: {r.status_code} — {r.text}")
            return False
    except Exception as e:
        st.error(f"⚠️ Erreur réseau login: {e}")
        return False

def guard_auth():
    if not st.session_state.token:
        st.warning("🔐 Tu dois être connecté pour utiliser l'API.")
        st.stop()

def load_user_data():
    """Charge les données utilisateur (wishlist + notifications)"""
    if not st.session_state.token:
        return
    
    try:
        # Charger wishlist
        r_wishlist = get("/wishlist")
        if r_wishlist.status_code == 200:
            st.session_state.wishlist_items = r_wishlist.json()
        
        # Charger notifications
        r_notifications = get("/wishlist/notifications")
        if r_notifications.status_code == 200:
            st.session_state.notifications = r_notifications.json()
    
    except Exception as e:
        st.error(f"Erreur chargement données: {e}")

def show_api_status():
    """Affiche le statut de l'API avec indicateur visuel"""
    try:
        resp = requests.get(f"{api_base()}/healthz", timeout=5)
        if resp.ok:
            data = resp.json()
            status = data.get("status", "unknown")
            if status == "healthy":
                st.markdown('<span class="status-online"></span>**API Online**', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-error"></span>**API Degraded**', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-error"></span>**API Error**', unsafe_allow_html=True)
    except:
        st.markdown('<span class="status-error"></span>**API Offline**', unsafe_allow_html=True)

# =========================
# ------- SIDEBAR --------
# =========================
with st.sidebar:
    st.markdown("## 🎮 Games API Pro")
    st.text_input("Base API URL", key="api_url", help="Ex: http://localhost:8000")
    
    show_api_status()
    st.divider()

    if st.session_state.token:
        st.markdown(f"**Connecté:** `{st.session_state.username}`")
        
        # Afficher le nombre de notifications non lues
        unread_count = len([n for n in st.session_state.notifications if not n.get('is_read', True)])
        if unread_count > 0:
            st.markdown(f"🔔 **{unread_count}** notifications non lues")
        
        if st.button("🚪 Se déconnecter"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.wishlist_items = []
            st.session_state.notifications = []
            st.rerun()
    else:
        st.markdown("### 🔐 Authentification")
        auth_tab = st.tabs(["Se connecter", "Créer compte"])
        
        with auth_tab[1]:
            u2 = st.text_input("Username (new)", key="reg_user")
            p2 = st.text_input("Password (new)", type="password", key="reg_pass")
            if st.button("🆕 Register"):
                try:
                    r = requests.post(f"{api_base()}/register",
                                    data={"username": u2, "password": p2})
                    if r.status_code == 200:
                        st.success("✅ Compte créé ! Connecte-toi maintenant.")
                    else:
                        st.error(f"⛔ {r.status_code} — {r.text}")
                except Exception as e:
                    st.error(f"⚠️ Erreur: {e}")

    if st.session_state.token:
        st.divider()
        if st.button("🔄 Actualiser données"):
            load_user_data()
            st.rerun()

# Charger les données au démarrage
if st.session_state.token and not st.session_state.wishlist_items:
    load_user_data()

# =========================
# --------- MAIN ----------
# =========================
st.title("🕹️ Games Recommender Pro — **Hybrid AI + Wishlist**")

tabs = st.tabs([
    "🔎 Recommandations Hybrides",
    "💝 Ma Wishlist", 
    "🔔 Notifications",
    "📊 Analytics Modèle",
    "🛠️ Admin"
])

# =========================
# --- TAB 1: RECOS AI ----
# =========================
with tabs[0]:
    guard_auth()
    
    st.subheader("🤖 Recommandations IA Hybride")
    st.markdown("*Modèle combinant Content-Based + Collaborative + Gradient Boosting*")
    
    with st.form("form_hybrid_reco", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Décris ton envie", 
                                value="Action RPG fantasy open world", 
                                placeholder="Ex: Indie platformer, Strategy medieval, Horror survival...")
        with col2:
            k = st.slider("Nb résultats", 1, 30, 10)
        
        col3, col4 = st.columns(2)
        with col3:
            min_conf = st.slider("Confiance min", 0.0, 1.0, 0.10, 0.01)
        with col4:
            algorithm = st.selectbox("Algorithme", [
                "hybrid_ensemble", "content_only", "collaborative_only", "gradient_boosting_only"
            ])
        
        submitted = st.form_submit_button("⚡ Recommander avec IA")
    
    if submitted:
        with st.spinner("🧠 IA hybride en cours..."):
            payload = {
                "query": query, 
                "k": k, 
                "min_confidence": min_conf,
                "algorithm": algorithm
            }
            
            r = post("/recommend/hybrid", payload)
            if r.ok:
                data = r.json()
                st.success(f"✅ {len(data.get('recommendations',[]))} résultats — {round(data.get('latency_ms',0),1)} ms")
                
                # Afficher la breakdown des scores
                if data.get('model_version'):
                    st.info(f"🧠 Modèle: {data['model_version']} | Algorithme: {algorithm}")
                
                for idx, g in enumerate(data.get("recommendations", []), start=1):
                    with st.container():
                        st.markdown(f"### {idx}. {g.get('title','?')}")
                        
                        # Métriques principales
                        cols = st.columns(5)
                        cols[0].metric("Score Global", f"{g.get('confidence',0):.3f}")
                        cols[1].metric("Rating", f"{g.get('rating',0):.1f}/5")
                        cols[2].metric("Metacritic", f"{g.get('metacritic',0)}")
                        
                        # Breakdown des scores par composant
                        if 'prediction_breakdown' in g:
                            breakdown = g['prediction_breakdown']
                            cols[3].metric("Content", f"{breakdown.get('content_based',0):.3f}")
                            cols[4].metric("Collaborative", f"{breakdown.get('collaborative',0):.3f}")
                            
                            # Graphique radar des composants
                            if algorithm == "hybrid_ensemble":
                                fig = go.Figure()
                                
                                categories = ['Content-Based', 'Collaborative', 'Gradient Boosting']
                                values = [
                                    breakdown.get('content_based', 0),
                                    breakdown.get('collaborative', 0), 
                                    breakdown.get('gradient_boosting', 0)
                                ]
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=categories,
                                    fill='toself',
                                    name=g['title'][:20],
                                    line_color='cyan'
                                ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(visible=True, range=[0, 1])
                                    ),
                                    showlegend=True,
                                    height=300,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Genres et plateformes
                        if g.get("genres"):
                            st.markdown(f"🏷️ **Genres:** {g['genres']}")
                        
                        # Bouton ajout à la wishlist
                        col_add, col_space = st.columns([1, 3])
                        with col_add:
                            if st.button(f"💝 Ajouter à la wishlist", key=f"add_wishlist_{idx}"):
                                st.session_state[f"add_to_wishlist_game"] = g['title']
                                st.rerun()
                        
                        st.divider()
            else:
                st.error(f"⛔ {r.status_code} — {r.text}")
    
    # Comparaison des algorithmes
    st.subheader("🔬 Comparaison des Algorithmes")
    if st.button("🧪 Tester tous les algorithmes"):
        test_query = query if 'query' in locals() else "Action RPG"
        
        algorithms = ["content_only", "collaborative_only", "gradient_boosting_only", "hybrid_ensemble"]
        comparison_data = []
        
        for algo in algorithms:
            try:
                r = post("/recommend/hybrid", {
                    "query": test_query, 
                    "k": 5, 
                    "algorithm": algo
                })
                if r.ok:
                    data = r.json()
                    comparison_data.append({
                        "Algorithm": algo.replace("_", " ").title(),
                        "Results": len(data.get("recommendations", [])),
                        "Latency (ms)": round(data.get("latency_ms", 0), 1),
                        "Avg Confidence": round(sum(rec.get("confidence", 0) for rec in data.get("recommendations", [])) / max(1, len(data.get("recommendations", []))), 3)
                    })
            except Exception as e:
                st.error(f"Erreur {algo}: {e}")
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Graphique de comparaison
            fig = px.bar(df_comparison, x="Algorithm", y=["Latency (ms)", "Avg Confidence"], 
                        title="Comparaison Performance Algorithmes",
                        color_discrete_sequence=['#00eaff', '#9d00ff'])
            st.plotly_chart(fig, use_container_width=True)

# =========================
# --- TAB 2: WISHLIST ----
# =========================
with tabs[1]:
    guard_auth()
    
    st.subheader("💝 Ma Wishlist — Surveillance des Prix")
    
    # Ajouter un jeu à la wishlist
    with st.expander("➕ Ajouter un jeu à surveiller", expanded=False):
        with st.form("add_wishlist_form"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                game_title = st.text_input("Nom du jeu", 
                                         value=st.session_state.get("add_to_wishlist_game", ""),
                                         placeholder="Ex: Cyberpunk 2077, The Witcher 3...")
            with col2:
                target_price = st.number_input("Prix cible (€)", min_value=0.01, max_value=200.0, 
                                             value=19.99, step=0.01)
            with col3:
                currency = st.selectbox("Devise", ["EUR", "USD", "GBP"])
            
            if st.form_submit_button("💝 Ajouter à la surveillance"):
                try:
                    payload = {
                        "game_title": game_title,
                        "target_price": target_price,
                        "price_currency": currency
                    }
                    r = post("/wishlist", payload)
                    if r.status_code == 201:
                        st.success(f"✅ '{game_title}' ajouté à la wishlist !")
                        load_user_data()
                        st.rerun()
                    else:
                        st.error(f"⛔ Erreur: {r.text}")
                except Exception as e:
                    st.error(f"⚠️ Erreur: {e}")
    
    # Afficher la wishlist
    if st.session_state.wishlist_items:
        st.markdown(f"**{len(st.session_state.wishlist_items)} jeux surveillés**")
        
        # Métriques globales wishlist
        active_items = [item for item in st.session_state.wishlist_items if item.get('is_active', True)]
        total_target_value = sum(item.get('target_price', 0) for item in active_items)
        alerts_count = sum(1 for item in active_items if item.get('alert_triggered', False))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-realtime">', unsafe_allow_html=True)
            st.metric("Jeux Actifs", len(active_items))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-realtime">', unsafe_allow_html=True)
            st.metric("Valeur Cible", f"€{total_target_value:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-realtime">', unsafe_allow_html=True)
            st.metric("Alertes Actives", alerts_count)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-realtime">', unsafe_allow_html=True)
            avg_target = total_target_value / len(active_items) if active_items else 0
            st.metric("Prix Moyen", f"€{avg_target:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Liste des items de wishlist
        for item in st.session_state.wishlist_items:
            with st.container():
                st.markdown('<div class="wishlist-card">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"### 🎮 {item['game_title']}")
                    created_date = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
                    st.caption(f"Ajouté le {created_date.strftime('%d/%m/%Y')}")
                
                with col2:
                    st.markdown(f'<div class="price-badge price-target">Cible: €{item["target_price"]:.2f}</div>', 
                              unsafe_allow_html=True)
                
                with col3:
                    current_price = item.get('current_price')
                    if current_price:
                        price_class = "price-alert" if current_price <= item['target_price'] else "price-current"
                        st.markdown(f'<div class="price-badge {price_class}">Actuel: €{current_price:.2f}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Calcul différence de prix
                        diff = current_price - item['target_price']
                        if diff <= 0:
                            st.markdown(f"🎯 **OBJECTIF ATTEINT** — Économie: €{abs(diff):.2f}")
                        else:
                            st.markdown(f"📈 Plus cher de €{diff:.2f}")
                    else:
                        st.markdown('<div class="price-badge">Prix: Non trouvé</div>', unsafe_allow_html=True)
                
                with col4:
                    # Actions
                    if st.button(f"✏️", key=f"edit_{item['id']}", help="Modifier"):
                        st.session_state[f"edit_wishlist_{item['id']}"] = True
                    
                    if st.button(f"🗑️", key=f"delete_{item['id']}", help="Supprimer"):
                        try:
                            r = delete(f"/wishlist/{item['id']}")
                            if r.status_code == 200:
                                st.success("✅ Supprimé !")
                                load_user_data()
                                st.rerun()
                            else:
                                st.error("❌ Erreur suppression")
                        except Exception as e:
                            st.error(f"❌ {e}")
                
                # Formulaire d'édition (conditionnel)
                if st.session_state.get(f"edit_wishlist_{item['id']}", False):
                    with st.form(f"edit_form_{item['id']}"):
                        new_price = st.number_input("Nouveau prix cible", 
                                                  value=float(item['target_price']),
                                                  min_value=0.01, max_value=200.0, step=0.01)
                        active = st.checkbox("Actif", value=item.get('is_active', True))
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.form_submit_button("💾 Sauvegarder"):
                                try:
                                    payload = {"target_price": new_price, "is_active": active}
                                    r = put(f"/wishlist/{item['id']}", payload)
                                    if r.status_code == 200:
                                        st.success("✅ Modifié !")
                                        del st.session_state[f"edit_wishlist_{item['id']}"]
                                        load_user_data()
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"❌ {e}")
                        
                        with col_cancel:
                            if st.form_submit_button("❌ Annuler"):
                                del st.session_state[f"edit_wishlist_{item['id']}"]
                                st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Bouton de vérification manuelle des prix
        if st.button("🔄 Vérifier les prix maintenant"):
            with st.spinner("🔍 Vérification des prix en cours..."):
                try:
                    r = post("/wishlist/check-prices", {})
                    if r.status_code == 200:
                        result = r.json()
                        alerts_created = result.get('alerts_created', 0)
                        if alerts_created > 0:
                            st.success(f"✅ {alerts_created} nouvelles alertes créées !")
                        else:
                            st.info("ℹ️ Aucune nouvelle alerte")
                        load_user_data()
                        st.rerun()
                    else:
                        st.error("❌ Erreur vérification prix")
                except Exception as e:
                    st.error(f"❌ {e}")
    else:
        st.info("📝 Aucun jeu dans votre wishlist. Ajoutez-en un pour commencer la surveillance des prix !")

# =========================
# --- TAB 3: NOTIFICATIONS 
# =========================
with tabs[2]:
    guard_auth()
    
    st.subheader("🔔 Notifications de Prix")
    
    if st.session_state.notifications:
        # Statistiques notifications
        unread_notifications = [n for n in st.session_state.notifications if not n.get('is_read', True)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Notifications totales", len(st.session_state.notifications))
        with col2:
            st.metric("Non lues", len(unread_notifications))
        
        # Bouton marquer toutes comme lues
        if unread_notifications:
            if st.button("✅ Marquer toutes comme lues"):
                for notif in unread_notifications:
                    try:
                        r = put(f"/wishlist/notifications/{notif['id']}/read", {})
                    except:
                        pass
                load_user_data()
                st.rerun()
        
        st.divider()
        
        # Afficher les notifications
        for notif in st.session_state.notifications:
            with st.container():
                css_class = "notification-card" if not notif.get('is_read', True) else "wishlist-card"
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{notif['game_title']}**")
                    st.markdown(notif['message'])
                    
                    created_date = datetime.fromisoformat(notif['created_at'].replace('Z', '+00:00'))
                    st.caption(f"📅 {created_date.strftime('%d/%m/%Y %H:%M')}")
                
                with col2:
                    st.markdown(f'<div class="price-badge price-alert">€{notif["price"]:.2f}</div>', 
                              unsafe_allow_html=True)
                
                with col3:
                    if not notif.get('is_read', True):
                        if st.button("✅ Marquer lu", key=f"read_{notif['id']}"):
                            try:
                                r = put(f"/wishlist/notifications/{notif['id']}/read", {})
                                if r.status_code == 200:
                                    load_user_data()
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ {e}")
                    else:
                        st.markdown("✅ **Lu**")
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🔕 Aucune notification pour le moment.")

# =========================
# --- TAB 4: ANALYTICS ----
# =========================
with tabs[3]:
    guard_auth()
    
    st.subheader("📊 Analytics du Modèle Hybride")
    
    # Métriques du modèle
    try:
        r = get("/model/hybrid/info")
        if r.ok:
            model_info = r.json()
            
            # Vue d'ensemble
            st.markdown("### 🧠 Vue d'ensemble du modèle")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Version", model_info.get('model_version', 'Unknown'))
                st.metric("Entraîné", "✅" if model_info.get('is_trained', False) else "❌")
            
            with col2:
                metrics = model_info.get('metrics', {})
                st.metric("Score Combiné", f"{metrics.get('combined_score', 0):.3f}")
                st.metric("Prédictions Totales", metrics.get('total_predictions', 0))
            
            with col3:
                st.metric("Temps Entraînement", f"{metrics.get('training_time', 0):.1f}s")
                st.metric("Temps Prédiction Moy", f"{metrics.get('avg_prediction_time', 0)*1000:.1f}ms")
            
            # Détails des composants
            st.markdown("### 🔧 Composants du Modèle")
            components = model_info.get('components', {})
            
            # Content-Based
            if 'content_based' in components:
                with st.expander("📝 Content-Based (TF-IDF + SVD)"):
                    cb = components['content_based']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Features TF-IDF", cb.get('vectorizer_features', 0))
                    with col2:
                        st.metric("Composantes SVD", cb.get('svd_components', 0))
                    
                    st.metric("Précision Content", f"{metrics.get('content_accuracy', 0):.3f}")
            
            # Collaborative
            if 'collaborative' in components:
                with st.expander("👥 Collaborative Filtering"):
                    collab = components['collaborative']
                    if collab.get('user_features_shape'):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Users Features", f"{collab['user_features_shape'][0]}x{collab['user_features_shape'][1]}")
                        with col2:
                            st.metric("Items Features", f"{collab['item_features_shape'][0]}x{collab['item_features_shape'][1]}")
                    
                    st.metric("Précision Collaborative", f"{metrics.get('collaborative_accuracy', 0):.3f}")
            
            # Gradient Boosting
            if 'gradient_boosting' in components:
                with st.expander("🚀 Gradient Boosting"):
                    gb = components['gradient_boosting']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("N Estimators", gb.get('n_estimators', 0))
                    with col2:
                        st.metric("Max Depth", gb.get('max_depth', 0))
                    with col3:
                        st.metric("Learning Rate", gb.get('learning_rate', 0))
                    
                    st.metric("R² Score", f"{metrics.get('gb_r2_score', 0):.3f}")
            
            # Poids d'ensemble
            st.markdown("### ⚖️ Poids d'Ensemble")
            ensemble_weights = model_info.get('ensemble_weights', {})
            
            weights_df = pd.DataFrame([
                {"Composant": "Content-Based", "Poids": ensemble_weights.get('content', 0)},
                {"Composant": "Collaborative", "Poids": ensemble_weights.get('collaborative', 0)},
                {"Composant": "Gradient Boosting", "Poids": ensemble_weights.get('gradient_boosting', 0)}
            ])
            
            fig = px.pie(weights_df, values='Poids', names='Composant', 
                        title="Répartition des Poids dans l'Ensemble",
                        color_discrete_sequence=['#00eaff', '#9d00ff', '#00ffa3'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance dans le temps
            st.markdown("### 📈 Performance dans le Temps")
            # Ici on pourrait ajouter des métriques temporelles
            st.info("📊 Les métriques temporelles seront ajoutées avec plus de données d'usage.")
            
        else:
            st.error("❌ Impossible de récupérer les informations du modèle")
    
    except Exception as e:
        st.error(f"❌ Erreur: {e}")

# =========================
# ---- TAB 5: ADMIN ------
# =========================
with tabs[4]:
    guard_auth()
    
    st.subheader("🛠️ Administration")
    
    # Entraînement du modèle hybride
    with st.expander("🚀 Entraînement Modèle Hybride"):
        col1, col2 = st.columns(2)
        with col1:
            model_version = st.text_input("Version modèle", value="3.0.0-hybrid")
            force_retrain = st.checkbox("Forcer le re-entraînement")
        
        with col2:
            ensemble_content = st.slider("Poids Content-Based", 0.0, 1.0, 0.4, 0.1)
            ensemble_collab = st.slider("Poids Collaborative", 0.0, 1.0, 0.3, 0.1)
            ensemble_gb = 1.0 - ensemble_content - ensemble_collab
            st.write(f"Poids Gradient Boosting: {ensemble_gb:.1f}")
        
        if st.button("🚀 Entraîner Modèle Hybride"):
            with st.spinner("🧠 Entraînement en cours..."):
                payload = {
                    "version": model_version,
                    "force_retrain": force_retrain,
                    "ensemble_weights": {
                        "content": ensemble_content,
                        "collaborative": ensemble_collab,
                        "gradient_boosting": ensemble_gb
                    }
                }
                try:
                    r = post("/model/train-hybrid", payload)
                    if r.ok:
                        result = r.json()
                        st.success(f"✅ Entraînement terminé en {result.get('training_time', 0):.2f}s")
                        
                        # Afficher les métriques d'entraînement
                        if 'content_metrics' in result:
                            st.json(result)
                    else:
                        st.error(f"❌ {r.status_code} — {r.text}")
                except Exception as e:
                    st.error(f"❌ {e}")
    
    # Gestion wishlist
    with st.expander("💝 Gestion Wishlist"):
        if st.button("🔄 Vérification massive des prix"):
            try:
                r = post("/admin/check-all-prices", {})
                if r.ok:
                    result = r.json()
                    st.success(f"✅ {result.get('alerts_created', 0)} alertes créées")
                else:
                    st.error("❌ Erreur")
            except Exception as e:
                st.error(f"❌ {e}")
        
        if st.button("🧹 Nettoyer anciennes notifications"):
            try:
                r = post("/admin/cleanup-notifications", {"days_old": 30})
                if r.ok:
                    result = r.json()
                    st.success(f"✅ {result.get('deleted', 0)} notifications supprimées")
                else:
                    st.error("❌ Erreur")
            except Exception as e:
                st.error(f"❌ {e}")
    
    # Health & Debug
    with st.expander("🩺 Health & Debug"):
        if st.button("🔎 Health Check Complet"):
            try:
                r = get("/healthz")
                if r.ok:
                    health_data = r.json()
                    st.json(health_data)
                else:
                    st.error(f"❌ {r.status_code}")
            except Exception as e:
                st.error(f"❌ {e}")

st.markdown("---")
st.caption("🎮 Games Recommender Pro — Hybrid AI with Wishlist & Notifications | Appuyez sur **R** pour actualiser") auth_tab[0]:
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            if st.button("🎯 Login"):
                if login(u, p):
                    st.success("✅ Connecté !")
                    load_user_data()
                    st.rerun()
        
        with
