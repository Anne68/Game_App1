import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page (une seule fois)
st.set_page_config(
    page_title="Game Recommendation API",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour une interface plus épurée
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .status-ok {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 4px;
        text-align: center;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 4px;
        text-align: center;
    }
    .debug-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .game-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# État de session pour la configuration API
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'api_url' not in st.session_state:
    st.session_state.api_url = "https://game-app-y8be.onrender.com"
if 'api_token' not in st.session_state:
    st.session_state.api_token = ""
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Configuration API compacte en haut de page
st.markdown('<div class="main-header"><h1>🎮 Game Recommendation API</h1><p>Plateforme avancée de recommandation avec modèles IA multiples</p></div>', unsafe_allow_html=True)

# Fonction d'authentification
def authenticate_user(username: str, password: str):
    try:
        response = requests.post(
            f"{st.session_state.api_url}/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        
        if response.status_code == 200:
            token_data = response.json()
            st.session_state.api_token = token_data["access_token"]
            st.session_state.authenticated = True
            return True
        else:
            st.error(f"Erreur d'authentification: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Erreur de connexion: {str(e)}")
        return False

# Configuration API et authentification
with st.container():
    if not st.session_state.api_configured:
        st.info("🔧 Configuration de l'API de jeux")
        
        config_col1, config_col2 = st.columns([1, 1])
        
        with config_col1:
            api_url = st.text_input("🌐 URL API", value="https://game-app-y8be.onrender.com")
        
        with config_col2:
            if st.button("✅ Connecter à l'API", type="primary"):
                st.session_state.api_url = api_url.rstrip('/')
                st.session_state.api_configured = True
                st.rerun()
    
    elif not st.session_state.authenticated:
        st.warning("🔐 Authentification requise")
        
        auth_col1, auth_col2, auth_col3 = st.columns([1, 1, 1])
        
        with auth_col1:
            username = st.text_input("👤 Nom d'utilisateur", placeholder="Votre nom d'utilisateur")
        
        with auth_col2:
            password = st.text_input("🔑 Mot de passe", type="password", placeholder="Votre mot de passe")
        
        with auth_col3:
            st.write("")  # Espace
            if st.button("🚀 Se connecter", type="primary"):
                if username and password:
                    if authenticate_user(username, password):
                        st.success("✅ Authentification réussie!")
                        st.rerun()
                else:
                    st.error("Veuillez remplir tous les champs")
    
    else:
        status_col1, status_col2, status_col3, status_col4 = st.columns([2, 1, 1, 1])
        
        with status_col1:
            st.success(f"🟢 Connecté et authentifié à {st.session_state.api_url}")
        
        with status_col2:
            if st.button("🔄 Test Health"):
                try:
                    response = requests.get(f"{st.session_state.api_url}/healthz", timeout=10)
                    if response.status_code == 200:
                        health_data = response.json()
                        if health_data.get("status") == "healthy":
                            st.success("✅ API OK")
                        else:
                            st.warning("⚠️ API dégradée")
                    else:
                        st.error(f"❌ {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
        
        with status_col3:
            if st.button("🐛 Debug"):
                st.session_state.debug_mode = not st.session_state.debug_mode
                st.rerun()
        
        with status_col4:
            if st.button("🚪 Déconnecter"):
                st.session_state.authenticated = False
                st.session_state.api_token = ""
                st.session_state.api_configured = False
                st.rerun()

# Mode debug
if st.session_state.debug_mode and st.session_state.authenticated:
    with st.expander("🐛 Informations de debug", expanded=True):
        st.markdown("**Configuration actuelle:**")
        st.code(f"""
URL API: {st.session_state.api_url}
Token: {st.session_state.api_token[:20]}...{st.session_state.api_token[-10:] if st.session_state.api_token else 'Non défini'}
Authenticated: {st.session_state.authenticated}
        """)

st.markdown("---")

# Fonction API avec authentification
def api_call(endpoint, method="GET", data=None, params=None):
    if not st.session_state.authenticated:
        st.warning("Authentification requise")
        return None
    
    url = f"{st.session_state.api_url}{endpoint}"
    headers = {
        "Authorization": f"Bearer {st.session_state.api_token}",
        "Content-Type": "application/json"
    }
    
    # Debug info
    if st.session_state.debug_mode:
        st.markdown(f'<div class="debug-info">🔍 {method} {url}</div>', unsafe_allow_html=True)
        if params:
            st.markdown(f'<div class="debug-info">📋 Params: {params}</div>', unsafe_allow_html=True)
        if data:
            st.markdown(f'<div class="debug-info">📦 Data: {json.dumps(data, indent=2)}</div>', unsafe_allow_html=True)
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=15)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=15)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data, timeout=15)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=15)
        
        # Debug de la réponse
        if st.session_state.debug_mode:
            st.markdown(f'<div class="debug-info">📡 Status: {response.status_code}</div>', unsafe_allow_html=True)
            if response.status_code == 401:
                st.error("Token expiré - reconnectez-vous")
                st.session_state.authenticated = False
                st.rerun()
            if response.status_code not in [200, 201]:
                st.markdown(f'<div class="debug-info">❌ Erreur: {response.text[:500]}</div>', unsafe_allow_html=True)
        
        return response
        
    except Exception as e:
        st.error(f"❌ Erreur de requête: {str(e)}")
        return None

# Interface principale - seulement si authentifié
if st.session_state.authenticated:
    tab_search, tab_recommendations, tab_model, tab_analytics = st.tabs([
        "🔍 Recherche par Titre", 
        "⭐ Recommandations ML", 
        "🤖 Gestion Modèle",
        "📊 Analytics"
    ])

    # ONGLET RECHERCHE PAR TITRE
    with tab_search:
        st.header("🔍 Recherche de jeux par titre")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("Rechercher un jeu", placeholder="cyberpunk, zelda, mario...", value="cyberpunk")
        
        with col2:
            search_limit = st.slider("Nombre de résultats", 1, 25, 10)
        
        if st.button("🔍 Rechercher par titre", type="primary", use_container_width=True):
            response = api_call(f"/recommend/by-title/{search_query}", params={"k": search_limit})
            
            if response and response.status_code == 200:
                try:
                    results = response.json()
                    
                    st.success(f"✅ Recommandations basées sur le titre '{results.get('title', search_query)}'")
                    
                    recommendations = results.get("recommendations", [])
                    
                    if recommendations:
                        for game in recommendations:
                            with st.container():
                                st.markdown('<div class="game-card">', unsafe_allow_html=True)
                                
                                game_col1, game_col2, game_col3 = st.columns([3, 1, 1])
                                
                                with game_col1:
                                    title = game.get('title', 'Titre inconnu')
                                    st.write(f"**{title}**")
                                    
                                    genres = game.get('genres', 'N/A')
                                    platforms = game.get('platforms', 'N/A')
                                    st.caption(f"Genres: {genres}")
                                    if platforms != 'N/A':
                                        st.caption(f"Plateformes: {platforms}")
                                
                                with game_col2:
                                    rating = game.get('rating', 0)
                                    st.metric("Note", f"{rating:.1f}/5")
                                
                                with game_col3:
                                    similarity = game.get('similarity_score', game.get('confidence', 0))
                                    st.metric("Similarité", f"{similarity:.2f}")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.divider()
                    else:
                        st.warning("Aucune recommandation trouvée")
                        
                except json.JSONDecodeError:
                    st.error("❌ Erreur de format de réponse")
            else:
                st.error("❌ Erreur lors de la recherche")

    # Autres onglets (Recommandations ML, Gestion Modèle, Analytics) restent inchangés...

else:
    st.info("👆 Veuillez vous connecter pour accéder aux fonctionnalités")

# Footer
st.markdown("---")
if st.session_state.debug_mode:
    api_base_url = st.session_state.api_url if st.session_state.api_url else "Non défini"
    
    with st.container():
        st.markdown("🔗 **Liens utiles:**")
        if api_base_url != "Non défini":
            st.markdown(f"- [📖 API Documentation]({api_base_url}/docs)")
            st.markdown(f"- [📊 Métriques Prometheus]({api_base_url}/metrics)")
        else:
            st.markdown("- 📖 API Documentation (URL non configurée)")
            st.markdown("- 📊 Métriques Prometheus (URL non configurée)")
    
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.8rem;">🐛 Mode debug activé | 🎮 Game Recommendation API - Interface de test</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.8rem;">🎮 Game Recommendation API - Interface simplifiée</div>',
        unsafe_allow_html=True
    )
