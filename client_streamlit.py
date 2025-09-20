import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Game Recommendation API",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
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
    .game-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9f9f9;
    }
    .status-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
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
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Configuration API compacte en haut de page
st.markdown('<div class="main-header"><h1>🎮 Game Recommendation API</h1><p>Plateforme de recommandation de jeux</p></div>', unsafe_allow_html=True)

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
            st.write("")
            if st.button("🚀 Se connecter", type="primary"):
                if username and password:
                    if authenticate_user(username, password):
                        st.success("✅ Authentification réussie!")
                        st.rerun()
                else:
                    st.error("Veuillez remplir tous les champs")
    
    else:
        status_col1, status_col2 = st.columns([3, 1])
        
        with status_col1:
            st.success(f"🟢 Connecté et authentifié à {st.session_state.api_url}")
        
        with status_col2:
            if st.button("🚪 Déconnecter"):
                st.session_state.authenticated = False
                st.session_state.api_token = ""
                st.session_state.api_configured = False
                st.rerun()

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
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=15)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=15)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data, timeout=15)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=15)
        
        return response
        
    except Exception as e:
        st.error(f"❌ Erreur de requête: {str(e)}")
        return None

# Interface principale - seulement si authentifié
if st.session_state.authenticated:
    tab_search, tab_recommendations, tab_health = st.tabs([
        "🔍 Recherche par Titre", 
        "⭐ Recommandations ML",
        "🏥 Santé API"
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
                if response:
                    st.error(f"❌ Erreur lors de la recherche: {response.status_code}")
                    if response.text:
                        st.code(response.text)
                else:
                    st.error("❌ Aucune réponse de l'API")

    # ONGLET RECOMMANDATIONS ML
    with tab_recommendations:
        st.header("⭐ Recommandations ML")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Configuration")
            
            user_query = st.text_input("Décrivez ce que vous cherchez", value="RPG fantasy", placeholder="RPG, action, aventure...")
            num_recommendations = st.slider("Nombre de suggestions", 1, 20, 10)
            confidence_min = st.slider("Confiance minimum", 0.0, 1.0, 0.1, 0.05)
        
        with col2:
            st.subheader("Obtenir des recommandations")
            
            if st.button("🎯 Obtenir des recommandations ML", type="primary", use_container_width=True):
                recommendation_data = {
                    "query": user_query,
                    "k": num_recommendations,
                    "min_confidence": confidence_min
                }
                
                with st.spinner("Génération des recommandations ML..."):
                    response = api_call("/recommend/ml", method="POST", data=recommendation_data)
                    
                    if response and response.status_code == 200:
                        try:
                            results = response.json()
                            
                            st.success(f"✅ {len(results.get('recommendations', []))} recommandation(s) générée(s)")
                            
                            # Afficher les métriques de performance si disponibles
                            if results.get('latency_ms'):
                                st.info(f"⚡ Temps de traitement: {results['latency_ms']:.1f}ms")
                            if results.get('model_version'):
                                st.info(f"🤖 Modèle: {results['model_version']}")
                            
                            recommendations = results.get("recommendations", [])
                            
                            for rec in recommendations:
                                with st.container():
                                    st.markdown('<div class="game-card">', unsafe_allow_html=True)
                                    
                                    rec_col1, rec_col2, rec_col3 = st.columns([2, 1, 1])
                                    
                                    with rec_col1:
                                        title = rec.get('title', 'Jeu recommandé')
                                        st.write(f"**{title}**")
                                        
                                        genres = rec.get('genres', 'N/A')
                                        st.caption(f"Genres: {genres}")
                                    
                                    with rec_col2:
                                        confidence = rec.get('confidence', rec.get('score', 0))
                                        st.metric("Confiance", f"{confidence:.3f}")
                                    
                                    with rec_col3:
                                        rating = rec.get('rating', 0)
                                        st.metric("Note", f"{rating:.1f}")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.divider()
                                    
                        except json.JSONDecodeError:
                            st.error("❌ Erreur de format de réponse")
                    else:
                        if response:
                            st.error(f"❌ Erreur lors de la génération: {response.status_code}")
                            if response.text:
                                st.code(response.text)
                        else:
                            st.error("❌ Aucune réponse de l'API")

    # ONGLET SANTÉ API
    with tab_health:
        st.header("🏥 Santé de l'API")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Status général")
            
            if st.button("🔄 Vérifier le statut", type="primary"):
                response = api_call("/healthz")
                
                if response and response.status_code == 200:
                    try:
                        health_data = response.json()
                        
                        status = health_data.get("status", "unknown")
                        if status == "healthy":
                            st.success("🟢 API en bonne santé")
                        else:
                            st.warning(f"⚠️ Statut: {status}")
                        
                        # Afficher les détails
                        st.markdown('<div class="status-card">', unsafe_allow_html=True)
                        st.write(f"**Base de données:** {'✅ Connectée' if health_data.get('db_ready', False) else '❌ Déconnectée'}")
                        st.write(f"**Modèle ML:** {'✅ Chargé' if health_data.get('model_loaded', False) else '❌ Non chargé'}")
                        st.write(f"**Version modèle:** {health_data.get('model_version', 'unknown')}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        if health_data.get("db_error"):
                            st.error(f"Erreur DB: {health_data['db_error']}")
                        
                    except json.JSONDecodeError:
                        st.error("❌ Erreur de format de réponse")
                else:
                    if response:
                        st.error(f"❌ API non disponible: {response.status_code}")
                    else:
                        st.error("❌ Impossible de contacter l'API")
        
        with col2:
            st.subheader("Métriques")
            
            if st.button("📊 Récupérer les métriques"):
                response = api_call("/model/metrics")
                
                if response and response.status_code == 200:
                    try:
                        metrics = response.json()
                        
                        st.markdown('<div class="status-card">', unsafe_allow_html=True)
                        st.metric("Total prédictions", metrics.get("total_predictions", 0))
                        st.metric("Confiance moyenne", f"{metrics.get('avg_confidence', 0):.3f}")
                        st.metric("Nombre de jeux", metrics.get("games_count", 0))
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        if metrics.get("last_training"):
                            st.info(f"Dernier entraînement: {metrics['last_training']}")
                        
                    except json.JSONDecodeError:
                        st.error("❌ Erreur de format des métriques")
                else:
                    if response:
                        st.error(f"❌ Métriques non disponibles: {response.status_code}")
                    else:
                        st.error("❌ Impossible de récupérer les métriques")

else:
    st.info("👆 Veuillez vous connecter pour accéder aux fonctionnalités")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.8rem;">🎮 Game Recommendation API - Interface simplifiée (sans graphiques)</div>',
    unsafe_allow_html=True
)
