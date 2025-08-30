import os
import time
import requests
import streamlit as st
from typing import Dict, List, Optional
import json

# Configuration
API_BASE_URL = (
    st.secrets.get("API_URL") if hasattr(st, 'secrets') and st.secrets else
    os.getenv("API_URL") or
    "https://game-app-y8be.onrender.com"
)

st.set_page_config(
    page_title="Games UI", 
    page_icon="🎮", 
    layout="wide"
)

# Session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'favorites' not in st.session_state:
    st.session_state.favorites = set()

# API Helper
def api_request(method: str, path: str, **kwargs) -> tuple:
    """Fonction pour requêtes API avec gestion d'erreur"""
    url = f"{API_BASE_URL}{path}"
    headers = {"accept": "application/json"}
    
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=30, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, timeout=60, **kwargs)
        else:
            return False, f"Méthode non supportée: {method}"
        
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = f"Erreur HTTP {response.status_code}"
            return False, error_detail
            
        return True, response.json()
        
    except requests.exceptions.Timeout:
        return False, "Timeout - L'API met trop de temps à répondre"
    except requests.exceptions.ConnectionError:
        return False, "Erreur de connexion - Vérifiez que l'API est accessible"
    except Exception as e:
        return False, f"Erreur: {str(e)}"

# Header
st.title("🎮 Games Recommendation System")
st.markdown(f"**API Endpoint:** {API_BASE_URL}")

# Vérification API
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Test API"):
        success, result = api_request("GET", "/healthz")
        if success:
            st.success("API Active")
        else:
            st.error("API Indisponible")

# Sidebar - Auth
with st.sidebar:
    st.header("👤 Authentification")
    
    if st.session_state.token:
        st.success(f"Connecté: {st.session_state.username}")
        if st.button("Se déconnecter"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
    else:
        st.subheader("Connexion")
        
        with st.form("login"):
            username = st.text_input("Nom d'utilisateur", placeholder="demo")
            password = st.text_input("Mot de passe", type="password", placeholder="demo123")
            
            if st.form_submit_button("Se connecter"):
                if username and password:
                    success, result = api_request("POST", "/token", data={
                        "username": username,
                        "password": password
                    })
                    
                    if success:
                        st.session_state.token = result["access_token"]
                        st.session_state.username = username
                        st.success("Connexion réussie!")
                        st.rerun()
                    else:
                        st.error(f"Erreur: {result}")

        st.subheader("Inscription")
        with st.form("register"):
            reg_user = st.text_input("Nom d'utilisateur")
            reg_pass = st.text_input("Mot de passe", type="password")
            
            if st.form_submit_button("S'inscrire"):
                if reg_user and reg_pass:
                    success, result = api_request("POST", "/register", data={
                        "username": reg_user,
                        "password": reg_pass
                    })
                    
                    if success:
                        st.success("Inscription réussie!")
                    else:
                        st.error(f"Erreur: {result}")

# Main tabs
if st.session_state.token:
    tab1, tab2, tab3 = st.tabs(["🔍 Recherche", "✨ Recommandations", "📊 Monitoring"])
    
    with tab1:
        st.subheader("Recherche de jeux")
        
        search_query = st.text_input("Rechercher un jeu", placeholder="Mario, Zelda...")
        
        if st.button("Rechercher") and search_query:
            with st.spinner("Recherche..."):
                success, result = api_request("GET", f"/games/by-title/{search_query}")
            
            if success:
                games = result.get("results", [])
                st.write(f"**{len(games)} résultat(s):**")
                
                for game in games:
                    with st.expander(f"🎮 {game['title']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Note:** {game.get('rating', 'N/A')}")
                            st.write(f"**Metacritic:** {game.get('metacritic', 'N/A')}")
                        with col2:
                            if game.get('best_price'):
                                st.write(f"**Prix:** {game['best_price']}")
                            if st.button("⭐ Favori", key=f"fav_{game['id']}"):
                                st.session_state.favorites.add(game['title'])
                                st.success("Ajouté aux favoris!")
            else:
                st.error(f"Erreur: {result}")
    
    with tab2:
        st.subheader("Recommandations ML")
        
        query = st.text_input("Décrivez ce que vous cherchez", 
                             placeholder="RPG fantasy, indie platformer...")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            k = st.slider("Nombre de suggestions", 1, 20, 5)
        with col2:
            min_conf = st.slider("Confiance min", 0.0, 1.0, 0.1)
        
        if st.button("Obtenir des recommandations") and query:
            with st.spinner("Génération des recommandations..."):
                success, result = api_request("POST", "/recommend/ml", json={
                    "query": query,
                    "k": k,
                    "min_confidence": min_conf
                })
            
            if success:
                recommendations = result.get("recommendations", [])
                latency = result.get("latency_ms", 0)
                
                if recommendations:
                    st.success(f"{len(recommendations)} recommandations en {latency:.1f}ms")
                    
                    for i, rec in enumerate(recommendations, 1):
                        confidence = rec.get("confidence", rec.get("score", 0))
                        with st.container():
                            st.write(f"**#{i}. {rec.get('title', 'Titre inconnu')}**")
                            st.write(f"Confiance: {confidence:.3f} | Genres: {rec.get('genres', 'N/A')}")
                            st.write("---")
                else:
                    st.info("Aucune recommandation trouvée")
            else:
                st.error(f"Erreur: {result}")
    
    with tab3:
        st.subheader("Monitoring")
        
        if st.button("Actualiser les métriques"):
            # Health check
            success, health = api_request("GET", "/healthz")
            if success:
                st.json(health)
            
            # Model metrics
            success, metrics = api_request("GET", "/model/metrics")
            if success:
                st.subheader("Métriques du modèle")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Prédictions totales", metrics.get('total_predictions', 0))
                with col2:
                    st.metric("Confiance moyenne", f"{metrics.get('avg_confidence', 0):.3f}")
                with col3:
                    st.metric("Jeux dans le modèle", metrics.get('games_count', 0))

else:
    st.warning("Veuillez vous connecter pour accéder aux fonctionnalités")

# Footer
st.divider()
st.markdown("**🎮 Games Recommendation System** - Propulsé par FastAPI + ML")
