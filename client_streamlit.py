import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time
import re
from typing import List, Dict, Optional

# Configuration de la page
st.set_page_config(
    page_title="GameFinder Pro",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une interface moderne
st.markdown("""
<style>
    /* Variables CSS */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4ade80;
        --warning-color: #fbbf24;
        --error-color: #ef4444;
        --background-gradient: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    }

    /* Header principal */
    .main-header {
        background: var(--background-gradient);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    /* Cards de jeux */
    .game-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .game-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-color: var(--primary-color);
    }

    .game-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--background-gradient);
    }

    /* Métriques et badges */
    .metric-badge {
        background: var(--background-gradient);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }

    .rating-badge {
        background: var(--success-color);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .platform-tag {
        background: #f3f4f6;
        color: #374151;
        padding: 0.25rem 0.5rem;
        border-radius: 8px;
        font-size: 0.75rem;
        margin: 0.1rem;
        display: inline-block;
    }

    /* Status cards */
    .status-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }

    .success-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-color: var(--success-color);
    }

    .warning-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-color: var(--warning-color);
    }

    .error-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-color: var(--error-color);
    }

    /* Sidebar styling */
    .sidebar-content {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Boutons personnalisés */
    .custom-button {
        background: var(--background-gradient);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Animation de chargement */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .game-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# État de session pour la configuration API
def init_session_state():
    defaults = {
        'api_configured': False,
        'api_url': "https://game-app-y8be.onrender.com",
        'api_token': "",
        'authenticated': False,
        'username': "",
        'show_register': False,
        'search_history': [],
        'favorites': [],
        'last_search_results': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Header principal
st.markdown('''
<div class="main-header">
    <h1>🎮 GameFinder Pro</h1>
    <p>Découvrez vos prochains jeux favoris avec l'IA</p>
</div>
''', unsafe_allow_html=True)

# Fonction d'authentification améliorée
def authenticate_user(username: str, password: str) -> bool:
    try:
        response = requests.post(
            f"{st.session_state.api_url}/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
        
        if response.status_code == 200:
            token_data = response.json()
            st.session_state.api_token = token_data["access_token"]
            st.session_state.authenticated = True
            st.session_state.username = username
            return True
        else:
            st.error(f"Échec de l'authentification: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Erreur de connexion: {str(e)}")
        return False

# Fonction d'inscription
def register_user(username: str, password: str, confirm_password: str) -> bool:
    if password != confirm_password:
        st.error("Les mots de passe ne correspondent pas")
        return False
    
    if len(password) < 6:
        st.error("Le mot de passe doit contenir au moins 6 caractères")
        return False
        
    try:
        response = requests.post(
            f"{st.session_state.api_url}/register",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
        
        if response.status_code == 200:
            st.success("Compte créé avec succès ! Vous pouvez maintenant vous connecter.")
            return True
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            st.error(f"Échec de l'inscription: {error_data.get('detail', 'Erreur inconnue')}")
            return False
    except Exception as e:
        st.error(f"Erreur lors de l'inscription: {str(e)}")
        return False

# Fonction API robuste avec retry
def api_call(endpoint: str, method: str = "GET", data: dict = None, params: dict = None, timeout: int = 30, retries: int = 2) -> Optional[requests.Response]:
    if not st.session_state.authenticated:
        st.warning("Authentification requise")
        return None
    
    url = f"{st.session_state.api_url}{endpoint}"
    headers = {
        "Authorization": f"Bearer {st.session_state.api_token}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(retries + 1):
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data, timeout=timeout)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, timeout=timeout)
            
            return response
            
        except requests.exceptions.Timeout:
            if attempt < retries:
                st.warning(f"⏳ Timeout (tentative {attempt + 1}/{retries + 1}), nouvelle tentative...")
                time.sleep(2)
                continue
            else:
                st.error(f"⚠️ Timeout après {retries + 1} tentatives. L'API pourrait être en veille.")
                return None
        except requests.exceptions.ConnectionError:
            st.error("❌ Erreur de connexion. Vérifiez l'URL de l'API")
            return None
        except Exception as e:
            st.error(f"❌ Erreur de requête: {str(e)}")
            return None

# Fonction pour afficher une carte de jeu moderne
def display_game_card(game: dict, show_similarity: bool = False):
    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            title = game.get('title', 'Titre inconnu')
            st.markdown(f"**{title}**")
            
            # Genres avec badges
            genres = game.get('genres', 'N/A')
            if genres and genres != 'N/A':
                genre_list = [g.strip() for g in genres.split(',')]
                genre_html = ' '.join([f'<span class="platform-tag">{g}</span>' for g in genre_list[:3]])
                st.markdown(genre_html, unsafe_allow_html=True)
            
            # Plateformes
            platforms = game.get('platforms', 'N/A')
            if platforms and platforms != 'N/A':
                if isinstance(platforms, list):
                    platform_list = platforms
                else:
                    platform_list = [p.strip() for p in str(platforms).split(',')]
                
                platform_html = ' '.join([f'<span class="platform-tag">📱 {p}</span>' for p in platform_list[:4]])
                st.markdown(platform_html, unsafe_allow_html=True)
        
        with col2:
            rating = game.get('rating', 0)
            if rating > 0:
                st.markdown(f'<div class="rating-badge">⭐ {rating:.1f}/5</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="platform-tag">Pas de note</div>', unsafe_allow_html=True)
        
        with col3:
            if show_similarity:
                similarity = game.get('similarity_score', game.get('confidence', 0))
                st.metric("Similarité", f"{similarity:.2f}")
            else:
                metacritic = game.get('metacritic', 0)
                if metacritic > 0:
                    st.metric("Metacritic", f"{metacritic}/100")
        
        with col4:
            # Boutons d'action
            if st.button(f"💖 Favori", key=f"fav_{game.get('id', title)}"):
                if game not in st.session_state.favorites:
                    st.session_state.favorites.append(game)
                    st.success("Ajouté aux favoris!")
            
            # Simulation du prix et lien d'achat (à personnaliser selon vos données)
            price = game.get('price', f"{(int(game.get('id', 1)) * 7) % 60 + 10}€")  # Prix simulé
            st.markdown(f"**Prix:** {price}")
            
            # Liens vers plateformes (simulation)
            st.markdown("🛒 **Acheter:**")
            if st.button("Steam", key=f"steam_{game.get('id', title)}", help="Acheter sur Steam"):
                st.success("Redirection vers Steam...")
            if st.button("Epic", key=f"epic_{game.get('id', title)}", help="Acheter sur Epic Games"):
                st.success("Redirection vers Epic Games...")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

# Sidebar pour navigation et authentification
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    if not st.session_state.api_configured:
        st.markdown("### ⚙️ Configuration API")
        api_url = st.text_input("🌐 URL API", value=st.session_state.api_url)
        
        if st.button("✅ Connecter à l'API", type="primary"):
            st.session_state.api_url = api_url.rstrip('/')
            st.session_state.api_configured = True
            st.rerun()
    
    elif not st.session_state.authenticated:
        st.markdown("### 🔐 Authentification")
        
        # Tabs pour login/register
        if not st.session_state.show_register:
            st.markdown("**Connexion**")
            username = st.text_input("👤 Nom d'utilisateur")
            password = st.text_input("🔑 Mot de passe", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Se connecter", type="primary"):
                    if username and password:
                        if authenticate_user(username, password):
                            st.success("✅ Connexion réussie!")
                            st.rerun()
                    else:
                        st.error("Veuillez remplir tous les champs")
            
            with col2:
                if st.button("📝 S'inscrire"):
                    st.session_state.show_register = True
                    st.rerun()
        
        else:
            st.markdown("**Inscription**")
            new_username = st.text_input("👤 Nom d'utilisateur")
            new_password = st.text_input("🔑 Mot de passe", type="password")
            confirm_password = st.text_input("🔑 Confirmer mot de passe", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 Créer compte", type="primary"):
                    if new_username and new_password and confirm_password:
                        if register_user(new_username, new_password, confirm_password):
                            st.session_state.show_register = False
                            st.rerun()
                    else:
                        st.error("Veuillez remplir tous les champs")
            
            with col2:
                if st.button("🔙 Retour"):
                    st.session_state.show_register = False
                    st.rerun()
    
    else:
        st.markdown(f"### 👋 Bonjour {st.session_state.username}")
        st.success("🟢 Connecté et authentifié")
        
        # Menu de navigation
        st.markdown("### 🧭 Navigation")
        page = st.selectbox(
            "Choisir une page",
            ["🔍 Recherche Avancée", "⭐ Recommandations IA", "💖 Mes Favoris", "📊 Tableau de Bord", "🏥 Statut API"]
        )
        
        if st.button("🚪 Déconnecter"):
            # Reset session state
            for key in ['authenticated', 'api_token', 'username', 'api_configured']:
                st.session_state[key] = False if key == 'authenticated' or key == 'api_configured' else ""
            st.rerun()
        
        # Historique de recherche
        if st.session_state.search_history:
            st.markdown("### 📜 Historique")
            for i, search in enumerate(st.session_state.search_history[-5:]):
                if st.button(f"🔍 {search}", key=f"history_{i}"):
                    st.session_state.current_search = search
    
    st.markdown('</div>', unsafe_allow_html=True)

# Interface principale
if st.session_state.authenticated:
    # Récupération de la page sélectionnée
    current_page = locals().get('page', "🔍 Recherche Avancée")
    
    if current_page == "🔍 Recherche Avancée":
        st.markdown("## 🔍 Recherche Avancée de Jeux")
        
        # Interface de recherche améliorée
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_type = st.selectbox(
                "Type de recherche",
                ["🎯 Par Titre", "🎮 Par Genre", "📱 Par Plateforme", "🔍 Recherche Libre"]
            )
            
            if search_type == "🎯 Par Titre":
                search_query = st.text_input("🔍 Nom du jeu", placeholder="The Witcher, Cyberpunk, Mario...")
                search_limit = st.slider("Nombre de résultats", 1, 25, 10)
                
                if st.button("🔍 Rechercher par titre", type="primary", use_container_width=True):
                    if search_query:
                        st.session_state.search_history.append(search_query)
                        with st.spinner("🎮 Recherche en cours..."):
                            response = api_call(f"/recommend/by-title/{search_query}", params={"k": search_limit})
                            
                            if response and response.status_code == 200:
                                results = response.json()
                                st.session_state.last_search_results = results.get("recommendations", [])
                                
                                st.success(f"✅ {len(st.session_state.last_search_results)} jeu(x) trouvé(s)")
                                
                                for game in st.session_state.last_search_results:
                                    display_game_card(game, show_similarity=True)
                            else:
                                st.error("❌ Aucun résultat trouvé")
            
            elif search_type == "🎮 Par Genre":
                genre = st.selectbox(
                    "Choisir un genre",
                    ["Action", "Adventure", "RPG", "Strategy", "Simulation", "Sports", "Racing", "Indie", "FPS", "Platform"]
                )
                genre_limit = st.slider("Nombre de jeux", 1, 30, 15)
                
                if st.button("🎯 Rechercher par genre", type="primary", use_container_width=True):
                    with st.spinner(f"🎮 Recherche de jeux {genre}..."):
                        response = api_call(f"/recommend/by-genre/{genre}", params={"k": genre_limit})
                        
                        if response and response.status_code == 200:
                            results = response.json()
                            st.session_state.last_search_results = results.get("recommendations", [])
                            
                            st.success(f"✅ {len(st.session_state.last_search_results)} jeu(x) {genre} trouvé(s)")
                            
                            for game in st.session_state.last_search_results:
                                display_game_card(game)
            
            elif search_type == "🔍 Recherche Libre":
                free_query = st.text_area("Décrivez ce que vous cherchez", 
                                        placeholder="Un RPG médiéval avec des dragons, ou un jeu de stratégie futuriste...")
                ml_limit = st.slider("Nombre de suggestions", 1, 20, 10)
                confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.1, 0.05)
                
                if st.button("🤖 Recherche IA", type="primary", use_container_width=True):
                    if free_query:
                        recommendation_data = {
                            "query": free_query,
                            "k": ml_limit,
                            "min_confidence": confidence_threshold
                        }
                        
                        with st.spinner("🧠 L'IA analyse votre demande..."):
                            response = api_call("/recommend/ml", method="POST", data=recommendation_data)
                            
                            if response and response.status_code == 200:
                                results = response.json()
                                st.session_state.last_search_results = results.get("recommendations", [])
                                
                                st.success(f"🤖 L'IA a trouvé {len(st.session_state.last_search_results)} recommandation(s)")
                                st.info(f"⚡ Temps de traitement: {results.get('latency_ms', 0):.1f}ms")
                                
                                for game in st.session_state.last_search_results:
                                    display_game_card(game, show_similarity=True)
        
        with col2:
            st.markdown("### 🎯 Filtres Avancés")
            
            # Filtres
            min_rating = st.slider("Note minimum", 0.0, 5.0, 3.0, 0.1)
            min_metacritic = st.slider("Score Metacritic minimum", 0, 100, 70)
            
            # Plateformes préférées
            preferred_platforms = st.multiselect(
                "Plateformes préférées",
                ["PC", "PlayStation", "Xbox", "Nintendo Switch", "Mobile", "VR"]
            )
            
            st.markdown("### 📊 Résultats")
            if st.session_state.last_search_results:
                st.metric("Jeux trouvés", len(st.session_state.last_search_results))
                avg_rating = sum(game.get('rating', 0) for game in st.session_state.last_search_results) / len(st.session_state.last_search_results)
                st.metric("Note moyenne", f"{avg_rating:.1f}/5")
    
    elif current_page == "💖 Mes Favoris":
        st.markdown("## 💖 Mes Jeux Favoris")
        
        if st.session_state.favorites:
            st.success(f"Vous avez {len(st.session_state.favorites)} jeu(x) favori(s)")
            
            for i, game in enumerate(st.session_state.favorites):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        display_game_card(game)
                    
                    with col2:
                        if st.button(f"🗑️ Retirer", key=f"remove_fav_{i}"):
                            st.session_state.favorites.pop(i)
                            st.rerun()
        else:
            st.info("Aucun jeu favori pour le moment. Utilisez la recherche pour en ajouter !")
    
    elif current_page == "📊 Tableau de Bord":
        st.markdown("## 📊 Tableau de Bord")
        
        # Métriques utilisateur
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔍 Recherches", len(st.session_state.search_history))
        
        with col2:
            st.metric("💖 Favoris", len(st.session_state.favorites))
        
        with col3:
            st.metric("🎮 Derniers résultats", len(st.session_state.last_search_results))
        
        with col4:
            if st.button("📊 Métriques API"):
                response = api_call("/model/metrics")
                if response and response.status_code == 200:
                    metrics = response.json()
                    st.json(metrics)
        
        # Graphiques des favoris par genre (simulation)
        if st.session_state.favorites:
            st.markdown("### 📈 Analyse de vos Favoris")
            
            # Analyse des genres favoris
            genre_count = {}
            for game in st.session_state.favorites:
                genres = game.get('genres', '').split(',')
                for genre in genres:
                    genre = genre.strip()
                    if genre:
                        genre_count[genre] = genre_count.get(genre, 0) + 1
            
            if genre_count:
                df_genres = pd.DataFrame(list(genre_count.items()), columns=['Genre', 'Count'])
                st.bar_chart(df_genres.set_index('Genre'))
    
    elif current_page == "🏥 Statut API":
        st.markdown("## 🏥 Statut de l'API")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Vérifier le statut", type="primary"):
                with st.spinner("Vérification du statut..."):
                    response = api_call("/healthz")
                    
                    if response and response.status_code == 200:
                        health_data = response.json()
                        
                        status = health_data.get("status", "unknown")
                        if status == "healthy":
                            st.markdown('<div class="success-card">🟢 API en parfaite santé</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="warning-card">⚠️ Statut: {status}</div>', unsafe_allow_html=True)
                        
                        # Détails du système
                        st.markdown("### Détails du système")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Base de données", "✅ Connectée" if health_data.get('db_ready', False) else "❌ Déconnectée")
                            st.metric("Modèle ML", "✅ Chargé" if health_data.get('model_loaded', False) else "❌ Non chargé")
                        
                        with col_b:
                            st.metric("Version modèle", health_data.get('model_version', 'unknown'))
                            st.metric("Compliance", "✅ Activée" if health_data.get('compliance_enabled', False) else "❌ Désactivée")
                        
                        if health_data.get("monitoring"):
                            st.markdown("### Métriques de monitoring")
                            st.json(health_data["monitoring"])
        
        with col2:
            st.markdown("### 🛠️ Outils de diagnostic")
            
            if st.button("🧪 Test de latence"):
                start_time = time.time()
                response = api_call("/healthz")
                end_time = time.time()
                
                if response:
                    latency = (end_time - start_time) * 1000
                    if latency < 500:
                        st.success(f"⚡ Latence: {latency:.1f}ms (Excellent)")
                    elif latency < 1000:
                        st.warning(f"⏳ Latence: {latency:.1f}ms (Correcte)")
                    else:
                        st.error(f"🐌 Latence: {latency:.1f}ms (Lente)")

else:
    # Page d'accueil pour utilisateurs non connectés
    st.markdown("## 🎮 Bienvenue sur GameFinder Pro")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Recherche Intelligente
        Trouvez vos jeux parfaits grâce à notre moteur de recherche alimenté par l'IA.
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Recommandations IA
        Découvrez de nouveaux jeux basés sur vos préférences et votre historique.
        """)
    
    with col3:
        st.markdown("""
        ### 💖 Gestion des Favoris
        Sauvegardez vos jeux préférés et créez votre collection personnelle.
        """)
    
    st.info("👆 Connectez-vous pour accéder à toutes les fonctionnalités")
    
    # Démonstration des fonctionnalités
    st.markdown("---")
    st.markdown("## ✨ Fonctionnalités Innovantes")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        ### 🎯 Recherche Multi-Critères
        - **Recherche par titre** : Trouvez rapidement un jeu spécifique
        - **Recherche par genre** : Explorez par catégories (RPG, Action, etc.)
        - **Recherche IA libre** : Décrivez ce que vous voulez et laissez l'IA trouver
        - **Filtres avancés** : Note minimum, score Metacritic, plateformes
        
        ### 🤖 IA Avancée
        - **Recommandations personnalisées** basées sur vos goûts
        - **Analyse de similarité** entre les jeux
        - **Exploration de clusters** de jeux similaires
        - **Métriques de confiance** pour chaque recommandation
        """)
    
    with feature_col2:
        st.markdown("""
        ### 🛒 Informations Commerciales
        - **Prix en temps réel** pour chaque jeu
        - **Liens directs** vers les plateformes d'achat (Steam, Epic, etc.)
        - **Disponibilité par plateforme** (PC, Console, Mobile)
        - **Notes et scores** (Metacritic, utilisateurs)
        
        ### 📊 Tableau de Bord Personnel
        - **Historique de recherche** pour retrouver vos découvertes
        - **Collection de favoris** avec gestion avancée
        - **Statistiques personnelles** d'utilisation
        - **Analyse de vos préférences** par genre
        """)
    
    # Exemples de jeux en démonstration
    st.markdown("---")
    st.markdown("## 🎮 Aperçu des Jeux Disponibles")
    
    demo_games = [
        {
            "title": "The Witcher 3: Wild Hunt",
            "genres": "RPG, Action, Adventure",
            "rating": 4.8,
            "metacritic": 93,
            "platforms": ["PC", "PlayStation", "Xbox", "Nintendo Switch"],
            "price": "39.99€"
        },
        {
            "title": "Hades",
            "genres": "Action, Roguelike, Indie",
            "rating": 4.7,
            "metacritic": 93,
            "platforms": ["PC", "PlayStation", "Xbox", "Nintendo Switch"],
            "price": "24.99€"
        },
        {
            "title": "Cyberpunk 2077",
            "genres": "RPG, Action, Sci-Fi",
            "rating": 4.2,
            "metacritic": 86,
            "platforms": ["PC", "PlayStation", "Xbox"],
            "price": "59.99€"
        }
    ]
    
    for i, game in enumerate(demo_games):
        with st.container():
            st.markdown('<div class="game-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{game['title']}**")
                genre_html = ' '.join([f'<span class="platform-tag">{g.strip()}</span>' for g in game['genres'].split(',')])
                st.markdown(genre_html, unsafe_allow_html=True)
                
                platform_html = ' '.join([f'<span class="platform-tag">📱 {p}</span>' for p in game['platforms']])
                st.markdown(platform_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="rating-badge">⭐ {game["rating"]}/5</div>', unsafe_allow_html=True)
            
            with col3:
                st.metric("Metacritic", f"{game['metacritic']}/100")
            
            with col4:
                st.markdown(f"**Prix:** {game['price']}")
                st.button("🔒 Connexion requise", disabled=True, key=f"demo_buy_{i}")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer informatif
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🚀 Technologies utilisées:**
    - FastAPI pour l'API backend
    - Machine Learning avec scikit-learn
    - Base de données MySQL
    - Authentification JWT
    """)

with col2:
    st.markdown("""
    **📈 Métriques en temps réel:**
    - Monitoring Prometheus
    - Métriques de latence
    - Suivi des performances
    - Logs détaillés
    """)

with col3:
    st.markdown("""
    **🔒 Sécurité:**
    - Authentification sécurisée
    - Validation des données
    - Protection CORS
    - Standards de compliance
    """)

# Script JavaScript pour des interactions avancées (optionnel)
st.markdown("""
<script>
// Animation smooth scroll pour les ancres
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Effet de hover sur les cartes
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.game-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});
</script>
""", unsafe_allow_html=True)
