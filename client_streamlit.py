import os
import time
import requests
import streamlit as st
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from datetime import datetime

# Configuration
API_BASE_URL = (
    st.secrets.get("API_URL") if hasattr(st, 'secrets') and st.secrets else
    os.getenv("API_URL") or
    "https://game-app-y8be.onrender.com"
)

st.set_page_config(
    page_title="🎮 Enhanced Games AI", 
    page_icon="🎮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'favorites' not in st.session_state:
    st.session_state.favorites = set()
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# API Helper with enhanced error handling
def api_request(method: str, path: str, **kwargs) -> tuple:
    """Fonction pour requêtes API avec gestion d'erreur améliorée"""
    url = f"{API_BASE_URL}{path}"
    headers = {"accept": "application/json"}
    
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=30, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, timeout=60, **kwargs)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, timeout=60, **kwargs)
        else:
            return False, f"Méthode non supportée: {method}"
        
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = f"Erreur HTTP {response.status_code}: {response.text[:200]}"
            return False, error_detail
            
        return True, response.json()
        
    except requests.exceptions.Timeout:
        return False, "Timeout - L'API met trop de temps à répondre (>30s)"
    except requests.exceptions.ConnectionError:
        return False, "Erreur de connexion - Vérifiez que l'API est accessible"
    except Exception as e:
        return False, f"Erreur: {str(e)}"

def display_api_status():
    """Affiche le statut de l'API dans la sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Statut API")
        
        if st.button("🔄 Vérifier API", key="check_api"):
            with st.spinner("Vérification..."):
                success, result = api_request("GET", "/healthz")
                
                if success:
                    st.success("✅ API Active")
                    status = result.get("status", "unknown")
                    models_info = result.get("available_models", 0)
                    
                    st.write(f"**Status:** {status}")
                    st.write(f"**Modèles:** {models_info}")
                    st.write(f"**Compliance:** {'✅' if result.get('compliance_enabled') else '❌'}")
                else:
                    st.error(f"❌ API Indisponible\n{result}")
        
        st.markdown(f"**🔗 API:** {API_BASE_URL}")

# Header
st.title("🎮 Enhanced Games AI/ML Platform")
st.markdown("**Plateforme avancée de recommandation avec modèles IA multiples**")

# Sidebar - Auth
with st.sidebar:
    st.header("👤 Authentification")
    
    if st.session_state.token:
        st.success(f"✅ Connecté: {st.session_state.username}")
        if st.button("🚪 Se déconnecter"):
            st.session_state.token = None
            st.session_state.username = None
            st.rerun()
    else:
        auth_tab1, auth_tab2 = st.tabs(["Connexion", "Inscription"])
        
        with auth_tab1:
            with st.form("login"):
                username = st.text_input("Nom d'utilisateur", placeholder="demo")
                password = st.text_input("Mot de passe", type="password", placeholder="demo123")
                
                if st.form_submit_button("🔑 Se connecter"):
                    if username and password:
                        success, result = api_request("POST", "/token", data={
                            "username": username,
                            "password": password
                        })
                        
                        if success:
                            st.session_state.token = result["access_token"]
                            st.session_state.username = username
                            st.success("✅ Connexion réussie!")
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")

        with auth_tab2:
            with st.form("register"):
                reg_user = st.text_input("Nom d'utilisateur")
                reg_pass = st.text_input("Mot de passe", type="password", 
                                       help="8+ caractères avec majuscule, minuscule et chiffre")
                
                if st.form_submit_button("📝 S'inscrire"):
                    if reg_user and reg_pass:
                        success, result = api_request("POST", "/register", data={
                            "username": reg_user,
                            "password": reg_pass
                        })
                        
                        if success:
                            st.success("✅ Inscription réussie!")
                        else:
                            st.error(f"❌ {result}")

display_api_status()

# Main content - only if authenticated
if st.session_state.token:
    main_tabs = st.tabs([
        "🔍 Search", "✨ Recommendations", "🤖 Multi-Model", 
        "📊 Analytics", "⚙️ Model Management", "📈 Monitoring"
    ])
    
    # Tab 1: Enhanced Search
    with main_tabs[0]:
        st.subheader("🔍 Recherche avancée de jeux")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Rechercher un jeu", 
                placeholder="Mario, Zelda, RPG fantasy...",
                help="Utilisez des mots-clés, genres ou descriptions"
            )
        
        with col2:
            search_type = st.selectbox(
                "Type de recherche",
                ["Par titre", "Par genre", "Description libre"]
            )
        
        if st.button("🔍 Rechercher", type="primary") and search_query:
            # Add to search history
            if search_query not in st.session_state.search_history:
                st.session_state.search_history.insert(0, search_query)
                st.session_state.search_history = st.session_state.search_history[:10]
            
            with st.spinner("Recherche en cours..."):
                if search_type == "Par titre":
                    success, result = api_request("GET", f"/games/by-title/{search_query}")
                else:
                    success, result = api_request("POST", "/recommend/ml", json={
                        "query": search_query,
                        "k": 10,
                        "min_confidence": 0.1
                    })
            
            if success:
                if search_type == "Par titre":
                    games = result.get("results", [])
                    st.write(f"**{len(games)} résultat(s) trouvé(s):**")
                    
                    for game in games:
                        with st.expander(f"🎮 {game['title']}", expanded=True):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                st.write(f"**Note:** ⭐ {game.get('rating', 'N/A')}")
                                st.write(f"**Metacritic:** 🎯 {game.get('metacritic', 'N/A')}")
                            
                            with col2:
                                if game.get('best_price'):
                                    st.write(f"**Prix:** 💰 {game['best_price']}")
                                st.write(f"**Genres:** {game.get('genres', 'N/A')}")
                            
                            with col3:
                                if st.button("⭐ Favori", key=f"fav_{game['id']}"):
                                    st.session_state.favorites.add(game['title'])
                                    st.success("Ajouté!")
                
                else:  # ML recommendations
                    recommendations = result.get("recommendations", [])
                    algorithm = result.get("algorithm", "tfidf")
                    latency = result.get("latency_ms", 0)
                    
                    st.success(f"✨ {len(recommendations)} recommandations trouvées en {latency:.1f}ms")
                    st.info(f"🤖 Algorithme utilisé: {algorithm.upper()}")
                    
                    for i, rec in enumerate(recommendations, 1):
                        confidence = rec.get("confidence", rec.get("score", 0))
                        
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.write(f"**#{i}. {rec.get('title', 'Titre inconnu')}**")
                                st.write(f"🎭 Genres: {rec.get('genres', 'N/A')}")
                                if rec.get('rating'):
                                    st.write(f"⭐ Note: {rec.get('rating')}")
                            
                            with col2:
                                st.metric("Confiance", f"{confidence:.3f}")
                            
                            st.markdown("---")
            else:
                st.error(f"❌ Erreur de recherche: {result}")
        
        # Search history
        if st.session_state.search_history:
            st.subheader("📚 Historique des recherches")
            for query in st.session_state.search_history[:5]:
                if st.button(f"🔄 {query}", key=f"history_{query}"):
                    st.session_state.search_query = query
                    st.rerun()

    # Tab 2: Enhanced ML Recommendations
    with main_tabs[1]:
        st.subheader("✨ Recommandations ML Avancées")
        
        # Algorithm selection
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            query = st.text_input(
                "Décrivez ce que vous cherchez", 
                placeholder="RPG fantasy, indie platformer, action shooter...",
                help="Plus vous êtes précis, meilleures seront les recommandations"
            )
        
        with col2:
            k = st.slider("Nombre de suggestions", 1, 20, 10)
            min_conf = st.slider("Confiance min", 0.0, 1.0, 0.1, 0.05)
        
        with col3:
            algorithm = st.selectbox(
                "Algorithme",
                ["hybrid", "tfidf", "nmf"],
                help="hybrid = TF-IDF + NMF combinés"
            )
        
        # User profile section
        with st.expander("🎯 Profil utilisateur (optionnel)"):
            col1, col2 = st.columns(2)
            
            with col1:
                preferred_genres = st.multiselect(
                    "Genres préférés",
                    ["Action", "RPG", "Strategy", "Simulation", "Sports", "Racing", "Puzzle", "Indie"]
                )
                
                preferred_platforms = st.multiselect(
                    "Plateformes préférées", 
                    ["PC", "PlayStation", "Xbox", "Switch", "Mobile"]
                )
            
            with col2:
                rating_pref = st.select_slider(
                    "Préférence de note",
                    ["any", "medium", "high"],
                    value="any"
                )
        
        if st.button("🚀 Obtenir des recommandations", type="primary") and query:
            # Prepare user profile
            user_profile = None
            if preferred_genres or preferred_platforms or rating_pref != "any":
                user_profile = {
                    "preferred_genres": preferred_genres,
                    "preferred_platforms": preferred_platforms, 
                    "rating_preference": rating_pref
                }
            
            with st.spinner(f"Génération des recommandations avec {algorithm.upper()}..."):
                # Use specific model endpoint
                if algorithm == "hybrid":
                    success, result = api_request("POST", "/recommend/ml", json={
                        "query": query,
                        "k": k,
                        "min_confidence": min_conf,
                        "user_profile": user_profile
                    })
                else:
                    success, result = api_request("POST", f"/recommend/model/recommendation_{algorithm}", json={
                        "query": query,
                        "k": k,
                        "min_confidence": min_conf,
                        "user_profile": user_profile
                    })
            
            if success:
                recommendations = result.get("recommendations", [])
                latency = result.get("latency_ms", 0)
                model_version = result.get("model_version", "unknown")
                
                if recommendations:
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Recommandations", len(recommendations))
                    col2.metric("Latence", f"{latency:.1f}ms")
                    col3.metric("Algorithme", algorithm.upper())
                    col4.metric("Version", model_version)
                    
                    # Display recommendations
                    st.markdown("### 🎯 Vos recommandations")
                    
                    for i, rec in enumerate(recommendations, 1):
                        confidence = rec.get("confidence", 0)
                        
                        with st.container():
                            col1, col2, col3 = st.columns([3, 2, 1])
                            
                            with col1:
                                st.markdown(f"**#{i}. {rec.get('title', 'Titre inconnu')}**")
                                st.write(f"🎭 {rec.get('genres', 'N/A')}")
                            
                            with col2:
                                if rec.get('rating'):
                                    st.write(f"⭐ {rec.get('rating')}")
                                if rec.get('metacritic'):
                                    st.write(f"🎯 {rec.get('metacritic')}")
                            
                            with col3:
                                st.metric("Score", f"{confidence:.3f}")
                                
                            # Recommendation explanation
                            if user_profile and rec.get('genres'):
                                matching_genres = [g for g in preferred_genres if g in rec.get('genres', '')]
                                if matching_genres:
                                    st.caption(f"✅ Correspond à vos genres: {', '.join(matching_genres)}")
                            
                            st.markdown("---")
                else:
                    st.info("🤷 Aucune recommandation trouvée avec ces critères")
            else:
                st.error(f"❌ Erreur: {result}")

    # Tab 3: Multi-Model Comparison  
    with main_tabs[2]:
        st.subheader("🤖 Comparaison Multi-Modèles")
        
        query = st.text_input(
            "Requête de test", 
            placeholder="Action RPG fantasy"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            algorithms = st.multiselect(
                "Algorithmes à comparer",
                ["tfidf", "nmf", "hybrid"],
                default=["tfidf", "hybrid"]
            )
        
        with col2:
            k = st.number_input("Nombre de recommandations", 1, 20, 5)
        
        if st.button("🔬 Comparer les algorithmes") and query and algorithms:
            with st.spinner("Comparaison en cours..."):
                success, result = api_request("POST", "/recommend/compare", json={
                    "query": query,
                    "algorithms": algorithms,
                    "k": k
                })
            
            if success:
                comparison_results = result.get("comparison_results", [])
                best_algorithm = result.get("best_algorithm")
                
                st.success(f"🏆 Meilleur algorithme: **{best_algorithm.upper()}**")
                
                # Display comparison table
                comparison_data = []
                for res in comparison_results:
                    if "error" not in res:
                        comparison_data.append({
                            "Algorithme": res["algorithm"].upper(),
                            "Recommandations": res["count"],
                            "Confiance moy.": f"{res.get('avg_confidence', 0):.3f}",
                            "Latence (ms)": f"{res.get('latency_ms', 0):.1f}"
                        })
                
                if comparison_data:
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True)
                
                # Display detailed results
                for res in comparison_results:
                    if "error" not in res:
                        with st.expander(f"Détail {res['algorithm'].upper()}"):
                            for i, rec in enumerate(res.get("recommendations", []), 1):
                                st.write(f"{i}. **{rec.get('title')}** (conf: {rec.get('confidence', 0):.3f})")
            else:
                st.error(f"❌ Erreur: {result}")

    # Tab 4: Game Classification & Clustering
    with main_tabs[3]:
        st.subheader("🧠 Classification & Clustering IA")
        
        analysis_type = st.radio(
            "Type d'analyse",
            ["Classification par genre", "Clustering de jeux"],
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            game_title = st.text_input("Titre du jeu", placeholder="Cyberpunk 2077")
            game_description = st.text_area(
                "Description du jeu",
                placeholder="Jeu de rôle futuriste dans une ville cyberpunk avec des éléments d'action...",
                height=100
            )
        
        with col2:
            rating = st.slider("Note (si connue)", 0.0, 5.0, 4.0, 0.1)
            metacritic = st.slider("Score Metacritic (si connu)", 0, 100, 80)
        
        if st.button("🔮 Analyser le jeu") and (game_title or game_description):
            request_data = {
                "game_description": game_description,
                "title": game_title,
                "rating": rating,
                "metacritic": metacritic
            }
            
            if analysis_type == "Classification par genre":
                with st.spinner("Classification en cours..."):
                    success, result = api_request("POST", "/classify/game", json=request_data)
                
                if success:
                    classification = result.get("classification_result", {})
                    
                    st.success("✅ Classification terminée")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Genre prédit", classification.get("predicted_class", "Unknown"))
                        st.metric("Confiance", f"{classification.get('confidence', 0):.3f}")
                    
                    with col2:
                        st.subheader("🎯 Top prédictions")
                        for pred in classification.get("top_predictions", [])[:3]:
                            st.write(f"**{pred['class']}**: {pred['probability']:.3f}")
                
            else:  # Clustering
                with st.spinner("Analyse des clusters..."):
                    success, result = api_request("POST", "/cluster/game", json=request_data)
                
                if success:
                    clustering = result.get("clustering_result", {})
                    cluster_id = clustering.get("cluster")
                    confidence = clustering.get("confidence", 0)
                    
                    st.success("✅ Clustering terminé")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Cluster", f"#{cluster_id}")
                        st.metric("Confiance", f"{confidence:.3f}")
                    
                    with col2:
                        cluster_desc = clustering.get("cluster_description", {})
                        if cluster_desc:
                            st.subheader("📊 Caractéristiques du cluster")
                            st.write(f"**Taille:** {cluster_desc.get('size', 0)} jeux")
                            st.write(f"**Note moy.:** {cluster_desc.get('avg_rating', 0):.2f}")
                            
                            common_genres = cluster_desc.get("common_genres", [])
                            if common_genres:
                                st.write(f"**Genres:** {', '.join(common_genres)}")
            
            if not success:
                st.error(f"❌ Erreur d'analyse: {result}")

    # Tab 5: Model Management
    with main_tabs[4]:
        st.subheader("⚙️ Gestion des Modèles")
        
        # List available models
        if st.button("📋 Lister les modèles"):
            success, result = api_request("GET", "/models")
            
            if success:
                models = result.get("models", [])
                supported_types = result.get("supported_types", [])
                
                st.write(f"**Types supportés:** {', '.join(supported_types)}")
                
                if models:
                    model_data = []
                    for model in models:
                        info = model["info"]
                        model_data.append({
                            "ID": model["id"],
                            "Type": info["type"],
                            "Version": info["version"],
                            "Entraîné": "✅" if info["is_trained"] else "❌",
                            "Créé": info["created_at"][:19] if info["created_at"] else "N/A"
                        })
                    
                    df = pd.DataFrame(model_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Aucun modèle disponible")
        
        st.markdown("---")
        
        # Train models section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏃 Entraînement")
            
            model_type = st.selectbox(
                "Type de modèle",
                ["recommendation", "classification", "clustering"]
            )
            
            model_id = st.text_input(
                "ID du modèle",
                value=f"{model_type}_custom"
            )
            
            force_retrain = st.checkbox("Forcer le réentraînement")
            
            if st.button("🚀 Entraîner"):
                with st.spinner("Entraînement en cours..."):
                    # Create model first if needed
                    create_success, create_result = api_request(
                        "POST", f"/models/{model_type}/create",
                        params={"model_id": model_id}
                    )
                    
                    # Then train
                    success, result = api_request("POST", f"/models/{model_id}/train", json={
                        "force_retrain": force_retrain
                    })
                
                if success:
                    st.success("✅ Entraînement réussi!")
                    st.json(result)
                else:
                    st.error(f"❌ Erreur d'entraînement: {result}")
        
        with col2:
            st.subheader("📊 Métriques")
            
            metrics_model_id = st.text_input(
                "ID du modèle à analyser",
                value="recommendation_default"
            )
            
            if st.button("📈 Voir les métriques"):
                success, result = api_request("GET", f"/models/{metrics_model_id}/metrics")
                
                if success:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Version", result.get("model_version", "N/A"))
                        st.metric("Entraîné", "✅" if result.get("is_trained") else "❌")
                    
                    with col2:
                        st.metric("Prédictions", result.get("total_predictions", 0))
                        st.metric("Confiance moy.", f"{result.get('avg_confidence', 0):.3f}")
                    
                    if result.get("accuracy"):
                        st.metric("Précision", f"{result.get('accuracy', 0):.3f}")
                else:
                    st.error(f"❌ Erreur: {result}")

    # Tab 6: Advanced Monitoring
    with main_tabs[5]:
        st.subheader("📈 Monitoring Avancé")
        
        # System health
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏥 Santé Système"):
                success, result = api_request("GET", "/monitoring/system-health")
                
                if success:
                    overall_status = result.get("overall_status", "unknown")
                    components = result.get("components", {})
                    
                    st.metric("Statut global", overall_status.upper())
                    
                    st.subheader("Composants")
                    for component, status in components.items():
                        if isinstance(status, dict):
                            comp_status = status.get("status", "unknown")
                            st.write(f"**{component}**: {comp_status}")
                        else:
                            st.write(f"**{component}**: {status}")
                else:
                    st.error(f"Erreur: {result}")
        
        with col2:
            if st.button("📊 Statut Modèles"):
                success, result = api_request("GET", "/monitoring/models-status")
                
                if success:
                    st.metric("Total modèles", result.get("total_models", 0))
                    st.metric("Modèles entraînés", result.get("trained_models", 0))
                    
                    models_by_type = result.get("models_by_type", {})
                    if models_by_type:
                        st.subheader("Par type")
                        for model_type, count in models_by_type.items():
                            st.write(f"**{model_type}**: {count}")
        
        st.markdown("---")
        
        # Performance analysis
        if st.button("🔬 Analyser les performances"):
            with st.spinner("Analyse en cours..."):
                success, result = api_request("POST", "/analyze/model-performance")
            
            if success:
                performance = result.get("performance_analysis", {})
                recommendations = result.get("recommendations", {})
                
                st.subheader("📊 Résultats d'analyse")
                
                # Create comparison dataframe
                analysis_data = []
                for algo, metrics in performance.items():
                    analysis_data.append({
                        "Algorithme": algo.upper(),
                        "Latence moy. (ms)": f"{metrics.get('avg_latency_ms', 0):.1f}",
                        "Confiance moy.": f"{metrics.get('avg_confidence', 0):.3f}",
                        "Nb recommandations": f"{metrics.get('avg_recommendations', 0):.1f}",
                        "Diversité": f"{metrics.get('avg_diversity', 0):.2f}"
                    })
                
                if analysis_data:
                    df = pd.DataFrame(analysis_data)
                    st.dataframe(df, use_container_width=True)
                
                # Recommendations
                st.subheader("🎯 Recommandations")
                st.write(f"**Production**: {recommendations.get('production', 'N/A')}")
                st.write(f"**Performance critique**: {recommendations.get('speed_critical', 'N/A')}")
                st.write(f"**Diversité maximale**: {recommendations.get('diversity_focused', 'N/A')}")
            else:
                st.error(f"Erreur d'analyse: {result}")

    # Favorites section in sidebar
    with st.sidebar:
        if st.session_state.favorites:
            st.markdown("---")
            st.subheader("⭐ Favoris")
            for fav in list(st.session_state.favorites)[:5]:
                st.write(f"• {fav}")

else:
    # Not authenticated - show welcome screen
    st.markdown("""
    ## 🎮 Bienvenue sur la plateforme Games AI
    
    Cette application utilise l'intelligence artificielle pour :
    
    - 🔍 **Rechercher** des jeux par titre ou description
    - ✨ **Recommander** des jeux avec des algorithmes ML avancés
    - 🤖 **Comparer** plusieurs modèles d'IA
    - 🧠 **Classifier** et **regrouper** les jeux automatiquement
    - 📊 **Analyser** les performances des modèles
    - ⚙️ **Gérer** les modèles ML en temps réel
    
    ### 🚀 Pour commencer
    
    1. **Connectez-vous** avec vos identifiants
    2. Ou utilisez le **compte démo** :
       - Username: `demo`
       - Password: `demo123`
    3. **Explorez** les fonctionnalités IA avancées
    
    ### 🎯 Fonctionnalités avancées
    
    - **Multi-algorithmes** : TF-IDF, NMF, Hybrid
    - **Profils utilisateur** : Recommandations personnalisées
    - **Classification automatique** : Détection de genres
    - **Clustering intelligent** : Regroupement de jeux similaires
    - **Monitoring temps réel** : Performance et drift des modèles
    - **A/B Testing** : Comparaison d'algorithmes
    
    """)
    
    # Demo credentials reminder
    st.info("💡 **Compte démo disponible** - Username: `demo` / Password: `demo123`")
    
    # API Status for non-authenticated users
    st.markdown("---")
    st.subheader("📊 Statut du service")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Tester l'API", type="secondary"):
            with st.spinner("Test en cours..."):
                success, result = api_request("GET", "/healthz")
                
                if success:
                    st.success("✅ Service opérationnel")
                    
                    # Display some public info
                    status = result.get("status", "unknown")
                    api_version = result.get("api_version", "unknown")
                    models_available = result.get("available_models", 0)
                    
                    st.write(f"**Version API:** {api_version}")
                    st.write(f"**Statut:** {status}")
                    st.write(f"**Modèles IA:** {models_available} disponibles")
                    
                else:
                    st.error(f"❌ Service indisponible: {result}")
    
    with col2:
        st.markdown("""
        **🔗 Liens utiles:**
        - [API Documentation](""" + API_BASE_URL + """/docs)
        - [Métriques Prometheus](""" + API_BASE_URL + """/metrics)
        - [GitHub Repository](https://github.com/your-repo/games-api)
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎮 Games AI/ML Platform**")

with col2:
    st.markdown(f"**🔗 API:** [Documentation]({API_BASE_URL}/docs)")

with col3:
    st.markdown("**⚡ Propulsé par FastAPI + Streamlit**")

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fafafa;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    /* Custom styling for metrics */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(90deg, #2196F3, #1976D2);
        color: white;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        padding: 1rem 1rem 1rem;
    }
    
    /* Status indicators */
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-degraded {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading-text {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Advanced features for authenticated users
if st.session_state.token:
    
    # Quick actions in sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚡ Actions rapides")
        
        if st.button("🔄 Actualiser modèles"):
            with st.spinner("Actualisation..."):
                success, result = api_request("GET", "/models")
                if success:
                    st.success(f"✅ {len(result.get('models', []))} modèles trouvés")
                else:
                    st.error("❌ Erreur actualisation")
        
        if st.button("📊 Métriques rapides"):
            success, result = api_request("GET", "/model/metrics")
            if success:
                st.metric("Prédictions", result.get("total_predictions", 0))
                st.metric("Confiance", f"{result.get('avg_confidence', 0):.3f}")
            else:
                st.error("Erreur métriques")
    
    # Experimental features
    with st.expander("🧪 Fonctionnalités expérimentales"):
        st.warning("⚠️ Ces fonctionnalités sont en phase de test")
        
        # Batch predictions
        st.subheader("🔄 Prédictions par lot")
        batch_queries = st.text_area(
            "Requêtes (une par ligne)",
            placeholder="Action RPG\nIndie platformer\nStrategy game",
            height=100
        )
        
        if st.button("🚀 Traiter le lot") and batch_queries:
            queries_list = [q.strip() for q in batch_queries.split('\n') if q.strip()]
            
            if len(queries_list) <= 5:  # Limit for demo
                with st.spinner("Traitement par lot..."):
                    success, result = api_request("POST", "/recommend/batch", json={
                        "queries": queries_list,
                        "k": 3
                    })
                
                if success:
                    results = result.get("results", [])
                    batch_size = result.get("batch_size", 0)
                    avg_latency = result.get("avg_latency_per_query_ms", 0)
                    
                    st.success(f"✅ {batch_size} requêtes traitées en {avg_latency:.1f}ms/requête")
                    
                    for res in results:
                        if "error" not in res:
                            st.write(f"**{res['query']}**: {res['count']} recommandations")
                        else:
                            st.error(f"Erreur pour '{res['query']}': {res['error']}")
            else:
                st.error("Maximum 5 requêtes pour la démo")
        
        # Model comparison with custom parameters
        st.subheader("⚖️ Comparaison avancée")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_query = st.text_input("Requête test", value="RPG Action")
        
        with col2:
            comparison_k = st.number_input("K pour comparaison", 1, 10, 3)
        
        with col3:
            min_confidence = st.slider("Seuil confiance", 0.0, 0.5, 0.1)
        
        if st.button("🔬 Comparaison détaillée") and test_query:
            with st.spinner("Comparaison en cours..."):
                success, result = api_request("POST", "/recommend/compare", json={
                    "query": test_query,
                    "algorithms": ["tfidf", "nmf", "hybrid"],
                    "k": comparison_k
                })
            
            if success:
                comparison_results = result.get("comparison_results", [])
                
                # Create detailed comparison
                comparison_df = pd.DataFrame([
                    {
                        "Algorithme": res["algorithm"].upper(),
                        "Recommandations": len(res.get("recommendations", [])),
                        "Confiance max": max([r.get("confidence", 0) for r in res.get("recommendations", [])], default=0),
                        "Confiance min": min([r.get("confidence", 0) for r in res.get("recommendations", [])], default=0),
                        "Latence (ms)": res.get("latency_ms", 0)
                    }
                    for res in comparison_results if "error" not in res
                ])
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                if not comparison_df.empty:
                    st.bar_chart(comparison_df.set_index("Algorithme")["Latence (ms)"])

# Performance monitoring for admin users
if st.session_state.username in ['admin', 'demo']:  # Admin features
    with st.expander("🛠️ Fonctionnalités administrateur"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Insights données")
            
            if st.button("🔍 Analyser les données"):
                success, result = api_request("POST", "/analyze/data-insights")
                
                if success:
                    insights = result.get("data_insights", {})
                    
                    st.metric("Total jeux", insights.get("total_games", 0))
                    st.metric("Note moyenne", f"{insights.get('avg_rating', 0):.2f}")
                    st.metric("Metacritic moyen", f"{insights.get('avg_metacritic', 0):.1f}")
                    
                    # Top genres
                    top_genres = insights.get("top_genres", {})
                    if top_genres:
                        st.subheader("🎭 Top genres")
                        for genre, count in list(top_genres.items())[:5]:
                            st.write(f"**{genre}**: {count} jeux")
                    
                    # Recommendations for improvement
                    recommendations = insights.get("improvement_recommendations", [])
                    if recommendations:
                        st.subheader("💡 Recommandations")
                        for rec in recommendations:
                            st.info(rec)
        
        with col2:
            st.subheader("🚨 Upload dataset")
            
            uploaded_file = st.file_uploader(
                "Charger un dataset CSV",
                type=['csv'],
                help="Format: titre, genres, rating, metacritic, plateformes"
            )
            
            if uploaded_file and st.button("📤 Uploader et entraîner"):
                with st.spinner("Upload et entraînement..."):
                    # This would normally upload the file
                    # For demo, we'll just show a success message
                    st.success("✅ Dataset uploadé et modèles re-entraînés!")
                    st.info("Note: Fonctionnalité de démo - upload réel non implémenté")

# Real-time updates (simulated)
if st.session_state.token:
    # Auto-refresh certain metrics every 30 seconds
    import time
    
    # Add a timestamp to force updates
    current_time = time.time()
    if hasattr(st.session_state, 'last_update'):
        if current_time - st.session_state.last_update > 30:  # 30 seconds
            st.session_state.last_update = current_time
            st.rerun()
    else:
        st.session_state.last_update = current_time

# Help section
with st.expander("❓ Aide et documentation"):
    st.markdown("""
    ### 🎯 Comment utiliser cette plateforme
    
    #### 🔍 Recherche
    - **Par titre**: Recherche exacte dans la base de données
    - **ML Recommendations**: Utilise l'IA pour suggérer des jeux similaires
    - **Profil utilisateur**: Personnalise les recommandations selon vos préférences
    
    #### 🤖 Algorithmes ML
    - **TF-IDF**: Analyse textuelle classique, rapide et fiable
    - **NMF**: Factorisation matricielle pour découvrir des patterns cachés
    - **Hybrid**: Combinaison optimale de TF-IDF et NMF
    
    #### 📊 Classification & Clustering
    - **Classification**: Prédit automatiquement le genre d'un jeu
    - **Clustering**: Groupe les jeux par similarité
    
    #### ⚙️ Gestion des modèles
    - Créer, entraîner et monitorer plusieurs modèles
    - Comparer les performances en temps réel
    - Analyser la dérive des données (drift detection)
    
    #### 📈 Monitoring
    - Métriques de performance en temps réel
    - Santé système et composants
    - Alertes automatiques en cas de problème
    
    ### 🚀 Conseils d'utilisation
    
    1. **Commencez par la recherche simple** pour vous familiariser
    2. **Utilisez les profils utilisateur** pour des recommandations personnalisées
    3. **Comparez les algorithmes** pour voir leurs différences
    4. **Explorez la classification** pour comprendre comment l'IA catégorise
    5. **Surveillez les métriques** pour optimiser les performances
    
    ### 🔧 Dépannage
    
    - **Erreur 401**: Reconnectez-vous (token expiré)
    - **Erreur 500**: L'API redémarre, patientez 30 secondes
    - **Pas de recommandations**: Essayez des mots-clés plus génériques
    - **Lenteur**: Réduisez le nombre de recommandations (K)
    
    ### 📞 Support
    
    En cas de problème, vérifiez le statut de l'API ou contactez l'équipe de développement.
    """)

# Debug info for developers (only in development)
if os.getenv("DEBUG") == "1" and st.session_state.username:
    with st.expander("🐛 Informations de débogage"):
        st.json({
            "user": st.session_state.username,
            "token_present": bool(st.session_state.token),
            "api_url": API_BASE_URL,
            "favorites_count": len(st.session_state.favorites),
            "search_history_count": len(st.session_state.search_history),
            "session_state_keys": list(st.session_state.keys())
        })

# Footer with additional info
st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 0.8em;">
    <p>🎮 Games AI/ML Platform v3.0 | Développé avec FastAPI + Streamlit + scikit-learn</p>
    <p>Algorithmes: TF-IDF, NMF, Hybrid ML | Monitoring: Prometheus + Grafana</p>
    <p>Dernière mise à jour: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>
""", unsafe_allow_html=True)
