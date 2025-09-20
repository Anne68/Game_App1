import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="API Hub",
    page_icon="⚡",
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
</style>
""", unsafe_allow_html=True)

# État de session pour la configuration API
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'api_url' not in st.session_state:
    st.session_state.api_url = ""
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Configuration API compacte en haut de page
st.markdown('<div class="main-header"><h1>⚡ API Hub</h1><p>Interface unifiée pour vos APIs</p></div>', unsafe_allow_html=True)

# Configuration API dans un conteneur compact
with st.container():
    if not st.session_state.api_configured:
        st.info("🔧 Configuration rapide requise")
        
        config_col1, config_col2, config_col3 = st.columns([2, 2, 1])
        
        with config_col1:
            api_url = st.text_input("🌐 URL API", placeholder="https://api.exemple.com", label_visibility="collapsed")
        
        with config_col2:
            api_key = st.text_input("🔑 Clé API", type="password", placeholder="Votre clé API", label_visibility="collapsed")
        
        with config_col3:
            if st.button("✅ Connecter", type="primary"):
                if api_url and api_key:
                    st.session_state.api_url = api_url.rstrip('/')
                    st.session_state.api_key = api_key
                    st.session_state.api_configured = True
                    st.rerun()
                else:
                    st.error("URL et clé requises")
    else:
        status_col1, status_col2, status_col3, status_col4 = st.columns([2, 1, 1, 1])
        
        with status_col1:
            st.success(f"🟢 Connecté à {st.session_state.api_url}")
        
        with status_col2:
            if st.button("🔄 Test"):
                try:
                    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
                    test_url = f"{st.session_state.api_url}/health"
                    
                    if st.session_state.debug_mode:
                        st.markdown(f'<div class="debug-info">Test URL: {test_url}</div>', unsafe_allow_html=True)
                    
                    response = requests.get(test_url, headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        st.success("✅ OK")
                    else:
                        st.error(f"❌ {response.status_code}")
                        if st.session_state.debug_mode:
                            st.text(f"Réponse: {response.text[:200]}")
                            
                except requests.exceptions.Timeout:
                    st.error("❌ Timeout")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Connexion impossible")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")
        
        with status_col3:
            if st.button("🐛 Debug"):
                st.session_state.debug_mode = not st.session_state.debug_mode
                st.rerun()
        
        with status_col4:
            if st.button("⚙️ Changer"):
                st.session_state.api_configured = False
                st.session_state.debug_mode = False
                st.rerun()

# Mode debug
if st.session_state.debug_mode and st.session_state.api_configured:
    with st.expander("🐛 Informations de debug", expanded=True):
        st.markdown("**Configuration actuelle:**")
        st.code(f"""
URL API: {st.session_state.api_url}
Clé API: {'*' * (len(st.session_state.api_key) - 4) + st.session_state.api_key[-4:] if st.session_state.api_key else 'Non définie'}
Headers: Authorization: Bearer [MASQUÉ]
        """)

st.markdown("---")

# Fonction API simplifiée avec debug amélioré
def api_call(endpoint, method="GET", data=None, params=None):
    if not st.session_state.api_configured:
        st.warning("Configuration API requise")
        return None
    
    # Construction de l'URL
    url = f"{st.session_state.api_url}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
    
    # Debug info
    if st.session_state.debug_mode:
        st.markdown(f'<div class="debug-info">🔍 {method} {url}</div>', unsafe_allow_html=True)
        if params:
            st.markdown(f'<div class="debug-info">📋 Params: {params}</div>', unsafe_allow_html=True)
        if data:
            st.markdown(f'<div class="debug-info">📦 Data: {json.dumps(data, indent=2)}</div>', unsafe_allow_html=True)
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=10)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=10)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=10)
        
        # Debug de la réponse
        if st.session_state.debug_mode:
            st.markdown(f'<div class="debug-info">📡 Status: {response.status_code}</div>', unsafe_allow_html=True)
            if response.status_code != 200:
                st.markdown(f'<div class="debug-info">❌ Erreur: {response.text[:300]}</div>', unsafe_allow_html=True)
        
        return response
        
    except requests.exceptions.Timeout:
        st.error("❌ Timeout de la requête")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Erreur de connexion - Vérifiez l'URL")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Erreur HTTP: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Erreur inattendue: {str(e)}")
        return None

# Interface principale avec onglets simplifiés
tab_search, tab_recommendations, tab_analytics, tab_admin = st.tabs([
    "🔍 Recherche", 
    "⭐ Recommandations", 
    "📊 Analytics", 
    "🔧 Admin"
])

# ONGLET RECHERCHE - Adapté aux captures d'écran
with tab_search:
    st.header("🔍 Recherche avancée de jeux")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("Rechercher un jeu", placeholder="cyberpunk, zelda, mario...")
    
    with col2:
        search_type = st.selectbox("Type de recherche", ["Par titre", "Par genre", "Par plateforme"])
    
    if st.button("🔍 Rechercher", type="primary", use_container_width=True):
        # Adaptation selon vos captures d'écran
        search_params = {
            "query": search_query,
            "type": search_type.lower().replace(" ", "_")
        }
        
        response = api_call("search", params=search_params)
        
        if response and response.status_code == 200:
            try:
                results = response.json()
                
                if results and len(results) > 0:
                    st.success(f"✅ {len(results)} résultat(s) trouvé(s)")
                    
                    for game in results[:5]:  # Limiter à 5 résultats
                        with st.container():
                            game_col1, game_col2, game_col3 = st.columns([3, 1, 1])
                            
                            with game_col1:
                                st.write(f"**{game.get('title', 'Titre inconnu')}**")
                                st.caption(f"Genre: {game.get('genre', 'N/A')} | Plateforme: {game.get('platform', 'N/A')}")
                            
                            with game_col2:
                                rating = game.get('rating', 0)
                                st.metric("Note", f"{rating}/10")
                            
                            with game_col3:
                                if st.button("📋 Détails", key=f"detail_{game.get('id', 0)}"):
                                    st.info(f"Détails pour: {game.get('title', 'Jeu')}")
                            
                            st.divider()
                else:
                    st.warning("❌ Aucun résultat trouvé")
            except json.JSONDecodeError:
                st.error("❌ Erreur de format de réponse")
        elif response and response.status_code == 404:
            st.error("❌ Erreur de recherche: Not Found - Vérifiez l'endpoint")
        else:
            st.error("❌ Erreur lors de la recherche")

# ONGLET RECOMMANDATIONS - Basé sur vos captures d'écran
with tab_recommendations:
    st.header("⭐ Recommandations ML Avancées")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        
        # Profil utilisateur (optionnel)
        with st.expander("🎯 Profil utilisateur (optionnel)", expanded=False):
            user_genres = st.multiselect(
                "Genres préférés", 
                ["RPG", "Action", "Aventure", "Sport", "Stratégie", "Simulation"],
                default=["RPG"]
            )
            user_platforms = st.multiselect(
                "Plateformes préférées",
                ["PC", "Xbox", "PlayStation", "Nintendo Switch"],
                default=["PC"]
            )
        
        # Paramètres de recommandation
        num_recommendations = st.slider("Nombre de suggestions", 1, 10, 2)
        algorithm = st.selectbox("Algorithme", ["hybrid", "collaborative", "content-based"])
        confidence_min = st.slider("Confiance minimum", 0.0, 1.0, 0.1, 0.05)
    
    with col2:
        st.subheader("Obtenir des recommandations")
        
        if st.button("🎯 Obtenir des recommandations", type="primary", use_container_width=True):
            # Préparation des données pour l'API
            recommendation_data = {
                "user_profile": {
                    "preferred_genres": user_genres,
                    "preferred_platforms": user_platforms
                },
                "algorithm": algorithm,
                "num_recommendations": num_recommendations,
                "min_confidence": confidence_min
            }
            
            with st.spinner("Génération des recommandations..."):
                response = api_call("recommendations", method="POST", data=recommendation_data)
                
                if response and response.status_code == 200:
                    try:
                        recommendations = response.json()
                        
                        if recommendations and len(recommendations) > 0:
                            st.success(f"✅ {len(recommendations)} recommandation(s) générée(s)")
                            
                            for i, rec in enumerate(recommendations):
                                with st.container():
                                    rec_col1, rec_col2, rec_col3 = st.columns([2, 1, 1])
                                    
                                    with rec_col1:
                                        st.write(f"**{rec.get('title', 'Jeu recommandé')}**")
                                        st.caption(f"Confiance: {rec.get('confidence', 0):.2f}")
                                    
                                    with rec_col2:
                                        score = rec.get('score', 0)
                                        st.metric("Score", f"{score:.1f}")
                                    
                                    with rec_col3:
                                        st.write(f"**{rec.get('reason', 'Recommandé pour vous')}**")
                                    
                                    st.divider()
                        else:
                            st.warning("❌ Aucune recommandation générée")
                    except json.JSONDecodeError:
                        st.error("❌ Erreur de format de réponse")
                elif response and response.status_code == 404:
                    st.error("❌ Endpoint de recommandations non trouvé")
                else:
                    st.error("❌ Erreur lors de la génération des recommandations")

# ONGLET ANALYTICS - Interface simplifiée
with tab_analytics:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📈 Métriques")
        
        if st.button("Actualiser", type="primary", use_container_width=True):
            response = api_call("analytics/metrics")
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    
                    st.metric("👥 Utilisateurs", data.get("total_users", "N/A"))
                    st.metric("🎮 Jeux", data.get("total_games", "N/A"))
                    st.metric("⭐ Recommandations", data.get("recommendations_served", "N/A"))
                    st.metric("🎯 Taux de succès", f"{data.get('success_rate', 0)}%")
                except json.JSONDecodeError:
                    st.error("❌ Erreur de format des métriques")
            else:
                # Données de démonstration
                st.metric("👥 Utilisateurs", "1,234")
                st.metric("🎮 Jeux", "15,678")
                st.metric("⭐ Recommandations", "45,123")
                st.metric("🎯 Taux de succès", "87%")
    
    with col2:
        st.subheader("📊 Tendances")
        
        # Graphique simplifié
        dates = pd.date_range('2024-01-01', periods=30)
        values = [100 + i*3 + (i%7)*5 for i in range(30)]
        
        fig = px.area(
            x=dates, 
            y=values,
            title="Évolution des recommandations",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ONGLET ADMIN - Interface de monitoring
with tab_admin:
    # Statut des services
    st.subheader("🚦 Statut des services")
    
    services = ["API", "ML Models", "Database", "Cache"]
    service_cols = st.columns(len(services))
    
    for i, service in enumerate(services):
        with service_cols[i]:
            if st.button(f"Test {service}", key=f"test_{service}"):
                response = api_call(f"health/{service.lower()}")
                
                if response and response.status_code == 200:
                    st.markdown(f'<div class="status-ok">✅ {service}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-error">❌ {service}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Logs et debug
    st.subheader("📝 Logs et debug")
    
    log_col1, log_col2 = st.columns(2)
    
    with log_col1:
        if st.button("📋 Récupérer les logs"):
            response = api_call("admin/logs", params={"limit": 10})
            
            if response and response.status_code == 200:
                try:
                    logs = response.json()
                    
                    for log in logs[:5]:
                        timestamp = log.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        level = log.get('level', 'INFO')
                        message = log.get('message', 'Message de log')
                        
                        if level == "ERROR":
                            st.error(f"🔴 {timestamp} | {message}")
                        elif level == "WARNING":
                            st.warning(f"🟡 {timestamp} | {message}")
                        else:
                            st.info(f"🔵 {timestamp} | {message}")
                except json.JSONDecodeError:
                    st.error("❌ Erreur de format des logs")
            else:
                # Logs de démonstration
                st.info("🔵 2024-09-20 11:20:15 | API démarrée avec succès")
                st.info("🔵 2024-09-20 11:18:32 | Modèle ML chargé")
                st.warning("🟡 2024-09-20 11:15:20 | Latence élevée détectée")
    
    with log_col2:
        if st.button("🔄 Test complet de l'API"):
            st.write("**Test des endpoints principaux:**")
            
            endpoints = [
                ("search", "GET"),
                ("recommendations", "POST"),
                ("analytics/metrics", "GET"),
                ("health", "GET")
            ]
            
            for endpoint, method in endpoints:
                test_data = {"test": True} if method == "POST" else None
                response = api_call(endpoint, method=method, data=test_data)
                
                if response:
                    if response.status_code == 200:
                        st.success(f"✅ {endpoint} ({method})")
                    else:
                        st.error(f"❌ {endpoint} ({method}) - {response.status_code}")
                else:
                    st.error(f"❌ {endpoint} ({method}) - Pas de réponse")

# Footer avec informations utiles
st.markdown("---")
if st.session_state.debug_mode:
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.8rem;">🐛 Mode debug activé | ⚡ API Hub - Interface de test</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.8rem;">⚡ API Hub - Interface simplifiée</div>',
        unsafe_allow_html=True
    )
