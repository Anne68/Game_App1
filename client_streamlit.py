import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="API Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration API dans la sidebar
with st.sidebar:
    st.title("⚙️ Configuration API")
    
    # Configuration de l'API
    api_base_url = st.text_input(
        "URL de base de l'API",
        value="https://api.exemple.com",
        help="URL de base sans le slash final"
    )
    
    api_key = st.text_input(
        "Clé API",
        type="password",
        help="Votre clé d'authentification API"
    )
    
    # Headers personnalisés
    st.subheader("Headers personnalisés")
    custom_headers = {}
    if api_key:
        custom_headers["Authorization"] = f"Bearer {api_key}"
    
    # Ajouter des headers personnalisés
    add_header = st.checkbox("Ajouter des headers personnalisés")
    if add_header:
        header_key = st.text_input("Nom du header")
        header_value = st.text_input("Valeur du header")
        if header_key and header_value:
            custom_headers[header_key] = header_value
    
    # Test de connexion
    if st.button("🔍 Tester la connexion"):
        try:
            response = requests.get(f"{api_base_url}/health", headers=custom_headers, timeout=5)
            if response.status_code == 200:
                st.success("✅ Connexion réussie!")
            else:
                st.error(f"❌ Erreur: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Erreur de connexion: {str(e)}")

# Fonction utilitaire pour faire des requêtes API
def make_api_request(endpoint, method="GET", data=None, params=None):
    """Fonction utilitaire pour effectuer des requêtes API"""
    url = f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=custom_headers, params=params, timeout=10)
        elif method == "POST":
            response = requests.post(url, headers=custom_headers, json=data, timeout=10)
        elif method == "PUT":
            response = requests.put(url, headers=custom_headers, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=custom_headers, timeout=10)
        
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de requête: {str(e)}")
        return None

# Titre principal
st.title("🚀 Dashboard API Multi-Usage")
st.markdown("---")

# Création des onglets
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Données Analytics", 
    "👥 Gestion Utilisateurs", 
    "📦 Produits/Services", 
    "📈 Monitoring", 
    "🛠️ Administration"
])

# ONGLET 1: DONNÉES ANALYTICS
with tab1:
    st.header("📊 Analytics et Rapports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Métriques générales")
        
        # Sélection de la période
        date_range = st.date_input(
            "Période d'analyse",
            value=[datetime.now().date()],
            help="Sélectionnez la période pour l'analyse"
        )
        
        if st.button("📈 Récupérer les métriques"):
            params = {
                "start_date": str(date_range[0]) if date_range else None,
                "end_date": str(date_range[-1]) if len(date_range) > 1 else None
            }
            
            response = make_api_request("analytics/metrics", params=params)
            
            if response and response.status_code == 200:
                data = response.json()
                
                # Affichage des métriques
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric("Utilisateurs actifs", data.get("active_users", "N/A"))
                with metrics_col2:
                    st.metric("Revenus", f"{data.get('revenue', 0)}€")
                with metrics_col3:
                    st.metric("Conversions", data.get("conversions", 0))
                with metrics_col4:
                    st.metric("Taux de conversion", f"{data.get('conversion_rate', 0)}%")
    
    with col2:
        st.subheader("Graphiques")
        
        # Exemple de graphique
        if st.button("📊 Générer graphique"):
            # Données d'exemple (remplacez par vos vraies données API)
            sample_data = {
                'Date': pd.date_range('2024-01-01', periods=30),
                'Ventes': [100 + i*5 + (i%7)*10 for i in range(30)],
                'Visiteurs': [200 + i*3 + (i%5)*15 for i in range(30)]
            }
            df = pd.DataFrame(sample_data)
            
            fig = px.line(df, x='Date', y=['Ventes', 'Visiteurs'], 
                         title="Évolution des ventes et visiteurs")
            st.plotly_chart(fig, use_container_width=True)

# ONGLET 2: GESTION UTILISATEURS
with tab2:
    st.header("👥 Gestion des Utilisateurs")
    
    # Recherche d'utilisateurs
    st.subheader("🔍 Rechercher des utilisateurs")
    search_query = st.text_input("Recherche (nom, email, ID)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Rechercher"):
            params = {"q": search_query} if search_query else {}
            response = make_api_request("users", params=params)
            
            if response and response.status_code == 200:
                users = response.json()
                if users:
                    df_users = pd.DataFrame(users)
                    st.dataframe(df_users, use_container_width=True)
                else:
                    st.info("Aucun utilisateur trouvé")
    
    with col2:
        st.subheader("➕ Créer un utilisateur")
        
        with st.form("create_user"):
            new_user_name = st.text_input("Nom")
            new_user_email = st.text_input("Email")
            new_user_role = st.selectbox("Rôle", ["user", "admin", "moderator"])
            
            submitted = st.form_submit_button("Créer l'utilisateur")
            
            if submitted:
                user_data = {
                    "name": new_user_name,
                    "email": new_user_email,
                    "role": new_user_role
                }
                
                response = make_api_request("users", method="POST", data=user_data)
                
                if response and response.status_code in [200, 201]:
                    st.success("✅ Utilisateur créé avec succès!")
                else:
                    st.error("❌ Erreur lors de la création")

# ONGLET 3: PRODUITS/SERVICES
with tab3:
    st.header("📦 Gestion des Produits/Services")
    
    # Liste des produits
    st.subheader("📋 Liste des produits")
    
    if st.button("📦 Charger les produits"):
        response = make_api_request("products")
        
        if response and response.status_code == 200:
            products = response.json()
            
            if products:
                for product in products[:5]:  # Limiter à 5 produits
                    with st.expander(f"🏷️ {product.get('name', 'Produit sans nom')}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**ID:** {product.get('id')}")
                            st.write(f"**Prix:** {product.get('price', 'N/A')}€")
                        
                        with col2:
                            st.write(f"**Stock:** {product.get('stock', 'N/A')}")
                            st.write(f"**Catégorie:** {product.get('category', 'N/A')}")
                        
                        with col3:
                            if st.button(f"✏️ Modifier", key=f"edit_{product.get('id')}"):
                                st.info("Fonction de modification à implémenter")
                            if st.button(f"🗑️ Supprimer", key=f"delete_{product.get('id')}"):
                                st.warning("Fonction de suppression à implémenter")
    
    # Ajouter un nouveau produit
    st.subheader("➕ Ajouter un produit")
    
    with st.form("add_product"):
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.text_input("Nom du produit")
            product_price = st.number_input("Prix", min_value=0.0, step=0.01)
        
        with col2:
            product_stock = st.number_input("Stock", min_value=0, step=1)
            product_category = st.text_input("Catégorie")
        
        product_description = st.text_area("Description")
        
        submitted = st.form_submit_button("Ajouter le produit")
        
        if submitted:
            product_data = {
                "name": product_name,
                "price": product_price,
                "stock": product_stock,
                "category": product_category,
                "description": product_description
            }
            
            response = make_api_request("products", method="POST", data=product_data)
            
            if response and response.status_code in [200, 201]:
                st.success("✅ Produit ajouté avec succès!")
            else:
                st.error("❌ Erreur lors de l'ajout")

# ONGLET 4: MONITORING
with tab4:
    st.header("📈 Monitoring et Performance")
    
    # Statut des services
    st.subheader("🚦 Statut des services")
    
    if st.button("🔄 Vérifier le statut"):
        services = ["api", "database", "cache", "storage"]
        
        cols = st.columns(len(services))
        
        for i, service in enumerate(services):
            with cols[i]:
                # Simulation du statut (remplacez par vos vraies vérifications)
                response = make_api_request(f"health/{service}")
                
                if response and response.status_code == 200:
                    st.success(f"✅ {service.upper()}")
                else:
                    st.error(f"❌ {service.upper()}")
    
    # Métriques de performance
    st.subheader("⚡ Métriques de performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Temps de réponse"):
            # Simulation de données (remplacez par vos vraies métriques)
            response_times = [120, 150, 98, 200, 175, 130, 145]
            timestamps = pd.date_range('2024-01-01', periods=7)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=response_times,
                mode='lines+markers',
                name='Temps de réponse (ms)'
            ))
            fig.update_layout(title="Temps de réponse API")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if st.button("🔄 Utilisation CPU/Mémoire"):
            # Simulation de données
            cpu_usage = [45, 52, 38, 60, 55, 42, 48]
            memory_usage = [65, 70, 58, 75, 72, 68, 63]
            timestamps = pd.date_range('2024-01-01', periods=7)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=cpu_usage, name='CPU %'))
            fig.add_trace(go.Scatter(x=timestamps, y=memory_usage, name='Mémoire %'))
            fig.update_layout(title="Utilisation des ressources")
            st.plotly_chart(fig, use_container_width=True)

# ONGLET 5: ADMINISTRATION
with tab5:
    st.header("🛠️ Administration")
    
    # Logs système
    st.subheader("📝 Logs système")
    
    log_level = st.selectbox("Niveau de log", ["INFO", "WARNING", "ERROR", "DEBUG"])
    
    if st.button("📋 Récupérer les logs"):
        params = {"level": log_level, "limit": 50}
        response = make_api_request("admin/logs", params=params)
        
        if response and response.status_code == 200:
            logs = response.json()
            
            for log in logs[:10]:  # Afficher les 10 derniers logs
                timestamp = log.get('timestamp', 'N/A')
                level = log.get('level', 'INFO')
                message = log.get('message', 'Pas de message')
                
                if level == "ERROR":
                    st.error(f"[{timestamp}] {level}: {message}")
                elif level == "WARNING":
                    st.warning(f"[{timestamp}] {level}: {message}")
                else:
                    st.info(f"[{timestamp}] {level}: {message}")
    
    st.markdown("---")
    
    # Actions administratives
    st.subheader("⚙️ Actions administratives")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Redémarrer les services"):
            with st.spinner("Redémarrage en cours..."):
                response = make_api_request("admin/restart", method="POST")
                if response and response.status_code == 200:
                    st.success("✅ Services redémarrés!")
                else:
                    st.error("❌ Échec du redémarrage")
    
    with col2:
        if st.button("🗑️ Nettoyer le cache"):
            response = make_api_request("admin/clear-cache", method="POST")
            if response and response.status_code == 200:
                st.success("✅ Cache nettoyé!")
            else:
                st.error("❌ Échec du nettoyage")
    
    with col3:
        if st.button("💾 Sauvegarder"):
            response = make_api_request("admin/backup", method="POST")
            if response and response.status_code == 200:
                st.success("✅ Sauvegarde créée!")
            else:
                st.error("❌ Échec de la sauvegarde")
    
    # Configuration avancée
    st.subheader("🔧 Configuration avancée")
    
    with st.expander("Paramètres de configuration"):
        config_key = st.text_input("Clé de configuration")
        config_value = st.text_input("Valeur")
        
        if st.button("💾 Sauvegarder la configuration"):
            config_data = {config_key: config_value}
            response = make_api_request("admin/config", method="PUT", data=config_data)
            
            if response and response.status_code == 200:
                st.success("✅ Configuration mise à jour!")
            else:
                st.error("❌ Erreur de configuration")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>🚀 Dashboard API Multi-Usage | Développé avec Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)
# Configuration de la page
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
            # Utiliser l'endpoint exact de votre API
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

    # ONGLET RECOMMANDATIONS ML
    with tab_recommendations:
        st.header("⭐ Recommandations ML Avancées")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Configuration")
            
            # Paramètres de recommandation selon votre API
            user_query = st.text_input("Décrivez ce que vous cherchez", value="RPG fantasy", placeholder="RPG, action, aventure...")
            num_recommendations = st.slider("Nombre de suggestions", 1, 20, 10)
            confidence_min = st.slider("Confiance minimum", 0.0, 1.0, 0.1, 0.05)
        
        with col2:
            st.subheader("Obtenir des recommandations")
            
            if st.button("🎯 Obtenir des recommandations ML", type="primary", use_container_width=True):
                # Utiliser l'endpoint exact de votre API
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
                            st.info(f"Modèle: {results.get('model_version', 'unknown')} | Latence: {results.get('latency_ms', 0):.1f}ms")
                            
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
                        st.error("❌ Erreur lors de la génération des recommandations")
        
        # Section recommandations par genre
        st.markdown("---")
        st.subheader("🎮 Recommandations par genre")
        
        genre_col1, genre_col2 = st.columns([1, 1])
        
        with genre_col1:
            genre = st.selectbox("Choisir un genre", ["RPG", "Action", "Adventure", "Strategy", "Simulation", "Sports", "Racing", "Indie"])
            genre_limit = st.slider("Nombre de jeux", 1, 20, 5, key="genre_limit")
        
        with genre_col2:
            st.write("")  # Espace
            if st.button("🎯 Recommandations par genre", use_container_width=True):
                response = api_call(f"/recommend/by-genre/{genre}", params={"k": genre_limit})
                
                if response and response.status_code == 200:
                    try:
                        results = response.json()
                        st.success(f"✅ Jeux du genre '{results.get('genre', genre)}'")
                        
                        for game in results.get("recommendations", []):
                            with st.container():
                                game_col1, game_col2 = st.columns([3, 1])
                                
                                with game_col1:
                                    st.write(f"**{game.get('title', 'Titre inconnu')}**")
                                    st.caption(f"Genres: {game.get('genres', 'N/A')}")
                                
                                with game_col2:
                                    st.metric("Note", f"{game.get('rating', 0):.1f}")
                                
                                st.divider()
                                
                    except json.JSONDecodeError:
                        st.error("❌ Erreur de format")

    # ONGLET GESTION MODÈLE
    with tab_model:
        st.header("🤖 Gestion du modèle ML")
        
        # Métriques du modèle
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Métriques du modèle")
            
            if st.button("📈 Récupérer les métriques", type="primary"):
                response = api_call("/model/metrics")
                
                if response and response.status_code == 200:
                    try:
                        metrics = response.json()
                        
                        st.metric("Version du modèle", metrics.get("model_version", "unknown"))
                        st.metric("Modèle entraîné", "✅ Oui" if metrics.get("is_trained", False) else "❌ Non")
                        st.metric("Total prédictions", metrics.get("total_predictions", 0))
                        st.metric("Confiance moyenne", f"{metrics.get('avg_confidence', 0):.3f}")
                        st.metric("Nombre de jeux", metrics.get("games_count", 0))
                        st.metric("Dimension des features", metrics.get("feature_dimension", 0))
                        
                        if metrics.get("last_training"):
                            st.info(f"Dernier entraînement: {metrics['last_training']}")
                        
                    except json.JSONDecodeError:
                        st.error("❌ Erreur de format des métriques")
        
        with col2:
            st.subheader("🔄 Actions sur le modèle")
            
            # Entraînement du modèle
            with st.form("train_model"):
                st.write("**Entraîner le modèle**")
                
                version = st.text_input("Version (optionnel)", placeholder="v1.0.0")
                force_retrain = st.checkbox("Forcer le ré-entraînement")
                
                if st.form_submit_button("🚀 Entraîner le modèle", type="primary"):
                    train_data = {
                        "version": version if version else None,
                        "force_retrain": force_retrain
                    }
                    
                    with st.spinner("Entraînement en cours..."):
                        response = api_call("/model/train", method="POST", data=train_data)
                        
                        if response and response.status_code == 200:
                            try:
                                result = response.json()
                                st.success(f"✅ Entraînement terminé!")
                                st.info(f"Version: {result.get('version', 'unknown')}")
                                st.info(f"Durée: {result.get('duration', 0):.2f} secondes")
                                
                                if result.get('result'):
                                    st.json(result['result'])
                                    
                            except json.JSONDecodeError:
                                st.error("❌ Erreur de format de réponse")
                        else:
                            st.error("❌ Erreur lors de l'entraînement")
            
            # Évaluation du modèle
            st.markdown("---")
            
            if st.button("🧪 Évaluer le modèle"):
                test_queries = ["RPG", "Action", "Indie", "Simulation"]
                
                with st.spinner("Évaluation en cours..."):
                    params = {"test_queries": test_queries}
                    response = api_call("/model/evaluate", method="POST", params=params)
                    
                    if response and response.status_code == 200:
                        try:
                            evaluation = response.json()
                            st.success("✅ Évaluation terminée")
                            st.json(evaluation)
                        except json.JSONDecodeError:
                            st.error("❌ Erreur de format d'évaluation")

    # ONGLET ANALYTICS
    with tab_analytics:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📈 Statut système")
            
            if st.button("🔄 Actualiser le statut", type="primary"):
                response = api_call("/healthz")
                
                if response and response.status_code == 200:
                    try:
                        health = response.json()
                        
                        status = health.get("status", "unknown")
                        if status == "healthy":
                            st.success("🟢 Système en bonne santé")
                        else:
                            st.warning(f"⚠️ Système: {status}")
                        
                        st.metric("Base de données", "✅ Connectée" if health.get("db_ready", False) else "❌ Déconnectée")
                        st.metric("Modèle ML", "✅ Chargé" if health.get("model_loaded", False) else "❌ Non chargé")
                        st.metric("Version modèle", health.get("model_version", "unknown"))
                        st.metric("Compliance", "✅ Activée" if health.get("compliance_enabled", False) else "❌ Désactivée")
                        
                        if health.get("db_error"):
                            st.error(f"Erreur DB: {health['db_error']}")
                        
                        # Métriques de monitoring
                        monitoring = health.get("monitoring", {})
                        if monitoring:
                            st.markdown("**Métriques de monitoring:**")
                            for key, value in monitoring.items():
                                st.text(f"{key}: {value}")
                        
                    except json.JSONDecodeError:
                        st.error("❌ Erreur de format du statut")
        
        with col2:
            st.subheader("📊 Métriques Prometheus")
            
            if st.button("📈 Voir les métriques Prometheus"):
                response = api_call("/metrics")
                
                if response and response.status_code == 200:
                    st.text_area("Métriques Prometheus", response.text, height=400)
                else:
                    st.error("❌ Impossible de récupérer les métriques")

else:
    st.info("👆 Veuillez vous connecter pour accéder aux fonctionnalités")

# Footer with proper string concatenation
st.markdown("---")
if st.session_state.debug_mode:
    # Fixed string concatenation issue by ensuring all variables are properly defined
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
