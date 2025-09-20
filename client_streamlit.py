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
