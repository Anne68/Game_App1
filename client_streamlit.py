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
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
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
                    st.session_state.api_url = api_url
                    st.session_state.api_key = api_key
                    st.session_state.api_configured = True
                    st.rerun()
                else:
                    st.error("URL et clé requises")
    else:
        status_col1, status_col2, status_col3 = st.columns([3, 1, 1])
        
        with status_col1:
            st.success(f"🟢 Connecté à {st.session_state.api_url}")
        
        with status_col2:
            if st.button("🔄 Test"):
                try:
                    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
                    response = requests.get(f"{st.session_state.api_url}/health", headers=headers, timeout=3)
                    if response.status_code == 200:
                        st.success("✅ OK")
                    else:
                        st.error(f"❌ {response.status_code}")
                except:
                    st.error("❌ Échec")
        
        with status_col3:
            if st.button("⚙️ Changer"):
                st.session_state.api_configured = False
                st.rerun()

st.markdown("---")

# Fonction API simplifiée
def api_call(endpoint, method="GET", data=None):
    if not st.session_state.api_configured:
        st.warning("Configuration API requise")
        return None
    
    url = f"{st.session_state.api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {st.session_state.api_key}"}
    
    try:
        if method == "GET":
            return requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            return requests.post(url, headers=headers, json=data, timeout=10)
        elif method == "PUT":
            return requests.put(url, headers=headers, json=data, timeout=10)
        elif method == "DELETE":
            return requests.delete(url, headers=headers, timeout=10)
    except Exception as e:
        st.error(f"❌ {str(e)}")
        return None

# Interface principale avec onglets simplifiés
tab_analytics, tab_users, tab_content, tab_system = st.tabs([
    "📊 Analytics", 
    "👥 Utilisateurs", 
    "📦 Contenu", 
    "🔧 Système"
])

# ONGLET ANALYTICS - Interface simplifiée
with tab_analytics:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📈 Métriques")
        
        if st.button("Actualiser", type="primary", use_container_width=True):
            response = api_call("analytics/metrics")
            
            if response and response.status_code == 200:
                data = response.json()
                
                st.metric("👥 Utilisateurs", data.get("users", "N/A"))
                st.metric("💰 Revenus", f"{data.get('revenue', 0)}€")
                st.metric("📈 Conversions", f"{data.get('conversion_rate', 0)}%")
                st.metric("🎯 Engagement", f"{data.get('engagement', 0)}%")
            else:
                # Données de démonstration
                st.metric("👥 Utilisateurs", "1,234")
                st.metric("💰 Revenus", "45,678€")
                st.metric("📈 Conversions", "12.5%")
                st.metric("🎯 Engagement", "67%")
    
    with col2:
        st.subheader("📊 Tendances")
        
        # Graphique simplifié
        dates = pd.date_range('2024-01-01', periods=30)
        values = [100 + i*3 + (i%7)*5 for i in range(30)]
        
        fig = px.area(
            x=dates, 
            y=values,
            title="Évolution des performances",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ONGLET UTILISATEURS - Interface épurée
with tab_users:
    search_col, action_col = st.columns([2, 1])
    
    with search_col:
        search_term = st.text_input("🔍 Rechercher un utilisateur", placeholder="Nom, email ou ID")
    
    with action_col:
        st.write("")  # Espace pour aligner
        if st.button("➕ Nouveau", type="primary"):
            st.session_state.show_user_form = True
    
    # Formulaire de création d'utilisateur (modal-like)
    if st.session_state.get('show_user_form', False):
        with st.form("user_form", clear_on_submit=True):
            st.subheader("👤 Nouveau utilisateur")
            
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                name = st.text_input("Nom*")
                role = st.selectbox("Rôle", ["Utilisateur", "Admin", "Modérateur"])
            
            with form_col2:
                email = st.text_input("Email*")
                status = st.selectbox("Statut", ["Actif", "Inactif"])
            
            submit_col1, submit_col2 = st.columns(2)
            
            with submit_col1:
                if st.form_submit_button("✅ Créer", type="primary", use_container_width=True):
                    if name and email:
                        user_data = {"name": name, "email": email, "role": role.lower(), "status": status.lower()}
                        response = api_call("users", method="POST", data=user_data)
                        
                        if response and response.status_code in [200, 201]:
                            st.success("✅ Utilisateur créé!")
                            st.session_state.show_user_form = False
                        else:
                            st.error("❌ Erreur lors de la création")
                    else:
                        st.error("Nom et email requis")
            
            with submit_col2:
                if st.form_submit_button("❌ Annuler", use_container_width=True):
                    st.session_state.show_user_form = False
    
    # Liste des utilisateurs simplifiée
    if search_term and st.button("Rechercher"):
        response = api_call("users", params={"q": search_term})
        
        if response and response.status_code == 200:
            users = response.json()
            if users:
                for user in users[:5]:
                    with st.container():
                        user_col1, user_col2, user_col3 = st.columns([2, 1, 1])
                        
                        with user_col1:
                            st.write(f"**{user.get('name', 'Sans nom')}**")
                            st.caption(user.get('email', 'Pas d\'email'))
                        
                        with user_col2:
                            role = user.get('role', 'user')
                            st.badge(role.title(), type="secondary")
                        
                        with user_col3:
                            if st.button("✏️", key=f"edit_{user.get('id')}", help="Modifier"):
                                st.info("Modification à implémenter")
                        
                        st.divider()

# ONGLET CONTENU - Interface moderne
with tab_content:
    content_action_col1, content_action_col2 = st.columns([1, 1])
    
    with content_action_col1:
        if st.button("📦 Charger le contenu", type="primary", use_container_width=True):
            response = api_call("products")
            
            if response and response.status_code == 200:
                products = response.json()
                
                if products:
                    for i, product in enumerate(products[:3]):
                        with st.container():
                            prod_col1, prod_col2, prod_col3 = st.columns([2, 1, 1])
                            
                            with prod_col1:
                                st.write(f"**{product.get('name', 'Produit')}**")
                                st.caption(f"ID: {product.get('id', 'N/A')}")
                            
                            with prod_col2:
                                price = product.get('price', 0)
                                st.metric("Prix", f"{price}€")
                            
                            with prod_col3:
                                stock = product.get('stock', 0)
                                if stock > 10:
                                    st.success(f"✅ {stock}")
                                elif stock > 0:
                                    st.warning(f"⚠️ {stock}")
                                else:
                                    st.error("❌ 0")
                            
                            st.divider()
            else:
                st.info("Aucun contenu disponible")
    
    with content_action_col2:
        if st.button("➕ Ajouter du contenu", use_container_width=True):
            st.session_state.show_content_form = True
    
    # Formulaire d'ajout de contenu
    if st.session_state.get('show_content_form', False):
        with st.form("content_form"):
            st.subheader("📦 Nouveau contenu")
            
            name = st.text_input("Nom*")
            price = st.number_input("Prix (€)", min_value=0.0, step=0.01)
            description = st.text_area("Description", height=100)
            
            if st.form_submit_button("✅ Ajouter", type="primary"):
                if name:
                    content_data = {"name": name, "price": price, "description": description}
                    response = api_call("products", method="POST", data=content_data)
                    
                    if response and response.status_code in [200, 201]:
                        st.success("✅ Contenu ajouté!")
                        st.session_state.show_content_form = False
                    else:
                        st.error("❌ Erreur lors de l'ajout")
                else:
                    st.error("Nom requis")

# ONGLET SYSTÈME - Interface de monitoring simplifiée
with tab_system:
    # Statut des services en une ligne
    st.subheader("🚦 Statut des services")
    
    services = ["API", "Base de données", "Cache", "Stockage"]
    service_cols = st.columns(len(services))
    
    for i, service in enumerate(services):
        with service_cols[i]:
            # Simulation de statut (remplacez par vos vraies vérifications)
            status = "ok" if i < 3 else "error"  # Simulation
            
            if status == "ok":
                st.markdown(f'<div class="status-ok">✅ {service}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-error">❌ {service}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Actions système simplifiées
    st.subheader("⚙️ Actions rapides")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔄 Redémarrer", use_container_width=True):
            with st.spinner("Redémarrage..."):
                response = api_call("admin/restart", method="POST")
                if response and response.status_code == 200:
                    st.success("✅ Redémarré!")
                else:
                    st.error("❌ Échec")
    
    with action_col2:
        if st.button("🗑️ Nettoyer", use_container_width=True):
            response = api_call("admin/clear-cache", method="POST")
            if response and response.status_code == 200:
                st.success("✅ Nettoyé!")
            else:
                st.error("❌ Échec")
    
    with action_col3:
        if st.button("💾 Sauvegarder", use_container_width=True):
            response = api_call("admin/backup", method="POST")
            if response and response.status_code == 200:
                st.success("✅ Sauvegardé!")
            else:
                st.error("❌ Échec")
    
    # Logs récents (compacts)
    st.subheader("📝 Logs récents")
    
    if st.button("Actualiser les logs"):
        response = api_call("admin/logs", params={"limit": 5})
        
        if response and response.status_code == 200:
            logs = response.json()
            
            for log in logs:
                timestamp = log.get('timestamp', 'N/A')
                level = log.get('level', 'INFO')
                message = log.get('message', 'Pas de message')
                
                # Affichage compact des logs
                if level == "ERROR":
                    st.error(f"🔴 {timestamp} | {message}")
                elif level == "WARNING":
                    st.warning(f"🟡 {timestamp} | {message}")
                else:
                    st.info(f"🔵 {timestamp} | {message}")
        else:
            # Logs de démonstration
            st.info("🔵 2024-09-20 11:20:15 | Système démarré avec succès")
            st.info("🔵 2024-09-20 11:18:32 | Connexion utilisateur établie")
            st.warning("🟡 2024-09-20 11:15:20 | Utilisation mémoire élevée (85%)")

# Footer minimaliste
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.8rem;">⚡ API Hub - Interface simplifiée</div>',
    unsafe_allow_html=True
)
