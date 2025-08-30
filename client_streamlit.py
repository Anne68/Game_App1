import os
import time
import requests
import streamlit as st
from typing import Dict, List, Optional
import json

# =============================================================================
# Configuration
# =============================================================================
API_BASE_URL = (
    st.secrets.get("API_URL") if hasattr(st, 'secrets') and st.secrets else
    os.getenv("API_URL") or
    "https://game-app-y8be.onrender.com"
)

st.set_page_config(
    page_title="Games UI", 
    page_icon="🎮", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'favorites' not in st.session_state:
    st.session_state.favorites = set()
if 'last_search' not in st.session_state:
    st.session_state.last_search = ""
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = "unknown"

# =============================================================================
# API Helper Functions
# =============================================================================
def api_headers() -> Dict[str, str]:
    """Retourne les headers pour les requêtes API"""
    headers = {"accept": "application/json"}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    return headers

def api_request(method: str, path: str, **kwargs) -> tuple:
    """Fonction générique pour les requêtes API avec gestion d'erreur"""
    url = f"{API_BASE_URL}{path}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=api_headers(), timeout=30, **kwargs)
        elif method.upper() == "POST":
            response = requests.post(url, headers=api_headers(), timeout=60, **kwargs)
        else:
            raise ValueError(f"Méthode HTTP non supportée: {method}")
        
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except:
                error_detail = response.text or f"Erreur HTTP {response.status_code}"
            return False, error_detail
            
        return True, response.json()
        
    except requests.exceptions.Timeout:
        return False, "⏰ Timeout - L'API met trop de temps à répondre"
    except requests.exceptions.ConnectionError:
        return False, "🌐 Erreur de connexion - Vérifiez que l'API est accessible"
    except requests.exceptions.RequestException as e:
        return False, f"🚨 Erreur de requête: {str(e)}"
    except Exception as e:
        return False, f"❌ Erreur inattendue: {str(e)}"

def check_api_health():
    """Vérifie la santé de l'API"""
    success, result = api_request("GET", "/healthz")
    if success:
        st.session_state.api_status = result.get("status", "unknown")
        return True, result
    else:
        st.session_state.api_status = "error"
        return False, result

# =============================================================================
# Styles CSS
# =============================================================================
st.markdown("""
<style>
/* Variables CSS */
:root { 
    --card-bg: rgba(255,255,255,0.7); 
    --soft-border: 1px solid rgba(0,0,0,0.1); 
    --success-color: #00D4AA;
    --error-color: #FF4B4B;
    --warning-color: #FFB800;
}

/* Header Hero */
.app-hero { 
    display: flex; 
    gap: 15px; 
    align-items: center; 
    margin: 10px 0; 
    padding: 20px; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    color: white;
}
.app-hero .logo { font-size: 32px; }
.app-hero .title { font-size: 28px; font-weight: 800; }

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 5px;
}
.status-healthy { background: var(--success-color); color: white; }
.status-error { background: var(--error-color); color: white; }
.status-warning { background: var(--warning-color); color: white; }

/* Cards */
.game-card { 
    border: var(--soft-border); 
    border-radius: 12px; 
    padding: 16px; 
    background: var(--card-bg);
    margin: 10px 0; 
    transition: transform 0.2s;
}
.game-card:hover { transform: translateY(-2px); }

.game-card h4 { 
    margin: 0 0 8px 0; 
    font-weight: 800; 
    color: #1f2937;
}

/* Metadata */
.meta { 
    opacity: 0.8; 
    font-size: 0.9rem; 
    color: #6b7280;
}

/* Platform badges */
.platform-badge { 
    display: inline-block; 
    padding: 3px 8px; 
    border-radius: 12px; 
    font-size: 0.75rem; 
    margin: 2px; 
    border: var(--soft-border);
}
.platform-pc { background: #e7f1ff; color: #1e40af; }
.platform-ps { background: #f3e8ff; color: #7c3aed; }
.platform-xbox { background: #ecfdf5; color: #059669; }
.platform-switch { background: #fef2f2; color: #dc2626; }

/* Recommendations */
.recommendation-item {
    padding: 12px;
    margin: 8px 0;
    border-left: 4px solid var(--success-color);
    background: rgba(0, 212, 170, 0.05);
    border-radius: 0 8px 8px 0;
}

/* Loading animation */
.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid var(--success-color);
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 10px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Header
# =============================================================================
st.markdown("""
<div class="app-hero">
  <div class="logo">🎮</div>
  <div class="title">Games Recommendation System</div>
</div>
""", unsafe_allow_html=True)

# Status de l'API
health_col1, health_col2 = st.columns([3, 1])
with health_col2:
    if st.button("🔄 Vérifier API", type="secondary"):
        with st.spinner("Vérification..."):
            check_api_health()

with health_col1:
    if st.session_state.api_status == "healthy":
        st.markdown('<span class="status-badge status-healthy">🟢 API Active</span>', unsafe_allow_html=True)
    elif st.session_state.api_status == "error":
        st.markdown('<span class="status-badge status-error">🔴 API Indisponible</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-warning">🟡 API Non testée</span>', unsafe_allow_html=True)

st.markdown(f"**API Endpoint:** `{API_BASE_URL}`")

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### 👤 Authentification")
    
    if st.session_state.token:
        st.success(f"✅ Connecté: **{st.session_state.username}**")
        if st.button("🚪 Se déconnecter", type="secondary", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = None
            st.success("Déconnecté avec succès!")
            st.rerun()
    else:
        st.info("🔐 Non connecté")
    
    st.divider()
    
    # API Status détaillé
    st.markdown("### 📊 Status API")
    if st.button("🏥 Health Check", type="secondary", use_container_width=True):
        success, result = check_api_health()
        if success:
            st.json(result)
        else:
            st.error(result)
    
    st.divider()
    
    # Favoris
    st.markdown("### ⭐ Favoris")
    if st.session_state.favorites:
        for fav in sorted(st.session_state.favorites):
            st.write(f"• {fav}")
    else:
        st.caption("Aucun favori pour le moment")

# =============================================================================
# Main Content - Tabs
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🔐 Authentification", "🔍 Recherche", "✨ Recommandations", "📊 Monitoring"])

# =============================================================================
# Tab 1: Authentification
# =============================================================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Inscription")
        with st.form("register_form"):
            reg_username = st.text_input("Nom d'utilisateur", placeholder="votre-nom")
            reg_password = st.text_input("Mot de passe", type="password", placeholder="motdepasse123")
            
            if st.form_submit_button("S'inscrire", type="primary", use_container_width=True):
                if reg_username and reg_password:
                    with st.spinner("Inscription en cours..."):
                        success, result = api_request("POST", "/register", data={
                            "username": reg_username, 
                            "password": reg_password
                        })
                    
                    if success:
                        st.success("✅ Inscription réussie! Vous pouvez maintenant vous connecter.")
                    else:
                        st.error(f"❌ Inscription échouée: {result}")
                else:
                    st.warning("⚠️ Veuillez remplir tous les champs")
    
    with col2:
        st.subheader("🔑 Connexion")
        with st.form("login_form"):
            login_username = st.text_input("Nom d'utilisateur", 
                                         value=st.session_state.username or "", 
                                         placeholder="demo")
            login_password = st.text_input("Mot de passe", type="password", 
                                         placeholder="demo123")
            
            if st.form_submit_button("Se connecter", type="primary", use_container_width=True):
                if login_username and login_password:
                    with st.spinner("Connexion en cours..."):
                        success, result = api_request("POST", "/token", data={
                            "username": login_username,
                            "password": login_password
                        })
                    
                    if success:
                        st.session_state.token = result["access_token"]
                        st.session_state.username = login_username
                        st.success("✅ Connexion réussie!")
                        st.rerun()
                    else:
                        st.error(f"❌ Connexion échouée: {result}")
                else:
                    st.warning("⚠️ Veuillez remplir tous les champs")

# =============================================================================
# Tab 2: Recherche
# =============================================================================
with tab2:
    st.subheader("🔍 Recherche de jeux")
    
    # Suggestions rapides
    st.markdown("**Suggestions rapides:**")
    suggestions = ["Mario", "Zelda", "Elden Ring", "The Witcher", "Halo", "Fortnite"]
    
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.last_search = suggestion
                st.rerun()
    
    # Formulaire de recherche
    with st.form("search_form"):
        search_query = st.text_input(
            "🎯 Rechercher un jeu", 
            value=st.session_state.last_search,
            placeholder="Tapez le nom d'un jeu..."
        )
        
        search_col1, search_col2 = st.columns([3, 1])
        with search_col2:
            submitted = st.form_submit_button("🔍 Rechercher", type="primary", use_container_width=True)
        
        if submitted and search_query:
            if not st.session_state.token:
                st.error("🔐 Veuillez vous connecter d'abord!")
            else:
                with st.spinner("🔍 Recherche en cours..."):
                    success, result = api_request("GET", f"/games/by-title/{search_query}")
                
                if success:
                    st.session_state.last_search = search_query
                    st.session_state.last_results = result.get("results", [])
                    st.success(f"✅ {len(st.session_state.last_results)} résultat(s) trouvé(s)")
                else:
                    st.error(f"❌ Erreur de recherche: {result}")
    
    # Affichage des résultats
    if st.session_state.last_results:
        st.divider()
        st.markdown(f"### 📋 Résultats pour '{st.session_state.last_search}'")
        
        for game in st.session_state.last_results:
            st.markdown(f"""
            <div class="game-card">
                <h4>🎮 {game['title']}</h4>
                <div class="meta">
                    ⭐ Note: <strong>{game.get('rating', 'N/A')}</strong> | 
                    🧪 Metacritic: <strong>{game.get('metacritic', 'N/A')}</strong> |
                    🆔 ID: {game.get('id', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Boutons d'action
            game_col1, game_col2, game_col3 = st.columns([2, 1, 1])
            
            with game_col1:
                if game.get('best_price') and game.get('site_url'):
                    st.markdown(f"💰 **{game['best_price']}** chez {game.get('best_shop', 'Store')}")
                    st.markdown(f"🔗 [Acheter]({game['site_url']})")
            
            with game_col2:
                if st.button("⭐ Favori", key=f"fav_{game['id']}", use_container_width=True):
                    st.session_state.favorites.add(game['title'])
                    st.success("Ajouté aux favoris!")
            
            with game_col3:
                if st.button("🎯 Recos", key=f"reco_{game['id']}", use_container_width=True):
                    st.session_state.selected_game_for_reco = game['title']
                    st.info(f"Allez dans l'onglet Recommandations pour voir les suggestions basées sur '{game['title']}'")

# =============================================================================
# Tab 3: Recommandations
# =============================================================================
with tab3:
    st.subheader("✨ Recommandations ML")
    
    if not st.session_state.token:
        st.warning("🔐 Connectez-vous d'abord pour accéder au monitoring!")
    else:
        # Monitoring en temps réel
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🔄 Actualiser", type="secondary", use_container_width=True):
                st.rerun()
        
        with col1:
            st.markdown("**Surveillance en temps réel du modèle ML**")
        
        # Health Check détaillé
        st.markdown("### 🏥 Santé du système")
        
        success, health_data = api_request("GET", "/healthz")
        if success:
            # Status général
            status = health_data.get("status", "unknown")
            if status == "healthy":
                st.success("✅ Système en bonne santé")
            else:
                st.warning(f"⚠️ Statut: {status}")
            
            # Métriques clés
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🤖 Modèle",
                    "Chargé" if health_data.get("model_loaded") else "Non chargé",
                    delta=health_data.get("model_version", "N/A")
                )
            
            with col2:
                st.metric(
                    "🗄️ Base de données",
                    "Prête" if health_data.get("db_ready") else "Non prête",
                    delta=health_data.get("db_error", "OK") if health_data.get("db_error") else "OK"
                )
            
            with col3:
                st.metric(
                    "📊 Monitoring",
                    "Actif",
                    delta=f"Version {health_data.get('model_version', 'N/A')}"
                )
        else:
            st.error(f"❌ Impossible de récupérer les données de santé: {health_data}")
        
        # Métriques du modèle
        st.markdown("### 🤖 Métriques du modèle ML")
        
        success, model_metrics = api_request("GET", "/model/metrics")
        if success:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📈 Prédictions totales",
                    f"{model_metrics.get('total_predictions', 0):,}",
                    delta="prédictions"
                )
            
            with col2:
                avg_conf = model_metrics.get('avg_confidence', 0)
                st.metric(
                    "🎯 Confiance moyenne",
                    f"{avg_conf:.3f}",
                    delta=f"{avg_conf*100:.1f}%"
                )
            
            with col3:
                games_count = model_metrics.get('games_count', 0)
                st.metric(
                    "🎮 Jeux dans le modèle",
                    f"{games_count:,}",
                    delta="jeux"
                )
            
            with col4:
                feature_dim = model_metrics.get('feature_dimension', 0)
                st.metric(
                    "🧮 Dimensions",
                    f"{feature_dim}",
                    delta="features"
                )
            
            # Informations détaillées
            with st.expander("📋 Détails du modèle"):
                st.json(model_metrics)
                
        else:
            st.error(f"❌ Impossible de récupérer les métriques: {model_metrics}")
        
        # Summary du monitoring
        st.markdown("### 📊 Résumé du monitoring")
        
        success, monitoring_data = api_request("GET", "/monitoring/summary")
        if success:
            if "monitoring" in monitoring_data:
                monitoring = monitoring_data["monitoring"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Statistiques générales:**")
                    st.write(f"• Prédictions totales: **{monitoring.get('total_predictions', 0):,}**")
                    st.write(f"• Confiance moyenne: **{monitoring.get('avg_confidence', 0):.3f}**")
                    st.write(f"• Version du modèle: **{monitoring.get('model_version', 'N/A')}**")
                
                with col2:
                    st.markdown("**🕒 Historique:**")
                    hist_len = monitoring.get('confidence_hist_len', 0)
                    st.write(f"• Échantillons de confiance: **{hist_len}**")
                    last_training = monitoring.get('last_training')
                    if last_training:
                        st.write(f"• Dernier entraînement: **{last_training[:10]}**")
                    else:
                        st.write("• Dernier entraînement: **Jamais**")
            
            with st.expander("🔍 Données complètes de monitoring"):
                st.json(monitoring_data)
        else:
            st.error(f"❌ Erreur monitoring: {monitoring_data}")
        
        # Détection de dérive (drift)
        st.markdown("### 🌊 Détection de dérive du modèle")
        
        success, drift_data = api_request("GET", "/monitoring/drift")
        if success:
            drift_status = drift_data.get("status", "unknown")
            drift_score = drift_data.get("drift_score", 0)
            
            if drift_status == "stable":
                st.success(f"✅ Modèle stable (score: {drift_score:.3f})")
            elif drift_status == "moderate_drift":
                st.warning(f"⚠️ Dérive modérée détectée (score: {drift_score:.3f})")
            elif drift_status == "high_drift":
                st.error(f"🚨 Dérive élevée détectée (score: {drift_score:.3f})")
            else:
                st.info(f"ℹ️ {drift_data.get('message', 'Données insuffisantes')}")
            
            recommendation = drift_data.get("recommendation")
            if recommendation:
                st.info(f"💡 **Recommandation:** {recommendation}")
        else:
            st.warning(f"⚠️ Impossible de vérifier la dérive: {drift_data}")
        
        # Actions de maintenance
        st.markdown("### 🔧 Actions de maintenance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Réentraîner le modèle", type="primary", use_container_width=True):
                with st.spinner("🤖 Réentraînement en cours... Cela peut prendre quelques minutes."):
                    success, result = api_request("POST", "/model/train", json={"force_retrain": True})
                
                if success:
                    duration = result.get("duration", 0)
                    version = result.get("version", "N/A")
                    st.success(f"✅ Modèle réentraîné avec succès! Version {version} ({duration:.1f}s)")
                else:
                    st.error(f"❌ Échec du réentraînement: {result}")
        
        with col2:
            if st.button("🧪 Évaluer le modèle", type="secondary", use_container_width=True):
                test_queries = ["RPG", "Action", "Indie", "Strategy", "Simulation"]
                
                with st.spinner("🧪 Évaluation en cours..."):
                    success, eval_result = api_request("POST", "/model/evaluate", params={
                        "test_queries": test_queries
                    })
                
                if success:
                    st.success("✅ Évaluation terminée!")
                    
                    with st.expander("📊 Résultats de l'évaluation"):
                        st.json(eval_result)
                else:
                    st.error(f"❌ Échec de l'évaluation: {eval_result}")

# =============================================================================
# Footer
# =============================================================================
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🎮 Games Recommendation System**")
    st.caption("Propulsé par FastAPI + ML + Streamlit")

with footer_col2:
    st.markdown("**🔗 Liens utiles:**")
    st.markdown(f"• [API Documentation]({API_BASE_URL}/docs)")
    st.markdown(f"• [Health Check]({API_BASE_URL}/healthz)")

with footer_col3:
    st.markdown("**📊 Status:**")
    if st.session_state.api_status == "healthy":
        st.success("🟢 Système opérationnel")
    elif st.session_state.api_status == "error":
        st.error("🔴 Système indisponible")
    else:
        st.warning("🟡 Status inconnu")

# Auto-refresh pour le monitoring (optionnel)
if st.session_state.token:
    # Vérification périodique de la santé de l'API
    import asyncio
    
    # Placeholder pour un refresh automatique en arrière-plan
    # (non implémenté pour éviter les requêtes excessives)
    pass
vous d'abord pour accéder aux recommandations!")
    else:
        # Options de recommandations
        reco_type = st.radio(
            "Type de recommandation:",
            ["🎯 Par mot-clé/genre", "🎮 Par titre de jeu"],
            horizontal=True
        )
        
        if reco_type == "🎯 Par mot-clé/genre":
            with st.form("ml_reco_form"):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    query = st.text_input("🎯 Décrivez ce que vous cherchez", 
                                        placeholder="RPG fantasy, indie platformer, space simulation...")
                
                with col2:
                    k = st.number_input("Nombre de suggestions", min_value=1, max_value=20, value=5)
                
                with col3:
                    min_confidence = st.number_input("Confiance min", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
                
                if st.form_submit_button("🚀 Obtenir des recommandations", type="primary", use_container_width=True):
                    if query.strip():
                        with st.spinner("🤖 L'IA analyse vos préférences..."):
                            success, result = api_request("POST", "/recommend/ml", json={
                                "query": query,
                                "k": k,
                                "min_confidence": min_confidence
                            })
                        
                        if success:
                            recommendations = result.get("recommendations", [])
                            latency = result.get("latency_ms", 0)
                            model_version = result.get("model_version", "unknown")
                            
                            if recommendations:
                                st.success(f"✅ {len(recommendations)} recommandations générées en {latency:.1f}ms (modèle {model_version})")
                                
                                for i, rec in enumerate(recommendations, 1):
                                    confidence = rec.get("confidence", rec.get("score", 0))
                                    st.markdown(f"""
                                    <div class="recommendation-item">
                                        <strong>#{i}. {rec.get('title', 'Titre inconnu')}</strong><br>
                                        <small>
                                            🎯 Confiance: <strong>{confidence:.3f}</strong> | 
                                            🎪 Genres: {rec.get('genres', 'N/A')} | 
                                            ⭐ Note: {rec.get('rating', 'N/A')}
                                        </small>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("🤔 Aucune recommandation trouvée avec ces critères. Essayez d'abaisser le seuil de confiance.")
                        else:
                            st.error(f"❌ Erreur: {result}")
                    else:
                        st.warning("⚠️ Veuillez entrer une description ou un genre")
        
        else:  # Recommandations par titre
            with st.form("title_reco_form"):
                title = st.text_input("🎮 Nom du jeu", placeholder="The Witcher 3")
                k_title = st.number_input("Nombre de suggestions", min_value=1, max_value=15, value=5)
                
                if st.form_submit_button("🎯 Recommandations similaires", type="primary", use_container_width=True):
                    if title.strip():
                        with st.spinner("🔍 Recherche de jeux similaires..."):
                            success, result = api_request("GET", f"/recommend/by-title/{title}", params={"k": k_title})
                        
                        if success:
                            recommendations = result.get("recommendations", [])
                            source_id = result.get("source_id")
                            
                            if recommendations:
                                st.success(f"✅ {len(recommendations)} jeux similaires à '{title}'")
                                
                                for i, rec in enumerate(recommendations, 1):
                                    confidence = rec.get("confidence", rec.get("score", 0))
                                    st.markdown(f"""
                                    <div class="recommendation-item">
                                        <strong>#{i}. {rec.get('title', 'Titre inconnu')}</strong><br>
                                        <small>
                                            🎯 Similarité: <strong>{confidence:.3f}</strong> | 
                                            🎪 Genres: {rec.get('genres', 'N/A')} | 
                                            ⭐ Note: {rec.get('rating', 'N/A')}
                                        </small>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("🤔 Aucun jeu similaire trouvé.")
                        else:
                            st.error(f"❌ {result}")
                    else:
                        st.warning("⚠️ Veuillez entrer un nom de jeu")

# =============================================================================
# Tab 4: Monitoring
# =============================================================================
with tab4:
    st.subheader("📊 Monitoring du système")
    
    if not st.session_state.token:
        st.warning("🔐 Connectez-
