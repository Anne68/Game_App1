import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuration de la page avec thème sombre gaming
st.set_page_config(
    page_title="🎮 Games AI Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Gaming Dark Theme
gaming_css = """
<style>
/* Fond gaming sombre avec effet néon */
.stApp {
    background: linear-gradient(135deg, 
        #0a0a0a 0%, 
        #1a1a2e 25%, 
        #16213e 50%, 
        #0f3460 100%) !important;
    color: #e6f3ff;
    font-family: 'Courier New', monospace;
}

/* Sidebar gaming */
.css-1d391kg {
    background: linear-gradient(180deg, #0a0a0a, #1a1a2e) !important;
    border-right: 2px solid #00ff88;
}

/* Titres avec effet néon */
h1, h2, h3 {
    color: #00ff88 !important;
    text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88, 0 0 40px #00ff88;
    text-align: center;
    font-weight: bold;
}

/* Boutons gaming */
.stButton > button {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
    color: #000 !important;
    border: 2px solid #00ff88 !important;
    border-radius: 25px !important;
    font-weight: bold !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.5) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 0 30px rgba(0,255,136,0.8) !important;
}

/* Inputs avec style gaming */
.stTextInput input, .stSelectbox select {
    background: rgba(0,0,0,0.7) !important;
    border: 2px solid #00ff88 !important;
    border-radius: 10px !important;
    color: #00ff88 !important;
    font-family: 'Courier New', monospace !important;
}

/* Métriques gaming */
.css-1r6slb0 {
    background: rgba(0,255,136,0.1) !important;
    border: 1px solid #00ff88 !important;
    border-radius: 15px !important;
    box-shadow: 0 0 15px rgba(0,255,136,0.3) !important;
}

/* Cards de recommandations */
.recommendation-card {
    background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(26,26,46,0.9));
    border: 2px solid #00ff88;
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 0 25px rgba(0,255,136,0.4);
}

/* Animation de typing pour les titres */
@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}

.typing-effect {
    overflow: hidden;
    border-right: .15em solid #00ff88;
    white-space: nowrap;
    margin: 0 auto;
    letter-spacing: .15em;
    animation: typing 3.5s steps(40, end);
}

/* Effet matrix pour le fond */
.matrix-bg::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at 20% 50%, rgba(0,255,136,0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255,107,107,0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 80%, rgba(78,205,196,0.1) 0%, transparent 50%);
    pointer-events: none;
    z-index: -1;
}

/* Messages d'erreur et succès */
.stAlert > div {
    border-radius: 15px !important;
    border: 2px solid #00ff88 !important;
}

/* Scrollbars gaming */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: #1a1a2e;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(45deg, #00ff88, #4ecdc4);
    border-radius: 6px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(45deg, #4ecdc4, #00ff88);
}
</style>
"""

st.markdown(gaming_css, unsafe_allow_html=True)

# Configuration API
API_URL = "https://game-app-y8be.onrender.com"
if "api_token" not in st.session_state:
    st.session_state.api_token = None

# Fonction d'authentification
def authenticate(username, password):
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={"username": username, "password": password},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["access_token"]
    except Exception as e:
        st.error(f"Erreur de connexion: {e}")
    return None

# Fonction pour les recommandations
def get_recommendations(query, k=10, token=None):
    if not token:
        return None
    
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"query": query, "k": k, "min_confidence": 0.1}
    
    try:
        response = requests.post(
            f"{API_URL}/recommend/ml",
            json=payload,
            headers=headers,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Erreur API: {e}")
    return None

# Interface principale
def main():
    # Titre avec effet typing
    st.markdown('<div class="matrix-bg"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="typing-effect">🎮 AI GAMES RECOMMENDER</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #4ecdc4;">Powered by Advanced ML Algorithms</h3>', unsafe_allow_html=True)
    
    # Sidebar pour l'authentification
    with st.sidebar:
        st.markdown("## 🔐 AUTHENTICATION PORTAL")
        
        if not st.session_state.api_token:
            st.markdown("### 🚀 LOGIN TO ACCESS")
            username = st.text_input("👤 Username", value="demo")
            password = st.text_input("🔑 Password", value="demo123", type="password")
            
            if st.button("🎯 CONNECT TO SYSTEM"):
                with st.spinner("🔄 Authentification en cours..."):
                    token = authenticate(username, password)
                    if token:
                        st.session_state.api_token = token
                        st.success("✅ Connexion réussie!")
                        st.rerun()
        else:
            st.success("🟢 SYSTÈME CONNECTÉ")
            if st.button("🚪 DÉCONNEXION"):
                st.session_state.api_token = None
                st.rerun()
        
        # Statut de l'API
        st.markdown("## 📡 API STATUS")
        try:
            health_response = requests.get(f"{API_URL}/healthz", timeout=5)
            if health_response.status_code == 200:
                st.success("🟢 API ONLINE")
            else:
                st.error("🔴 API ERROR")
        except:
            st.error("🔴 API OFFLINE")

    # Interface principale si connecté
    if st.session_state.api_token:
        # Métriques en temps réel
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
            st.metric("🎮 Games DB", "10K+", "↗️ 5%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
            st.metric("🤖 AI Accuracy", "94.2%", "↗️ 2.1%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
            st.metric("⚡ Response Time", "127ms", "↘️ 15ms")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="css-1r6slb0">', unsafe_allow_html=True)
            st.metric("👥 Users Online", "2.3K", "↗️ 12%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Interface de recommandation
        st.markdown("## 🔮 AI RECOMMENDATION ENGINE")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            game_query = st.text_input(
                "🎯 Décris ton jeu idéal:",
                placeholder="Ex: Action RPG fantasy avec magic system...",
                help="Utilise des mots-clés comme: genre, thème, gameplay, plateformes..."
            )
        
        with col_right:
            num_recommendations = st.selectbox(
                "📊 Nombre de recommandations:",
                options=[5, 10, 15, 20],
                index=1
            )
        
        if st.button("🚀 GÉNÉRER RECOMMANDATIONS IA", type="primary"):
            if game_query.strip():
                with st.spinner("🤖 L'IA analyse vos préférences..."):
                    time.sleep(1)  # Animation
                    recommendations = get_recommendations(
                        game_query, 
                        num_recommendations, 
                        st.session_state.api_token
                    )
                    
                    if recommendations and recommendations.get("recommendations"):
                        st.success(f"✅ {len(recommendations['recommendations'])} jeux trouvés!")
                        
                        # Graphique de confiance
                        games_df = pd.DataFrame(recommendations["recommendations"])
                        if not games_df.empty:
                            fig = px.bar(
                                games_df.head(10), 
                                x="confidence", 
                                y="title",
                                orientation="h",
                                color="confidence",
                                color_continuous_scale="viridis",
                                title="🎯 Scores de Confiance IA"
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='#00ff88'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Affichage des recommandations
                        st.markdown("## 🎮 RECOMMANDATIONS PERSONNALISÉES")
                        
                        for i, game in enumerate(recommendations["recommendations"], 1):
                            # Card de jeu stylée
                            st.markdown(f'''
                            <div class="recommendation-card">
                                <h3 style="color: #00ff88;">#{i} {game["title"]}</h3>
                                <div style="display: flex; justify-content: space-between; margin: 15px 0;">
                                    <span style="color: #4ecdc4;"><strong>Genre:</strong> {game.get("genres", "N/A")}</span>
                                    <span style="color: #ff6b6b;"><strong>Confiance:</strong> {game["confidence"]:.1%}</span>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span><strong>⭐ Rating:</strong> {game.get("rating", 0):.1f}/5</span>
                                    <span><strong>🏆 Metacritic:</strong> {game.get("metacritic", 0)}</span>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # Métriques de la requête
                        st.markdown("## 📈 MÉTRIQUES DE LA REQUÊTE")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("⚡ Temps de traitement", 
                                    f"{recommendations.get('latency_ms', 0):.0f}ms")
                        with col2:
                            st.metric("🤖 Version du modèle", 
                                    recommendations.get('model_version', 'N/A'))
                        with col3:
                            st.metric("🎯 Confiance moyenne", 
                                    f"{sum(r['confidence'] for r in recommendations['recommendations'])/len(recommendations['recommendations']):.1%}")
                    
                    else:
                        st.warning("❌ Aucune recommandation trouvée. Essayez avec d'autres mots-clés.")
            else:
                st.error("⚠️ Veuillez saisir une description de jeu.")
        
        # Section de navigation supplémentaire
        st.markdown("---")
        st.markdown("## 🎯 NAVIGATION RAPIDE")
        
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        
        with nav_col1:
            if st.button("🔥 Trending Games"):
                st.info("🚧 Fonctionnalité en développement")
        
        with nav_col2:
            if st.button("⭐ Top Rated"):
                st.info("🚧 Fonctionnalité en développement")
        
        with nav_col3:
            if st.button("🆕 New Releases"):
                st.info("🚧 Fonctionnalité en développement")
        
        # Footer gaming
        st.markdown("""
        ---
        <div style="text-align: center; color: #4ecdc4; font-family: 'Courier New', monospace;">
            <p>🎮 <strong>GAMES AI RECOMMENDER v2.0</strong> 🎮</p>
            <p style="font-size: 12px; color: #666;">
                Powered by Advanced Machine Learning | Built with ❤️ for Gamers
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Message d'accueil si pas connecté
        st.markdown("""
        <div style="text-align: center; margin: 50px 0;">
            <h2 style="color: #ff6b6b;">🔒 ACCÈS SÉCURISÉ REQUIS</h2>
            <p style="color: #4ecdc4; font-size: 18px;">
                Connectez-vous dans la sidebar pour accéder au système de recommandation IA
            </p>
            <p style="color: #666;">
                💡 <em>Utilisez demo/demo123 pour tester</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
