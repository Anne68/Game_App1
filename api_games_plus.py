# =====================================================================
# ENDPOINTS RECOMMANDATIONS SIMPLIFIÉS
# =====================================================================

@app.post("/recommend/similar-game", tags=["recommendations"])
@measure_latency("recommend_similar_game")
def recommend_similar_game(
    game_id: int = Query(..., description="ID du jeu de référence"),
    k: int = Query(10, ge=1, le=50, description="Nombre de recommandations"),
    user: str = Depends(verify_token)
):
    """Recommandations basées sur la similarité avec un jeu spécifique (KNN)"""
    model = get_model()
    
    if not model.is_trained:
        # Auto-entraînement si nécessaire
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        recommendations = model.predict_similar_by_game(game_id, k=k)
        
        if not recommendations:
            return {
                "game_id": game_id,
                "message": f"Aucun jeu similaire trouvé pour l'ID {game_id}",
                "recommendations": []
            }
        
        return {
            "game_id": game_id,
            "method": "knn_similarity",
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.get("/recommend/cluster/{cluster_id}", tags=["recommendations"])
@measure_latency("recommend_cluster")
def recommend_cluster(
    cluster_id: int = Path(..., ge=0, le=20, description="ID du cluster"),
    k: int = Query(10, ge=1, le=50, description="Nombre de jeux du cluster"),
    user: str = Depends(verify_token)
):
    """Recommandations des meilleurs jeux d'un cluster spécifique"""
    model = get_model()
    
    if not model.is_trained:
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        cluster_games = model.get_cluster_games(cluster_id, k=k)
        
        if not cluster_games:
            return {
                "cluster_id": cluster_id,
                "message": f"Aucun jeu trouvé dans le cluster {cluster_id}",
                "games": []
            }
        
        return {
            "cluster_id": cluster_id,
            "method": "cluster_based",
            "games": cluster_games,
            "count": len(cluster_games)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/cluster-explore", tags=["recommendations"])
@measure_latency("recommend_cluster_explore")
def recommend_cluster_explore(user: str = Depends(verify_token)):
    """Exploration des clusters disponibles avec leurs caractéristiques"""
    model = get_model()
    
    if not model.is_trained:
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        cluster_info = model.get_cluster_info()
        
        return {
            "clusters_count": len(cluster_info),
            "clusters": cluster_info,
            "method": "cluster_analysis"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/random-cluster", tags=["recommendations"])
@measure_latency("recommend_random_cluster")
def recommend_random_cluster(
    k: int = Query(10, ge=1, le=30, description="Nombre de jeux aléatoires"),
    user: str = Depends(verify_token)
):
    """Recommandations aléatoires d'un cluster aléatoire pour la découverte"""
    model = get_model()
    
    if not model.is_trained:
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        random_games = model.get_random_cluster_games(k=k)
        
        return {
            "method": "random_cluster_discovery",
            "recommendations": random_games,
            "count": len(random_games)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/by-title", tags=["recommendations"])
@measure_latency("recommend_by_title")
def recommend_by_title(
    title: str = Query(..., min_length=2, description="Titre ou partie de titre à rechercher"),
    k: int = Query(10, ge=1, le=30, description="Nombre de recommandations"),
    user: str = Depends(verify_token)
):
    """Recommandations basées sur la similarité de titre"""
    model = get_model()
    
    if not model.is_trained:
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        recommendations = model.recommend_by_title_similarity(title, k=k)
        
        if not recommendations:
            return {
                "title_query": title,
                "message": f"Aucun jeu trouvé avec un titre similaire à '{title}'",
                "recommendations": []
            }
        
        return {
            "title_query": title,
            "method": "title_similarity",
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/by-genre", tags=["recommendations"])
@measure_latency("recommend_by_genre")
def recommend_by_genre(
    genre: str = Query(..., min_length=3, description="Genre de jeu (Action, RPG, Strategy, etc.)"),
    k: int = Query(10, ge=1, le=30, description="Nombre de recommandations"),
    user: str = Depends(verify_token)
):
    """Recommandations par genre, triées par rating"""
    model = get_model()
    
    if not model.is_trained:
        games = fetch_games_for_ml()
        model.train(games)
    
    try:
        recommendations = model.recommend_by_genre(genre, k=k)
        
        if not recommendations:
            return {
                "genre_query": genre,
                "message": f"Aucun jeu trouvé pour le genre '{genre}'",
                "recommendations": [],
                "available_genres": [
                    "Action", "RPG", "Strategy", "Indie", "Adventure", 
                    "Simulation", "Sports", "Racing", "Puzzle", "Platformer"
                ]
            }
        
        return {
            "genre_query": genre,
            "method": "genre_filter",
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# ENDPOINTS AUTH (identiques à l'original)
# =====================================================================

@app.post("/token", tags=["auth"])
def token(username: str = Form(...), password: str = Form(...)):
