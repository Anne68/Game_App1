# Nouveaux endpoints à ajouter dans api_games_plus.py

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

    # Corrections à apporter aux endpoints existants dans api_games_plus.py
# À insérer après les endpoints de recommandation existants (ligne ~350)

# ---------------------------------------------------------------------
# Nouveaux endpoints K-Means (CORRIGES)
# ---------------------------------------------------------------------

@app.get("/recommend/similar-game/{game_id}", tags=["recommend"])
@measure_latency("recommend_similar_game")
def recommend_similar_game(game_id: int, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    """KNN sur un jeu spécifique"""
    _ensure_model_trained_with_db()
    model = get_model()
    
    try:
        recommendations = model.predict_similar_game(game_id, k)
        
        response = {
            "game_id": game_id,
            "recommendations": recommendations,
            "algorithm": "KNN",
            "model_version": model.model_version
        }
        
        # Enhancement accessibilité si disponible
        accessibility_validator = get_accessibility_validator()
        if accessibility_validator:
            try:
                response = accessibility_validator.enhance_response_accessibility(response)
            except Exception as e:
                logger.warning(f"Accessibility enhancement failed: {e}")
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Similar game prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/recommend/cluster/{cluster_id}", tags=["recommend"])
@measure_latency("recommend_cluster")
def recommend_cluster_games(cluster_id: int, k: int = Query(20, ge=1, le=50), user: str = Depends(verify_token)):
    """Jeux d'un cluster spécifique"""
    _ensure_model_trained_with_db()
    model = get_model()
    
    try:
        games = model.get_cluster_games(cluster_id, k)
        cluster_analysis = model.explore_clusters()
        
        response = {
            "cluster_id": cluster_id,
            "games": games,
            "cluster_info": cluster_analysis.get(f"cluster_{cluster_id}", {}),
            "total_games": len(games),
            "model_version": model.model_version
        }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Cluster recommendation failed")
        raise HTTPException(status_code=500, detail="Cluster recommendation failed")

@app.get("/recommend/cluster-explore", tags=["recommend"])
@measure_latency("recommend_cluster_explore")
def explore_clusters(user: str = Depends(verify_token)):
    """Analyse complète des clusters"""
    _ensure_model_trained_with_db()
    model = get_model()
    
    try:
        cluster_analysis = model.explore_clusters()
        
        response = {
            "clusters": cluster_analysis,
            "total_clusters": len(cluster_analysis),
            "algorithm": "K-Means",
            "model_version": model.model_version,
            "description": "Analyse des 8 clusters automatiques créés par K-Means"
        }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Cluster exploration failed")
        raise HTTPException(status_code=500, detail="Cluster exploration failed")

@app.get("/recommend/random-cluster", tags=["recommend"])
@measure_latency("recommend_random_cluster")
def discover_random_cluster(k: int = Query(10, ge=1, le=30), user: str = Depends(verify_token)):
    """Découverte aléatoire d'un cluster"""
    _ensure_model_trained_with_db()
    model = get_model()
    
    try:
        result = model.get_random_cluster_games(k)
        
        response = {
            **result,
            "algorithm": "K-Means + Random",
            "model_version": model.model_version,
            "description": f"Découverte aléatoire du cluster {result['cluster_id']}"
        }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Random cluster discovery failed")
        raise HTTPException(status_code=500, detail="Random cluster discovery failed")

@app.get("/recommend/by-title-similarity/{title}", tags=["recommend"])
@measure_latency("recommend_by_title_similarity")
def recommend_by_title_similarity(title: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    """Similarité de titres avec KNN"""
    _ensure_model_trained_with_db()
    model = get_model()
    
    # Nettoyage du titre avec compliance si disponible
    clean_title = title.strip()
    security_validator = get_security_validator()
    if security_validator:
        try:
            clean_title = security_validator.sanitize_input(clean_title)
        except Exception as e:
            logger.warning(f"Title sanitization failed: {e}")
    
    if not clean_title:
        raise HTTPException(status_code=400, detail="Title is empty")
    
    try:
        recommendations = model.predict_by_title_similarity(clean_title, k)
        
        if not recommendations:
            # Fallback sur recherche classique
            matches = find_games_by_title(clean_title, limit=k)
            recommendations = []
            for match in matches:
                recommendations.append({
                    "title": match["title"],
                    "genres": "",
                    "rating": match.get("rating", 0.0),
                    "metacritic": match.get("metacritic", 0),
                    "cluster": -1  # Pas de cluster pour fallback
                })
        
        response = {
            "query_title": clean_title,
            "recommendations": recommendations,
            "algorithm": "Title Similarity + KNN",
            "model_version": model.model_version,
            "found_matches": len(recommendations) > 0
        }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Title similarity recommendation failed")
        raise HTTPException(status_code=500, detail="Title similarity recommendation failed")

@app.get("/recommend/by-genre-filter/{genre}", tags=["recommend"])
@measure_latency("recommend_by_genre_filter")
def recommend_by_genre_filter(genre: str, k: int = Query(10, ge=1, le=50), user: str = Depends(verify_token)):
    """Filtrage par genre avec classification"""
    _ensure_model_trained_with_db()
    model = get_model()
    
    # Nettoyage du genre avec compliance si disponible
    clean_genre = genre.strip()
    security_validator = get_security_validator()
    if security_validator:
        try:
            clean_genre = security_validator.sanitize_input(clean_genre)
        except Exception as e:
            logger.warning(f"Genre sanitization failed: {e}")
    
    if not clean_genre:
        raise HTTPException(status_code=400, detail="Genre is empty")
    
    try:
        recommendations = model.predict_by_genre(clean_genre, k)
        
        response = {
            "genre": clean_genre,
            "recommendations": recommendations,
            "algorithm": "Genre Filter + Rating Sort",
            "model_version": model.model_version,
            "found_matches": len(recommendations)
        }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Genre filter recommendation failed")
        raise HTTPException(status_code=500, detail="Genre filter recommendation failed")

