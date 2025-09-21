fix_indentation_error.py#!/usr/bin/env python3
"""
fix_indentation_error.py - Correction de l'erreur d'indentation ligne 812
"""

import re
from pathlib import Path

def fix_indentation_error():
    """Corrige l'erreur d'indentation à la ligne 812"""
    
    api_file = Path("api_games_plus.py")
    
    if not api_file.exists():
        print("❌ api_games_plus.py non trouvé")
        return False
    
    print("🔧 Correction de l'erreur d'indentation ligne 812...")
    
    # Lire le fichier
    with open(api_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Fichier lu: {len(lines)} lignes")
    
    # Chercher autour de la ligne 812
    error_line = 811  # Index 0-based
    
    if error_line < len(lines):
        problematic_line = lines[error_line]
        print(f"🔍 Ligne 812: '{problematic_line.rstrip()}'")
        
        # Vérifier les lignes autour pour comprendre le contexte
        context_start = max(0, error_line - 5)
        context_end = min(len(lines), error_line + 5)
        
        print("📋 Contexte:")
        for i in range(context_start, context_end):
            line_num = i + 1
            line = lines[i].rstrip()
            marker = "🔴" if i == error_line else "  "
            print(f"{marker} {line_num:3d}: '{line}'")
        
        # Correction automatique
        corrections_made = 0
        
        # Cas 1: Ligne avec indentation incorrecte après return
        if error_line > 0:
            prev_line = lines[error_line - 1].rstrip()
            current_line = lines[error_line]
            
            # Si la ligne précédente est un return et la ligne actuelle est indentée
            if ("return" in prev_line or prev_line.strip().endswith("}") or prev_line.strip().endswith(")")):
                # La ligne suivante ne devrait pas être indentée plus que nécessaire
                if current_line.startswith("        ") and not current_line.strip().startswith("except") and not current_line.strip().startswith("finally"):
                    # Réduire l'indentation
                    lines[error_line] = current_line[4:]  # Enlever 4 espaces
                    corrections_made += 1
                    print("✅ Correction: Réduit l'indentation excessive")
        
        # Cas 2: Ligne complètement mal indentée
        if corrections_made == 0:
            current_line = lines[error_line]
            # Si la ligne a une indentation bizarre, la corriger
            if current_line.startswith("    ") and not current_line.startswith("        "):
                # Probablement une fonction ou une classe
                if any(keyword in current_line for keyword in ["def ", "class ", "@", "if ", "for ", "while ", "try:", "except", "finally"]):
                    lines[error_line] = current_line  # Garder tel quel
                else:
                    # Probablement du code dans une fonction
                    lines[error_line] = "    " + current_line.lstrip()
                    corrections_made += 1
                    print("✅ Correction: Ajusté l'indentation")
        
        # Cas 3: Ligne avec caractères d'indentation mixtes
        if corrections_made == 0:
            current_line = lines[error_line]
            if "\t" in current_line:
                # Remplacer les tabs par des espaces
                lines[error_line] = current_line.expandtabs(4)
                corrections_made += 1
                print("✅ Correction: Remplacé les tabs par des espaces")
        
        # Cas 4: Ligne vide mal formatée
        if corrections_made == 0:
            current_line = lines[error_line]
            if current_line.strip() == "" and len(current_line) > 1:
                lines[error_line] = "\n"
                corrections_made += 1
                print("✅ Correction: Nettoyé ligne vide")
        
        if corrections_made > 0:
            # Sauvegarder le fichier corrigé
            with open(api_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print(f"✅ {corrections_made} correction(s) appliquée(s)")
            return True
        else:
            print("⚠️ Aucune correction automatique trouvée")
            return False
    
    print("❌ Ligne 812 introuvable")
    return False

def validate_python_syntax():
    """Valide la syntaxe Python après correction"""
    print("🧪 Validation de la syntaxe Python...")
    
    try:
        import ast
        
        with open("api_games_plus.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parser le code pour vérifier la syntaxe
        ast.parse(content)
        print("✅ Syntaxe Python valide")
        return True
        
    except SyntaxError as e:
        print(f"❌ Erreur de syntaxe persistante:")
        print(f"   Ligne {e.lineno}: {e.text}")
        print(f"   Erreur: {e.msg}")
        return False
    except IndentationError as e:
        print(f"❌ Erreur d'indentation persistante:")
        print(f"   Ligne {e.lineno}: {e.text}")
        print(f"   Erreur: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Erreur de validation: {e}")
        return False

def create_clean_section():
    """Crée une section propre pour remplacer la partie problématique"""
    
    clean_section = '''
# Endpoints interactions utilisateur - SECTION PROPRE
@app.post("/interactions", tags=["user-data"], status_code=201)
def add_user_interaction(
    game_id_rawg: int,
    interaction_type: str,
    rating: Optional[float] = None,
    username: str = Depends(verify_token)
):
    """Ajoute une interaction utilisateur"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                # Vérifier si le jeu existe
                cur.execute("SELECT game_id_rawg FROM games WHERE game_id_rawg = %s", (game_id_rawg,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail="Game not found")
                
                # Ajouter ou mettre à jour l'interaction
                cur.execute("""
                    INSERT INTO user_interactions (user_id, game_id_rawg, interaction_type, rating, timestamp)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                    interaction_type = VALUES(interaction_type),
                    rating = VALUES(rating),
                    timestamp = NOW()
                """, (user_id, game_id_rawg, interaction_type, rating))
        
        return {
            "message": "Interaction added successfully",
            "user_id": user_id,
            "game_id_rawg": game_id_rawg,
            "interaction_type": interaction_type,
            "rating": rating
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding user interaction: {e}")
        raise HTTPException(status_code=500, detail="Failed to add interaction")

@app.get("/interactions", tags=["user-data"])
def get_user_interactions(username: str = Depends(verify_token)):
    """Récupère les interactions de l'utilisateur"""
    
    try:
        user_id = get_user_id_from_username(username)
        
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ui.*, g.title, g.rating, g.genres
                    FROM user_interactions ui
                    JOIN games g ON ui.game_id_rawg = g.game_id_rawg
                    WHERE ui.user_id = %s
                    ORDER BY ui.timestamp DESC
                """, (user_id,))
                
                interactions = cur.fetchall()
        
        return {
            "user_id": user_id,
            "interactions": [
                {
                    "game_id_rawg": interaction["game_id_rawg"],
                    "title": interaction["title"],
                    "interaction_type": interaction["interaction_type"],
                    "rating": interaction.get("rating"),
                    "timestamp": interaction["timestamp"].isoformat() if interaction.get("timestamp") else None,
                    "game_rating": interaction.get("rating"),
                    "genres": interaction.get("genres")
                }
                for interaction in interactions
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching user interactions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch interactions")
'''
    
    with open("clean_interactions_section.py", "w") as f:
        f.write(clean_section)
    
    print("✅ Section propre créée: clean_interactions_section.py")

def main():
    """Fonction principale de correction"""
    
    print("🚨 CORRECTION ERREUR INDENTATION LIGNE 812")
    print("=" * 45)
    
    # Tentative de correction automatique
    if fix_indentation_error():
        if validate_python_syntax():
            print("\n🎉 CORRECTION RÉUSSIE!")
            print("✅ Erreur d'indentation résolue")
            return True
        else:
            print("\n⚠️ Erreur persistante après correction")
    
    # Solution de secours
    print("\n🆘 Solution de secours:")
    create_clean_section()
    
    print("\n📋 Actions manuelles:")
    print("1. Ouvrez api_games_plus.py ligne 812")
    print("2. Vérifiez l'indentation (4 espaces par niveau)")
    print("3. Utilisez clean_interactions_section.py si nécessaire")
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
