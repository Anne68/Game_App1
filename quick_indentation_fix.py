#!/usr/bin/env python3
"""
quick_indentation_fix.py - Correction rapide de l'indentation
"""

def fix_specific_line_812():
    """Corrige spécifiquement la ligne 812"""
    
    with open("api_games_plus.py", "r") as f:
        lines = f.readlines()
    
    if len(lines) > 811:
        # Ligne 812 (index 811)
        line_812 = lines[811]
        
        print(f"Ligne 812 actuelle: '{repr(line_812)}'")
        
        # Corrections communes pour l'indentation
        fixed_line = line_812
        
        # Supprimer l'indentation excessive
        if line_812.startswith("            "):  # 12 espaces
            fixed_line = line_812[4:]  # Réduire à 8 espaces
        elif line_812.startswith("        "):     # 8 espaces inappropriés
            # Vérifier le contexte
            if len(lines) > 810:
                prev_line = lines[810].strip()
                if prev_line.endswith("}") or prev_line.endswith(")") or "return" in prev_line:
                    # Cette ligne devrait être au niveau fonction (4 espaces)
                    fixed_line = "    " + line_812.lstrip()
        
        # Remplacer les tabs par des espaces
        fixed_line = fixed_line.expandtabs(4)
        
        # Si la ligne contient juste des espaces, la vider
        if fixed_line.strip() == "":
            fixed_line = "\n"
        
        if fixed_line != line_812:
            lines[811] = fixed_line
            
            with open("api_games_plus.py", "w") as f:
                f.writelines(lines)
            
            print(f"Ligne 812 corrigée: '{repr(fixed_line)}'")
            return True
    
    return False

def validate_syntax():
    """Test rapide de syntaxe"""
    try:
        import ast
        with open("api_games_plus.py", "r") as f:
            content = f.read()
        ast.parse(content)
        print("✅ Syntaxe valide")
        return True
    except (SyntaxError, IndentationError) as e:
        print(f"❌ Erreur ligne {e.lineno}: {e.msg}")
        return False

if __name__ == "__main__":
    print("🔧 CORRECTION RAPIDE LIGNE 812")
    
    if fix_specific_line_812():
        if validate_syntax():
            print("🎉 CORRIGÉ!")
        else:
            print("⚠️ Erreur persistante")
    else:
        print("⚠️ Aucune correction appliquée")
