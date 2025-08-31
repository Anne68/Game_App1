#!/usr/bin/env python3
"""
Correction ciblée de l'erreur logger spécifique trouvée dans les logs Render
"""

def fix_specific_error():
    """Corrige l'erreur spécifique dans api_games_plus.py"""
    
    try:
        # Lire le fichier
        with open('api_games_plus.py', 'r') as f:
            content = f.read()
        
        # Corrections spécifiques basées sur l'erreur des logs
        fixes = [
            # Erreur principale trouvée dans les logs
            ('logger.info("API startup complete.")()', 'logger.info("API startup complete.")'),
            
            # Autres patterns similaires potentiels
            ('logger.warning("E4 Compliance standards not available")()', 'logger.warning("E4 Compliance standards not available")'),
            ('logger.info("E4 Compliance standards active")()', 'logger.info("E4 Compliance standards active")'),
            ('logger.info("Starting Games API with ML and E4 compliance...")()', 'logger.info("Starting Games API with ML and E4 compliance...")'),
        ]
        
        changes_made = 0
        for wrong, correct in fixes:
            if wrong in content:
                content = content.replace(wrong, correct)
                changes_made += 1
                print(f"✅ Fixed: {wrong}")
        
        # Sauvegarder si des changements ont été faits
        if changes_made > 0:
            with open('api_games_plus.py', 'w') as f:
                f.write(content)
            print(f"✅ {changes_made} correction(s) appliquée(s)")
            return True
        else:
            print("No specific errors found to fix")
            return False
            
    except Exception as e:
        print(f"Error fixing file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 CORRECTION ERREUR LOGGER RENDER")
    print("Fixing: TypeError: 'NoneType' object is not callable")
    print()
    
    if fix_specific_error():
        print("\n✅ CORRECTION TERMINÉE!")
        print("Maintenant committez et poussez:")
        print("git add api_games_plus.py")
        print("git commit -m 'fix: Remove extra parentheses causing TypeError'")
        print("git push")
    else:
        print("\n⚠️ Vérifiez manuellement le fichier api_games_plus.py")
        print("Cherchez les lignes avec logger.info() suivies de ()")
