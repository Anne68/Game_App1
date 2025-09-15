#!/usr/bin/env python3
"""
Correction de l'erreur de syntaxe logger dans api_games_plus.py
TypeError: 'NoneType' object is not callable
"""

import re
from pathlib import Path

def fix_logger_syntax_error():
    """Corrige l'erreur de syntaxe dans le logger"""
    print("🔧 Correction de l'erreur logger...")
    
    api_file = Path("api_games_plus.py")
    
    if not api_file.exists():
        print("❌ api_games_plus.py non trouvé")
        return False
    
    # Lire le contenu
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Chercher et corriger l'erreur spécifique
    error_patterns = [
        r'logger\.info\("API startup complete\."\)\(\)',  # Erreur identifiée
        r'logger\.info\([^)]+\)\(\)',  # Pattern général pour ce type d'erreur
        r'logger\.(info|warning|error|debug)\([^)]+\)\(\)',  # Plus général
    ]
    
    corrections_made = 0
    
    for pattern in error_patterns:
        matches = re.findall(pattern, content)
        if matches:
            print(f"   🔍 Trouvé pattern problématique: {pattern}")
            # Corriger en supprimant les () supplémentaires
            corrected_content = re.sub(pattern, lambda m: m.group(0)[:-2], content)
            if corrected_content != content:
                content = corrected_content
                corrections_made += 1
                print(f"   ✅ Corrigé")
    
    # Corrections spécifiques basées sur l'erreur
    specific_fixes = [
        # Correction de l'erreur exacte trouvée dans les logs
        ('logger.info("API startup complete.")()', 'logger.info("API startup complete.")'),
        # Autres corrections potentielles
        ('logger.info("Starting Games API with ML and E4 compliance...")()', 'logger.info("Starting Games API with ML and E4 compliance...")'),
        ('logger.info("E4 Compliance standards active")()', 'logger.info("E4 Compliance standards active")'),
    ]
    
    for wrong, correct in specific_fixes:
        if wrong in content:
            content = content.replace(wrong, correct)
            corrections_made += 1
            print(f"   ✅ Corrigé: {wrong} → {correct}")
    
    if corrections_made > 0:
        # Sauvegarder le fichier corrigé
        with open(api_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {corrections_made} correction(s) appliquée(s) dans api_games_plus.py")
        return True
    else:
        print("⚠️ Aucune correction nécessaire trouvée")
        return False

def validate_syntax():
    """Valide la syntaxe Python après correction"""
    print("🧪 Validation syntaxe Python...")
    
    try:
        import ast
        
        with open("api_games_plus.py", 'r') as f:
            content = f.read()
        
        # Parser le code pour vérifier la syntaxe
        ast.parse(content)
        print("✅ Syntaxe Python valide")
        return True
        
    except SyntaxError as e:
        print(f"❌ Erreur de syntaxe: {e}")
        print(f"   Ligne {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Erreur validation: {e}")
        return False

def test_import():
    """Test l'import du module après correction"""
    print("🧪 Test import module...")
    
    import sys
    import os
    
    # Variables d'environnement pour le test
    os.environ.update({
        'SECRET_KEY': 'test-key-very-long-for-testing',
        'DB_REQUIRED': 'false',
        'DEMO_LOGIN_ENABLED': 'true',
        'LOG_LEVEL': 'INFO'
    })
    
    try:
        # Recharger le module si déjà importé
        if 'api_games_plus' in sys.modules:
            del sys.modules['api_games_plus']
        
        import api_games_plus
        print("✅ Import réussi")
        
        # Test basique de l'app
        app = api_games_plus.app
        print(f"✅ App créée: {app.title}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur import: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de correction"""
    print("🚨 CORRECTION ERREUR LOGGER RENDER")
    print("=" * 35)
    
    print("🔍 Erreur identifiée dans les logs:")
    print("   TypeError: 'NoneType' object is not callable")
    print("   logger.info('API startup complete.')()")
    print("   → Parenthèses supplémentaires après logger.info()")
    print()
    
    # Appliquer les corrections
    steps = [
        ("Correction syntaxe logger", fix_logger_syntax_error),
        ("Validation syntaxe Python", validate_syntax),
        ("Test import module", test_import)
    ]
    
    success_count = 0
    for name, func in steps:
        print(f"📋 {name}...")
        try:
            if func():
                success_count += 1
            else:
                print(f"❌ {name} échoué")
        except Exception as e:
            print(f"❌ {name} erreur: {e}")
    
    print(f"\n🎯 RÉSULTATS: {success_count}/{len(steps)} étapes réussies")
    
    if success_count >= 2:
        print("\n✅ CORRECTION TERMINÉE!")
        print("📋 Prochaines étapes:")
        print("1. Committez la correction:")
        print("   git add api_games_plus.py")
        print("   git commit -m 'fix: Remove extra parentheses in logger call'")
        print("   git push")
        print()
        print("2. L'API Render devrait maintenant démarrer correctement")
        print("3. Le health check devrait retourner 200 OK")
        print("4. Votre pipeline de monitoring devrait passer")
        return True
    else:
        print("\n❌ Des problèmes persistent")
        print("🔧 Vérifiez manuellement le fichier api_games_plus.py")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
