# Testez avec le script rapide
python test_quick_fix.py

# Puis relancez vos tests d'import
python -c "
import sys
import os
sys.path.insert(0, '.')

try:
    from compliance.standards_compliance import SecurityValidator
    print('✅ Compliance import successful')
    
    import api_games_plus
    print('✅ API import successful')
    
    print('🎉 All imports working!')
except Exception as e:
    print(f'❌ Error: {e}')
"
