# compliance/__init__.py
"""
Module de conformité pour E4 - Compétence C17
Standards de sécurité et d'accessibilité
"""

__version__ = "1.0.0"

# Import sécurisé des composants
try:
    from .standards_compliance import SecurityValidator, AccessibilityValidator
    __all__ = ['SecurityValidator', 'AccessibilityValidator']
    COMPLIANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Compliance components not available: {e}")
    __all__ = []
    COMPLIANCE_AVAILABLE = False
