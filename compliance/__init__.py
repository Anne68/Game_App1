# compliance/__init__.py
"""
Module de conformité E4 - Version minimale pour déploiement
"""

__version__ = "1.0.0"

try:
    from .standards_compliance import SecurityValidator, AccessibilityValidator
    __all__ = ['SecurityValidator', 'AccessibilityValidator']
    COMPLIANCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Compliance components not available: {e}")
    __all__ = []
    COMPLIANCE_AVAILABLE = False
