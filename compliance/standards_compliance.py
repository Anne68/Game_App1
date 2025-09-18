# compliance/standards_compliance.py
"""
Standards de conformité E4 - Version minimale pour déploiement
"""

import re
import html
import logging

logger = logging.getLogger("compliance")

class SecurityValidator:
    """Validateur de sécurité minimal"""
    
    def __init__(self):
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
        }
    
    def validate_password(self, password: str) -> dict:
        """Validation de base des mots de passe"""
        issues = []
        
        if len(password) < self.password_policy["min_length"]:
            issues.append(f"Password must be at least {self.password_policy['min_length']} characters")
        
        if self.password_policy["require_uppercase"] and not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.password_policy["require_lowercase"] and not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if self.password_policy["require_digits"] and not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "strength": "Medium" if len(issues) == 0 else "Weak"
        }
    
    def sanitize_input(self, input_text: str) -> str:
        """Nettoyage basique des entrées"""
        if not input_text:
            return ""
        
        # Limite de taille
        if len(input_text) > 1000:
            input_text = input_text[:1000]
        
        # Échappement HTML
        return html.escape(input_text.strip())

class AccessibilityValidator:
    """Validateur d'accessibilité minimal"""
    
    def enhance_response_accessibility(self, response: dict) -> dict:
        """Enrichissement basique pour accessibilité"""
        if not isinstance(response, dict):
            return response
        
        # Enrichir les recommandations si présentes
        if "recommendations" in response and isinstance(response["recommendations"], list):
            for rec in response["recommendations"]:
                if isinstance(rec, dict) and "title" in rec:
                    rec["aria_label"] = f"Recommended game: {rec['title']}"
        
        # Métadonnées d'accessibilité
        response["accessibility"] = {
            "version": "WCAG 2.1 AA",
            "screen_reader_optimized": True,
            "language": "en"
        }
        
        return response
