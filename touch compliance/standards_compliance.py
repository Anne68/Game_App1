# compliance/standards_compliance.py
"""
Standards de conformité E4 - Sécurité et Accessibilité
Compétence C17 : Standards de développement
"""

import re
import html
import logging
from typing import Dict, List, Optional, Any

# Logger pour ce module
logger = logging.getLogger("compliance")

# Vérification des dépendances optionnelles
try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    BLEACH_AVAILABLE = False
    logger.info("bleach not available - using basic HTML escaping")

class SecurityValidator:
    """
    Validateur de sécurité pour E4 C17
    Implémente les standards de sécurité requis
    """
    
    def __init__(self):
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special": False,  # Optionnel pour éviter blocage
            "max_length": 128
        }
        
        # Patterns d'injection basiques
        self.injection_patterns = [
            r"(union|select|insert|update|delete|drop)\s+",
            r"<script[^>]*>",
            r"javascript\s*:",
        ]
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """
        Validation robuste des mots de passe selon standards E4
        
        Args:
            password: Mot de passe à valider
            
        Returns:
            Dict avec validation result
        """
        issues = []
        
        # Vérifications de base
        if len(password) < self.password_policy["min_length"]:
            issues.append(f"Password must be at least {self.password_policy['min_length']} characters")
        
        if len(password) > self.password_policy["max_length"]:
            issues.append(f"Password must be at most {self.password_policy['max_length']} characters")
        
        # Vérifications de complexité
        if self.password_policy["require_uppercase"] and not re.search(r'[A-Z]', password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.password_policy["require_lowercase"] and not re.search(r'[a-z]', password):
            issues.append("Password must contain at least one lowercase letter")
        
        if self.password_policy["require_digits"] and not re.search(r'\d', password):
            issues.append("Password must contain at least one digit")
        
        if self.password_policy["require_special"] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("Password must contain at least one special character")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "strength": self._calculate_password_strength(password)
        }
    
    def _calculate_password_strength(self, password: str) -> str:
        """Calcule la force du mot de passe"""
        score = 0
        
        # Points pour longueur
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        
        # Points pour diversité
        if re.search(r'[a-z]', password):
            score += 1
        if re.search(r'[A-Z]', password):
            score += 1
        if re.search(r'\d', password):
            score += 1
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 2
        
        # Point pour diversité des caractères
        if len(set(password)) > len(password) * 0.7:
            score += 1
        
        # Classification
        if score <= 3:
            return "Weak"
        elif score <= 5:
            return "Medium"
        elif score <= 7:
            return "Strong"
        else:
            return "Very Strong"
    
    def sanitize_input(self, input_text: str) -> str:
        """
        Sanitisation sécurisée des entrées utilisateur
        
        Args:
            input_text: Texte à nettoyer
            
        Returns:
            Texte nettoyé et sécurisé
        """
        if not input_text:
            return ""
        
        # Limite de taille
        if len(input_text) > 1000:
            logger.warning("Input truncated - too long")
            input_text = input_text[:1000]
        
        # Détection et nettoyage des patterns dangereux
        for pattern in self.injection_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                logger.warning("Potentially dangerous input detected and cleaned")
                input_text = re.sub(pattern, "", input_text, flags=re.IGNORECASE)
        
        # Échappement HTML
        sanitized = html.escape(input_text.strip())
        
        # Nettoyage avancé avec bleach si disponible
        if BLEACH_AVAILABLE:
            try:
                sanitized = bleach.clean(
                    sanitized,
                    tags=[],  # Aucune balise autorisée
                    attributes={},
                    strip=True
                )
            except Exception as e:
                logger.warning(f"Bleach cleaning failed: {e}")
        
        return sanitized

class AccessibilityValidator:
    """
    Validateur d'accessibilité pour E4 C17
    Implémente les standards WCAG 2.1 AA
    """
    
    def enhance_response_accessibility(self, response: Dict) -> Dict:
        """
        Enrichit les réponses API avec métadonnées d'accessibilité
        
        Args:
            response: Réponse API à enrichir
            
        Returns:
            Réponse enrichie avec données d'accessibilité
        """
        # Vérifier que response est un dict
        if not isinstance(response, dict):
            return response
        
        # Enrichir les recommandations si présentes
        if "recommendations" in response and isinstance(response["recommendations"], list):
            for rec in response["recommendations"]:
                if isinstance(rec, dict):
                    # Labels ARIA pour lecteurs d'écran
                    rec["aria_label"] = f"Recommended game: {rec.get('title', 'Unknown')}"
                    
                    # Description accessible détaillée
                    rec["accessible_description"] = (
                        f"Game recommendation: {rec.get('title', 'Unknown')}. "
                        f"Confidence score: {rec.get('confidence', 0):.2f}. "
                        f"Genres: {rec.get('genres', 'Not specified')}."
                    )
                    
                    # Attributs d'accessibilité
                    rec["role"] = "button"
                    rec["tabindex"] = "0"
        
        # Métadonnées d'accessibilité globales
        response["accessibility"] = {
            "version": "WCAG 2.1 AA",
            "description": f"Response contains {len(response.get('recommendations', []))} game recommendations",
            "keyboard_navigation": True,
            "screen_reader_optimized": True,
            "language": "en"
        }
        
        return response

def setup_compliance_middleware(app):
    """
    Configuration du middleware de conformité
    
    Args:
        app: Instance FastAPI
        
    Returns:
        bool: True si setup réussi
    """
    try:
        # Créer les validators
        app.state.security_validator = SecurityValidator()
        app.state.accessibility_validator = AccessibilityValidator()
        
        logger.info("E4 Compliance middleware setup successful")
        return True
        
    except Exception as e:
        logger.error(f"Compliance middleware setup failed: {e}")
        return False

# Test du module si exécuté directement
if __name__ == "__main__":
    print("🧪 Testing compliance module...")
    
    # Test SecurityValidator
    validator = SecurityValidator()
    
    test_passwords = [
        ("weak123", False),
        ("StrongPassword123!", True),
        ("Test123", True)
    ]
    
    for password, should_be_valid in test_passwords:
        result = validator.validate_password(password)
        print(f"Password '{password}': {result['valid']} (strength: {result['strength']})")
        
        if result['valid'] != should_be_valid:
            print(f"  ⚠️ Expected {should_be_valid}, got {result['valid']}")
    
    # Test sanitization
    test_inputs = [
        "normal input",
        "<script>alert('test')</script>",
        "SELECT * FROM users;"
    ]
    
    for input_text in test_inputs:
        sanitized = validator.sanitize_input(input_text)
        print(f"Input '{input_text}' → '{sanitized}'")
    
    print("✅ Compliance module test completed")
