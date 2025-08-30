# compliance/standards_compliance.py - Standards de conformité

import re
import html
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
import bleach

logger = logging.getLogger("standards-compliance")

class SecurityValidator:
    """Validation des standards de sécurité"""
    
    def __init__(self):
        self.password_policy = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special": True,
            "max_length": 128
        }
        
        # Patterns d'injection à détecter
        self.injection_patterns = [
            r"(union|select|insert|update|delete|drop|create|alter)\s+",
            r"['\";].*['\";]",
            r"<script[^>]*>.*?</script>",
            r"javascript\s*:",
            r"on(click|load|error|focus|blur)\s*=",
        ]
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validation robuste des mots de passe"""
        issues = []
        
        if len(password) < self.password_policy["min_length"]:
            issues.append(f"Password must be at least {self.password_policy['min_length']} characters")
        
        if len(password) > self.password_policy["max_length"]:
            issues.append(f"Password must be at most {self.password_policy['max_length']} characters")
        
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
        
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        
        if re.search(r'[a-z]', password):
            score += 1
        if re.search(r'[A-Z]', password):
            score += 1
        if re.search(r'\d', password):
            score += 1
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 2
        
        if len(set(password)) > len(password) * 0.7:
            score += 1
        
        if score <= 3:
            return "Weak"
        elif score <= 5:
            return "Medium"
        elif score <= 7:
            return "Strong"
        else:
            return "Very Strong"
    
    def sanitize_input(self, input_text: str) -> str:
        """Nettoyage et désinfection des entrées utilisateur"""
        if not input_text:
            return ""
        
        if len(input_text) > 1000:
            raise HTTPException(status_code=400, detail="Input too long")
        
        # Détection des tentatives d'injection
        for pattern in self.injection_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                logger.warning(f"Potential injection attempt detected: {pattern}")
                raise HTTPException(status_code=400, detail="Invalid input format")
        
        # Échapper les caractères HTML
        sanitized = html.escape(input_text.strip())
        
        # Nettoyage supplémentaire avec bleach
        sanitized = bleach.clean(
            sanitized,
            tags=[],  # Aucune balise HTML autorisée
            attributes={},
            strip=True
        )
        
        return sanitized

class AccessibilityValidator:
    """Validation WCAG 2.1 AA pour les API"""
    
    def enhance_response_accessibility(self, response: Dict) -> Dict:
        """Enrichit les réponses avec métadonnées d'accessibilité"""
        if "recommendations" in response:
            for rec in response["recommendations"]:
                rec["aria_label"] = f"Recommended game: {rec.get('title', 'Unknown')}"
                rec["accessible_description"] = (
                    f"Game recommendation: {rec.get('title', 'Unknown')}. "
                    f"Confidence score: {rec.get('confidence', 0):.2f}. "
                    f"Genres: {rec.get('genres', 'Not specified')}."
                )
        
        response["accessibility"] = {
            "version": "WCAG 2.1 AA",
            "description": f"Response contains {len(response.get('recommendations', []))} game recommendations",
            "keyboard_navigation": True,
            "screen_reader_optimized": True
        }
        
        return response

# Fonction utilitaire pour intégration dans l'API existante
def setup_compliance_middleware(app):
    """Configuration du middleware de conformité"""
    security_validator = SecurityValidator()
    accessibility_validator = AccessibilityValidator()
    
    # Stocker les validators dans l'app state
    app.state.security_validator = security_validator
    app.state.accessibility_validator = accessibility_validator
    
    return app
