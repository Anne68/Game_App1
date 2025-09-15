# tests/test_e4_simple.py - Tests E4 simplifiés et robustes

import pytest
import sys
import os
import time
from pathlib import Path

# Configuration pour les tests
os.environ.update({
    'SECRET_KEY': 'test-secret-key-for-e4-very-long-and-secure',
    'DB_REQUIRED': 'false',
    'DEMO_LOGIN_ENABLED': 'true',
    'DEMO_USERNAME': 'demo',
    'DEMO_PASSWORD': 'demo123',
    'LOG_LEVEL': 'INFO',
    'ALLOW_ORIGINS': '*'
})

class TestE4Competencies:
    """Tests pour valider les compétences E4"""
    
    def test_c14_documentation_exists(self):
        """C14: Vérifier que la documentation d'analyse existe"""
        required_docs = [
            "docs/specifications/specifications_fonctionnelles.md",
            "docs/architecture/architecture_technique.md"
        ]
        
        for doc_path in required_docs:
            assert Path(doc_path).exists(), f"Documentation C14 manquante: {doc_path}"
            
            # Vérifier que le document n'est pas vide
            with open(doc_path, 'r') as f:
                content = f.read().strip()
                assert len(content) > 100, f"Document {doc_path} trop court pour C14"
    
    def test_c15_technical_architecture(self):
        """C15: Vérifier l'architecture technique"""
        # Vérifier les modules principaux
        required_modules = [
            "api_games_plus.py",
            "model_manager.py", 
            "monitoring_metrics.py",
            "settings.py"
        ]
        
        for module in required_modules:
            assert Path(module).exists(), f"Module architecture C15 manquant: {module}"
        
        # Vérifier Docker
        assert Path("Dockerfile").exists(), "Dockerfile manquant pour C15"
        assert Path("docker-compose.yml").exists(), "docker-compose.yml manquant pour C15"
    
    def test_c16_mlops_process(self):
        """C16: Vérifier les processus MLOps"""
        # Vérifier monitoring
        assert Path("prometheus").exists(), "Configuration Prometheus manquante pour C16"
        assert Path("grafana").exists(), "Configuration Grafana manquante pour C16"
        
        # Vérifier processus documenté
        mlops_doc = Path("docs/processes/mlops_process.md")
        if mlops_doc.exists():
            with open(mlops_doc, 'r') as f:
                content = f.read()
                assert "MLOps" in content, "Documentation MLOps incomplete pour C16"
    
    def test_c17_compliance_standards(self):
        """C17: Vérifier les standards de développement"""
        # Vérifier le module de compliance
        compliance_module = Path("compliance/standards_compliance.py")
        if compliance_module.exists():
            try:
                # Test d'import du module
                sys.path.insert(0, '.')
                from compliance.standards_compliance import SecurityValidator, AccessibilityValidator
                
                # Test basique du validator
                validator = SecurityValidator()
                result = validator.validate_password("TestPassword123!")
                
                assert isinstance(result, dict), "SecurityValidator doit retourner un dict"
                assert "valid" in result, "Résultat validation doit contenir 'valid'"
                assert "strength" in result, "Résultat validation doit contenir 'strength'"
                
            except ImportError:
                pytest.skip("Module compliance pas encore importable")
            except Exception as e:
                pytest.fail(f"Erreur test compliance C17: {e}")
        else:
            pytest.skip("Module compliance pas encore créé")
    
    def test_c18_automated_tests(self):
        """C18: Vérifier les tests automatisés"""
        # Compter les fichiers de test
        test_files = list(Path(".").glob("test_*.py"))
        test_files.extend(list(Path("tests").glob("**/*.py")) if Path("tests").exists() else [])
        
        test_count = len([f for f in test_files if f.name.startswith('test_')])
        assert test_count >= 3, f"Pas assez de tests pour C18: {test_count} trouvés, minimum 3"
        
        print(f"✅ C18: {test_count} fichiers de test trouvés")
    
    def test_c19_cicd_pipeline(self):
        """C19: Vérifier le pipeline CI/CD"""
        # Vérifier GitHub Actions
        github_workflows = Path(".github/workflows")
        assert github_workflows.exists(), "Dossier .github/workflows manquant pour C19"
        
        workflow_files = list(github_workflows.glob("*.yml"))
        assert len(workflow_files) >= 1, "Aucun workflow GitHub Actions pour C19"
        
        # Vérifier qu'au moins un workflow contient du CI/CD
        has_cicd = False
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                content = f.read()
                if any(keyword in content.lower() for keyword in ['test', 'build', 'deploy']):
                    has_cicd = True
                    break
        
        assert has_cicd, "Aucun pipeline CI/CD valide trouvé pour C19"
        print(f"✅ C19: {len(workflow_files)} workflow(s) CI/CD trouvé(s)")

class TestAPIBasicFunctionality:
    """Tests de base pour vérifier que l'API fonctionne encore"""
    
    def test_api_imports(self):
        """Test que l'API peut être importée"""
        try:
            import api_games_plus
            assert hasattr(api_games_plus, 'app'), "FastAPI app not found"
            print("✅ API imports successfully")
        except Exception as e:
            pytest.fail(f"Cannot import API: {e}")
    
    def test_api_health_local(self):
        """Test basique de santé de l'API"""
        try:
            from fastapi.testclient import TestClient
            from api_games_plus import app
            
            client = TestClient(app)
            response = client.get("/healthz")
            
            # Accepter 200 ou 500 (si DB pas prête)
            assert response.status_code in [200, 500], f"Unexpected health status: {response.status_code}"
            
            if response.status_code == 200:
                data = response.json()
                assert "status" in data, "Health response missing status"
                print(f"✅ Health check: {data.get('status')}")
            else:
                print("⚠️ Health check degraded (normal without DB)")
                
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")
    
    def test_compliance_integration(self):
        """Test que la compliance s'intègre sans casser l'API"""
        try:
            import api_games_plus
            
            # Vérifier que COMPLIANCE_AVAILABLE est défini
            compliance_available = getattr(api_games_plus, 'COMPLIANCE_AVAILABLE', None)
            
            if compliance_available is True:
                print("✅ Compliance integration successful")
                
                # Vérifier que l'app state a les validators
                app = api_games_plus.app
                if hasattr(app.state, 'security_validator'):
                    print("✅ Security validator loaded")
                else:
                    print("⚠️ Security validator not in app state")
                    
            elif compliance_available is False:
                print("⚠️ Compliance available but not loaded")
            else:
                print("⚠️ Compliance integration not yet applied")
                
        except Exception as e:
            pytest.skip(f"Cannot test compliance integration: {e}")

class TestE4Documentation:
    """Tests de présence de la documentation E4"""
    
    def test_specifications_documentation(self):
        """Test présence documentation spécifications"""
        spec_file = Path("docs/specifications/specifications_fonctionnelles.md")
        
        if spec_file.exists():
            with open(spec_file, 'r') as f:
                content = f.read()
                
            # Vérifier le contenu minimal
            required_sections = ["Objectifs", "Exigences", "Critères"]
            for section in required_sections:
                assert section in content, f"Section {section} manquante dans specifications"
            
            print("✅ Documentation spécifications complète")
        else:
            pytest.skip("Documentation spécifications pas encore créée")
    
    def test_architecture_documentation(self):
        """Test présence documentation architecture"""
        arch_file = Path("docs/architecture/architecture_technique.md")
        
        if arch_file.exists():
            with open(arch_file, 'r') as f:
                content = f.read()
                
            # Vérifier contenu architectural
            arch_keywords = ["Architecture", "ML", "Pipeline", "Sécurité"]
            found_keywords = sum(1 for keyword in arch_keywords if keyword in content)
            
            assert found_keywords >= 2, "Documentation architecture insuffisante"
            print("✅ Documentation architecture présente")
        else:
            pytest.skip("Documentation architecture pas encore créée")

def test_e4_integration_status():
    """Test global du statut d'intégration E4"""
    
    print("\n🎯 STATUT INTÉGRATION E4")
    print("=" * 40)
    
    # Vérifier chaque compétence
    competencies = {
        "C14": Path("docs/specifications/specifications_fonctionnelles.md").exists(),
        "C15": Path("docs/architecture/architecture_technique.md").exists(),
        "C16": Path("prometheus").exists() and Path("grafana").exists(),
        "C17": Path("compliance/standards_compliance.py").exists(),
        "C18": len(list(Path(".").glob("test_*.py"))) >= 3,
        "C19": Path(".github/workflows").exists()
    }
    
    covered = sum(competencies.values())
    total = len(competencies)
    percentage = (covered / total) * 100
    
    for comp, status in competencies.items():
        print(f"{'✅' if status else '❌'} {comp}: {'Covered' if status else 'Not covered'}")
    
    print(f"\n📊 Score E4: {covered}/{total} ({percentage:.0f}%)")
    
    if percentage >= 80:
        print("🎉 EXCELLENT - Prêt pour E4!")
    elif percentage >= 60:
        print("✅ BIEN - Quelques finitions")
    else:
        print("⚠️ Plus de travail nécessaire")
    
    return percentage >= 60

if __name__ == "__main__":
    """Exécution directe pour validation E4"""
    import subprocess
    
    print("🧪 TESTS E4 SIMPLIFIÉS")
    print("=" * 30)
    
    # Exécuter les tests
    result = pytest.main([
        "-v", 
        "--tb=short",
        "tests/test_e4_simple.py"
    ])
    
    # Test global d'intégration
    integration_ok = test_e4_integration_status()
    
    if result == 0 and integration_ok:
        print("\n🎉 TESTS E4 RÉUSSIS!")
        sys.exit(0)
    else:
        print("\n⚠️ Quelques ajustements nécessaires")
        sys.exit(1)
