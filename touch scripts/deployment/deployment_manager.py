# scripts/deployment/deployment_manager.py - Gestionnaire de déploiement

import os
import time
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deployment-manager')

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    """Configuration de déploiement"""
    environment: str
    service_url: str
    health_check_path: str = "/healthz"
    timeout_seconds: int = 600
    rollback_enabled: bool = True
    smoke_tests_enabled: bool = True

class DeploymentManager:
    """Gestionnaire de déploiement avec capacités MLOps"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_id = f"deploy-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        self.deployment_log = []
        
    def deploy(self, image_tag: str) -> Dict[str, Any]:
        """Déploie une nouvelle version de l'application"""
        
        deployment_result = {
            "deployment_id": self.deployment_id,
            "status": DeploymentStatus.PENDING,
            "start_time": datetime.utcnow().isoformat(),
            "environment": self.config.environment,
            "image_tag": image_tag,
            "steps_completed": [],
            "errors": []
        }
        
        try:
            logger.info(f"Starting deployment {self.deployment_id} to {self.config.environment}")
            deployment_result["status"] = DeploymentStatus.IN_PROGRESS
            
            # Étape 1: Pré-déploiement
            self._log_step("Pre-deployment checks")
            if not self._pre_deployment_checks():
                raise Exception("Pre-deployment checks failed")
            deployment_result["steps_completed"].append("pre_checks")
            
            # Étape 2: Sauvegarde version actuelle
            self._log_step("Backing up current version")
            current_version = self._backup_current_version()
            deployment_result["previous_version"] = current_version
            deployment_result["steps_completed"].append("backup")
            
            # Étape 3: Déploiement nouvelle version
            self._log_step("Deploying new version")
            deploy_success = self._deploy_new_version(image_tag)
            if not deploy_success:
                raise Exception("Deployment of new version failed")
            deployment_result["steps_completed"].append("deploy")
            
            # Étape 4: Health checks
            self._log_step("Running health checks")
            if not self._wait_for_health(timeout=300):
                raise Exception("Health checks failed")
            deployment_result["steps_completed"].append("health_check")
            
            # Étape 5: Tests de fumée
            if self.config.smoke_tests_enabled:
                self._log_step("Running smoke tests")
                smoke_results = self._run_smoke_tests()
                deployment_result["smoke_test_results"] = smoke_results
                if not smoke_results["passed"]:
                    raise Exception("Smoke tests failed")
                deployment_result["steps_completed"].append("smoke_tests")
            
            deployment_result["status"] = DeploymentStatus.SUCCESS
            deployment_result["end_time"] = datetime.utcnow().isoformat()
            
            logger.info(f"Deployment {self.deployment_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Deployment {self.deployment_id} failed: {e}")
            deployment_result["status"] = DeploymentStatus.FAILED
            deployment_result["errors"].append(str(e))
            deployment_result["end_time"] = datetime.utcnow().isoformat()
            
            # Rollback automatique si possible
            if self.config.rollback_enabled and "backup" in deployment_result["steps_completed"]:
                logger.info("Attempting automatic rollback...")
                rollback_result = self._rollback(current_version)
                deployment_result["rollback_attempted"] = True
                deployment_result["rollback_success"] = rollback_result
                
                if rollback_result:
                    deployment_result["status"] = DeploymentStatus.ROLLED_BACK
        
        finally:
            self._save_deployment_log(deployment_result)
        
        return deployment_result
    
    def _pre_deployment_checks(self) -> bool:
        """Vérifications avant déploiement"""
        try:
            response = requests.get(
                f"{self.config.service_url}{self.config.health_check_path}",
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Service not healthy before deployment: {response.status_code}")
                return False
            
            logger.info("Pre-deployment checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Pre-deployment checks failed: {e}")
            return False
    
    def _backup_current_version(self) -> Optional[str]:
        """Sauvegarde la version actuelle pour rollback"""
        try:
            response = requests.get(
                f"{self.config.service_url}{self.config.health_check_path}",
                timeout=30
            )
            
            if response.status_code == 200:
                health_data = response.json()
                current_version = health_data.get("model_version", "unknown")
                logger.info(f"Current version backed up: {current_version}")
                return current_version
            
        except Exception as e:
            logger.warning(f"Could not backup current version: {e}")
        
        return None
    
    def _deploy_new_version(self, image_tag: str) -> bool:
        """Déploie la nouvelle version"""
        try:
            logger.info(f"Deploying image: {image_tag}")
            
            # Simulation du déploiement
            time.sleep(30)
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _wait_for_health(self, timeout: int = 300) -> bool:
        """Attend que le service soit sain"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.config.service_url}{self.config.health_check_path}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        logger.info("Service is healthy")
                        return True
                
                logger.info(f"Waiting for service to be healthy... ({response.status_code})")
                
            except requests.RequestException as e:
                logger.info(f"Health check failed, retrying... ({e})")
            
            time.sleep(10)
        
        logger.error("Service did not become healthy within timeout")
        return False
    
    def _run_smoke_tests(self) -> Dict[str, Any]:
        """Exécute les tests de fumée"""
        smoke_results = {
            "passed": False,
            "tests_run": 0,
            "tests_passed": 0,
            "details": []
        }
        
        smoke_tests = [
            {"name": "Health Check", "endpoint": "/healthz", "expected_status": 200},
            {"name": "Metrics Check", "endpoint": "/metrics", "expected_status": 200},
        ]
        
        for test in smoke_tests:
            smoke_results["tests_run"] += 1
            
            try:
                response = requests.get(
                    f"{self.config.service_url}{test['endpoint']}",
                    timeout=30
                )
                
                if response.status_code == test["expected_status"]:
                    smoke_results["tests_passed"] += 1
                    smoke_results["details"].append({
                        "test": test["name"],
                        "status": "PASSED",
                        "response_time": response.elapsed.total_seconds()
                    })
                else:
                    smoke_results["details"].append({
                        "test": test["name"],
                        "status": "FAILED",
                        "error": f"Expected {test['expected_status']}, got {response.status_code}"
                    })
                    
            except Exception as e:
                smoke_results["details"].append({
                    "test": test["name"],
                    "status": "ERROR",
                    "error": str(e)
                })
        
        smoke_results["passed"] = smoke_results["tests_passed"] == smoke_results["tests_run"]
        
        logger.info(f"Smoke tests: {smoke_results['tests_passed']}/{smoke_results['tests_run']} passed")
        return smoke_results
    
    def _rollback(self, previous_version: Optional[str]) -> bool:
        """Effectue un rollback vers la version précédente"""
        try:
            logger.info(f"Rolling back to version: {previous_version}")
            
            # Simulation du rollback
            time.sleep(20)
            
            if self._wait_for_health(timeout=120):
                logger.info("Rollback completed successfully")
                return True
            else:
                logger.error("Rollback failed - service not healthy")
                return False
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _save_deployment_log(self, deployment_result: Dict[str, Any]):
        """Sauvegarde le log de déploiement"""
        try:
            log_file = f"deployment_logs/{self.deployment_id}.json"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            with open(log_file, 'w') as f:
                json.dump(deployment_result, f, indent=2)
                
            logger.info(f"Deployment log saved: {log_file}")
            
        except Exception as e:
            logger.warning(f"Could not save deployment log: {e}")
    
    def _log_step(self, step: str):
        """Log une étape du déploiement"""
        timestamp = datetime.utcnow().isoformat()
        self.deployment_log.append({"timestamp": timestamp, "step": step})
        logger.info(f"Step: {step}")

def main():
    """Script principal de déploiement"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Games Recommendation API")
    parser.add_argument("--environment", required=True, choices=["staging", "production"], 
                       help="Target environment")
    parser.add_argument("--image-tag", required=True, help="Docker image tag to deploy")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip smoke tests")
    
    args = parser.parse_args()
    
    # Configuration de déploiement
    configs = {
        "staging": DeploymentConfig(
            environment="staging",
            service_url="https://games-api-staging.onrender.com",
            rollback_enabled=True,
            smoke_tests_enabled=not args.skip_tests,
        ),
        "production": DeploymentConfig(
            environment="production", 
            service_url="https://game-app-y8be.onrender.com",
            rollback_enabled=True,
            smoke_tests_enabled=not args.skip_tests,
        )
    }
    
    config = configs[args.environment]
    
    # Déploiement
    manager = DeploymentManager(config)
    result = manager.deploy(args.image_tag)
    
    # Affichage du résultat
    print(f"\n{'='*60}")
    print(f"DEPLOYMENT RESULT: {result['status'].value.upper()}")
    print(f"{'='*60}")
    print(f"Deployment ID: {result['deployment_id']}")
    print(f"Environment: {result['environment']}")
    print(f"Image: {result['image_tag']}")
    print(f"Steps completed: {', '.join(result['steps_completed'])}")
    
    if result['errors']:
        print(f"Errors: {', '.join(result['errors'])}")
    
    if result['status'] == DeploymentStatus.SUCCESS:
        print("🎉 Deployment completed successfully!")
        exit(0)
    elif result['status'] == DeploymentStatus.ROLLED_BACK:
        print("🔄 Deployment failed but rollback successful")
        exit(1)
    else:
        print("❌ Deployment failed")
        exit(1)

if __name__ == "__main__":
    main()
