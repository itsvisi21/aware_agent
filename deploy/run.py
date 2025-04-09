import os
import sys
import logging
from typing import Optional
from config import DeploymentConfig
from deploy import DeploymentManager
from monitoring import MonitoringSetup
from backup import BackupManager
from scaling import ScalingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(os.path.join(log_dir, "deployment.log"))
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def validate_environment() -> Optional[str]:
    """Validate deployment environment"""
    config = DeploymentConfig()
    
    # Check required environment variables
    required_vars = [
        "DATABASE_URL",
        "REDIS_URL",
        "AUTH_SECRET"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            return f"Missing required environment variable: {var}"
    
    # Check AWS credentials
    if config.is_production:
        aws_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION"
        ]
        
        for var in aws_vars:
            if not os.getenv(var):
                return f"Missing required AWS environment variable: {var}"
    
    return None

def main():
    """Main deployment function"""
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting deployment process...")
        
        # Validate environment
        if error := validate_environment():
            logger.error(f"Environment validation failed: {error}")
            sys.exit(1)
        
        # Initialize managers
        deployment_manager = DeploymentManager()
        monitoring_setup = MonitoringSetup()
        backup_manager = BackupManager()
        scaling_manager = ScalingManager()
        
        # Execute deployment steps
        logger.info("Step 1: Deploying application...")
        deployment_manager.deploy()
        
        logger.info("Step 2: Setting up monitoring...")
        monitoring_setup.setup()
        
        logger.info("Step 3: Creating initial backup...")
        backup_manager.backup()
        
        logger.info("Step 4: Setting up scaling...")
        scaling_manager.check_and_scale()
        
        logger.info("Deployment completed successfully!")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 