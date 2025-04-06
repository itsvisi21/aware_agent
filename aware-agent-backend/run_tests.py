import os
import sys
import subprocess
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_tests(test_files: Optional[List[str]] = None) -> bool:
    """Run the test suite"""
    try:
        # Setup environment
        os.environ["PYTHONPATH"] = os.getcwd()
        os.environ["TESTING"] = "true"
        
        # Default test files if none specified
        if not test_files:
            test_files = [
                "tests/final/system_test.py",
                "tests/final/performance_test.py",
                "tests/final/security_test.py",
                "tests/final/load_test.py",
                "tests/final/security_audit.py",
                "tests/final/backup_recovery.py"
            ]
        
        # Run each test file
        for test_file in test_files:
            logger.info(f"Running tests in {test_file}...")
            
            result = subprocess.run([
                "pytest",
                test_file,
                "-v",
                "--cov=src",
                "--cov-report=term-missing"
            ], check=False)
            
            if result.returncode != 0:
                logger.error(f"Tests in {test_file} failed")
                return False
        
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        return False

def main():
    """Main function"""
    try:
        # Get test files from command line arguments
        test_files = sys.argv[1:] if len(sys.argv) > 1 else None
        
        # Run tests
        success = run_tests(test_files)
        
        # Exit with appropriate status code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 