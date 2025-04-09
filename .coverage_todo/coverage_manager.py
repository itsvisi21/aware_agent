import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict

class CoverageManager:
    def __init__(self):
        # Get the absolute path of the workspace root
        self.workspace_root = Path(__file__).parent.parent.absolute()
        self.backend_dir = self.workspace_root / "aware-agent-backend"
        self.frontend_dir = self.workspace_root / "aware-agent-frontend"
        self.backend_todo = self.workspace_root / ".coverage_todo/backend_files.txt"
        self.frontend_todo = self.workspace_root / ".coverage_todo/frontend_files.txt"
        self.backend_coverage = self.workspace_root / ".coverage_todo/backend_coverage.json"
        self.frontend_coverage = self.workspace_root / ".coverage_todo/frontend_coverage.json"
        
        # Create todo files if they don't exist
        self.backend_todo.parent.mkdir(exist_ok=True)
        self.frontend_todo.parent.mkdir(exist_ok=True)
        self.backend_todo.touch(exist_ok=True)
        self.frontend_todo.touch(exist_ok=True)

    def run_initial_coverage(self):
        """Run initial coverage for both frontend and backend"""
        print("Running initial coverage scan...")
        
        # Backend coverage
        print("\nRunning backend coverage...")
        os.chdir(str(self.backend_dir))
        subprocess.run(["coverage", "run", "-m", "pytest"], check=True)
        subprocess.run(["coverage", "json", "-o", str(self.backend_coverage)], check=True)
        os.chdir(str(self.workspace_root))
        
        # Frontend coverage
        print("\nRunning frontend coverage...")
        os.chdir(str(self.frontend_dir))
        subprocess.run(["npm", "run", "test", "--", "--coverage", "--json", 
                       f"--outputFile={self.frontend_coverage}"], check=True)
        os.chdir(str(self.workspace_root))
        
        print("\nInitial coverage scan complete!")

    def parse_coverage_files(self):
        """Parse coverage files and update todo lists"""
        print("\nParsing coverage files...")
        
        # Backend coverage
        with open(self.backend_coverage) as f:
            backend_data = json.load(f)
            uncovered_files = [
                file for file, data in backend_data.items()
                if any(stat < 100 for stat in data.values())
            ]
            with open(self.backend_todo, 'w') as f:
                f.write('\n'.join(uncovered_files))
        
        # Frontend coverage
        with open(self.frontend_coverage) as f:
            frontend_data = json.load(f)
            uncovered_files = [
                file for file, data in frontend_data.items()
                if any(stat < 100 for stat in data.values())
            ]
            with open(self.frontend_todo, 'w') as f:
                f.write('\n'.join(uncovered_files))
        
        print("Coverage files parsed and todo lists updated!")

    def run_file_coverage(self, file_path: str, is_backend: bool):
        """Run coverage for a specific file"""
        if is_backend:
            os.chdir(str(self.backend_dir))
            subprocess.run(["coverage", "run", "-m", "pytest", file_path], check=True)
            os.chdir(str(self.workspace_root))
        else:
            os.chdir(str(self.frontend_dir))
            subprocess.run(["npm", "run", "test", "--", file_path, "--coverage"], check=True)
            os.chdir(str(self.workspace_root))

    def process_todo_files(self):
        """Process files in todo lists"""
        print("\nProcessing todo files...")
        
        # Process backend files
        with open(self.backend_todo) as f:
            backend_files = [line.strip() for line in f if line.strip()]
            for file in backend_files:
                print(f"\nTesting backend file: {file}")
                self.run_file_coverage(file, True)
        
        # Process frontend files
        with open(self.frontend_todo) as f:
            frontend_files = [line.strip() for line in f if line.strip()]
            for file in frontend_files:
                print(f"\nTesting frontend file: {file}")
                self.run_file_coverage(file, False)
        
        print("\nTodo files processed!")

    def run_final_validation(self):
        """Run final validation coverage check"""
        print("\nRunning final validation coverage check...")
        
        # Backend validation
        print("\nValidating backend coverage...")
        os.chdir(str(self.backend_dir))
        subprocess.run(["coverage", "run", "-m", "pytest"], check=True)
        subprocess.run(["coverage", "report"], check=True)
        os.chdir(str(self.workspace_root))
        
        # Frontend validation
        print("\nValidating frontend coverage...")
        os.chdir(str(self.frontend_dir))
        subprocess.run(["npm", "run", "test", "--", "--coverage"], check=True)
        os.chdir(str(self.workspace_root))
        
        print("\nFinal validation complete!")

def main():
    manager = CoverageManager()
    
    # Initial coverage run
    manager.run_initial_coverage()
    
    # Parse coverage files
    manager.parse_coverage_files()
    
    # Process todo files
    manager.process_todo_files()
    
    # Final validation
    manager.run_final_validation()

if __name__ == "__main__":
    main() 