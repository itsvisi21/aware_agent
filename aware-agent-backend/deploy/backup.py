import os
import json
import shutil
import datetime
import asyncio
import subprocess
import logging
import sys
from typing import Dict, List, Optional
from pathlib import Path
from deploy.config import DeploymentConfig
from deploy.monitoring import MonitoringService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupService:
    def __init__(self):
        self.config = DeploymentConfig()
        self.monitoring = MonitoringService(self.config)
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    async def create_backup(self, backup_type: str = "full") -> str:
        """Create a backup of the system"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{backup_type}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        try:
            # Create backup directory
            backup_path.mkdir()
            
            # Backup database
            if self.config.database_url.startswith("sqlite"):
                db_path = self.config.database_url.replace("sqlite:///", "")
                if os.path.exists(db_path):
                    shutil.copy2(db_path, backup_path / "database.db")
            
            # Backup configuration
            config_backup = {
                'database_url': self.config.database_url,
                'cache_ttl': self.config.cache_ttl,
                'websocket_port': self.config.websocket_port,
                'log_level': self.config.log_level,
                'debug_mode': self.config.debug_mode
            }
            with open(backup_path / "config.json", "w") as f:
                json.dump(config_backup, f, indent=2)
            
            # Backup logs
            logs_dir = Path("logs")
            if logs_dir.exists():
                shutil.copytree(logs_dir, backup_path / "logs")
            
            # Create backup manifest
            manifest = {
                'backup_name': backup_name,
                'timestamp': timestamp,
                'backup_type': backup_type,
                'components': ['database', 'config', 'logs'],
                'status': 'success'
            }
            with open(backup_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            
            # Log backup creation
            self.monitoring.log_info(f"Backup created: {backup_name}")
            
            return backup_name
            
        except Exception as e:
            self.monitoring.log_error(f"Backup failed: {str(e)}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    async def restore_backup(self, backup_name: str) -> bool:
        """Restore a backup"""
        backup_path = self.backup_dir / backup_name
        
        try:
            # Verify backup exists
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup {backup_name} not found")
            
            # Read manifest
            with open(backup_path / "manifest.json", "r") as f:
                manifest = json.load(f)
            
            # Restore database
            if (backup_path / "database.db").exists():
                db_path = self.config.database_url.replace("sqlite:///", "")
                shutil.copy2(backup_path / "database.db", db_path)
            
            # Restore configuration
            if (backup_path / "config.json").exists():
                with open(backup_path / "config.json", "r") as f:
                    config_backup = json.load(f)
                # Update config with backup values
                self.config.database_url = config_backup['database_url']
                self.config.cache_ttl = config_backup['cache_ttl']
                self.config.websocket_port = config_backup['websocket_port']
                self.config.log_level = config_backup['log_level']
                self.config.debug_mode = config_backup['debug_mode']
            
            # Restore logs
            if (backup_path / "logs").exists():
                logs_dir = Path("logs")
                if logs_dir.exists():
                    shutil.rmtree(logs_dir)
                shutil.copytree(backup_path / "logs", logs_dir)
            
            # Log restore
            self.monitoring.log_info(f"Backup restored: {backup_name}")
            
            return True
            
        except Exception as e:
            self.monitoring.log_error(f"Restore failed: {str(e)}")
            return False
    
    async def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                manifest_path = backup_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                        backups.append(manifest)
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    async def delete_backup(self, backup_name: str) -> bool:
        """Delete a backup"""
        backup_path = self.backup_dir / backup_name
        
        try:
            if backup_path.exists():
                shutil.rmtree(backup_path)
                self.monitoring.log_info(f"Backup deleted: {backup_name}")
                return True
            return False
        except Exception as e:
            self.monitoring.log_error(f"Delete backup failed: {str(e)}")
            return False

class BackupManager:
    def __init__(self):
        self.config = DeploymentConfig()
    
    def backup_database(self):
        """Backup PostgreSQL database"""
        logger.info("Backing up database...")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.config.backup_dir, f"db_backup_{timestamp}.sql")
        
        subprocess.run([
            "pg_dump",
            "-h", "localhost",
            "-U", "postgres",
            "-d", "aware_agent",
            "-f", backup_file
        ], check=True)
        
        # Compress backup
        subprocess.run(["gzip", backup_file], check=True)
        
        return f"{backup_file}.gz"
    
    def backup_redis(self):
        """Backup Redis data"""
        logger.info("Backing up Redis data...")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.config.backup_dir, f"redis_backup_{timestamp}.rdb")
        
        subprocess.run([
            "redis-cli",
            "SAVE"
        ], check=True)
        
        # Copy Redis dump file
        subprocess.run([
            "cp",
            "/var/lib/redis/dump.rdb",
            backup_file
        ], check=True)
        
        # Compress backup
        subprocess.run(["gzip", backup_file], check=True)
        
        return f"{backup_file}.gz"
    
    def cleanup_old_backups(self):
        """Remove old backups"""
        logger.info("Cleaning up old backups...")
        
        # List all backup files
        backup_files = os.listdir(self.config.backup_dir)
        
        # Sort by modification time
        backup_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.config.backup_dir, x)))
        
        # Remove old backups
        while len(backup_files) > self.config.backup_retention:
            old_backup = backup_files.pop(0)
            os.remove(os.path.join(self.config.backup_dir, old_backup))
    
    def restore_database(self, backup_file: str):
        """Restore PostgreSQL database from backup"""
        logger.info(f"Restoring database from {backup_file}...")
        
        # Decompress backup
        subprocess.run(["gunzip", backup_file], check=True)
        sql_file = backup_file[:-3]  # Remove .gz extension
        
        # Restore database
        subprocess.run([
            "psql",
            "-h", "localhost",
            "-U", "postgres",
            "-d", "aware_agent",
            "-f", sql_file
        ], check=True)
        
        # Remove decompressed file
        os.remove(sql_file)
    
    def restore_redis(self, backup_file: str):
        """Restore Redis data from backup"""
        logger.info(f"Restoring Redis data from {backup_file}...")
        
        # Decompress backup
        subprocess.run(["gunzip", backup_file], check=True)
        rdb_file = backup_file[:-3]  # Remove .gz extension
        
        # Stop Redis
        subprocess.run(["redis-cli", "SHUTDOWN"], check=True)
        
        # Copy backup file
        subprocess.run([
            "cp",
            rdb_file,
            "/var/lib/redis/dump.rdb"
        ], check=True)
        
        # Start Redis
        subprocess.run(["redis-server", "--daemonize", "yes"], check=True)
        
        # Remove decompressed file
        os.remove(rdb_file)
    
    def backup(self):
        """Execute backup process"""
        try:
            db_backup = self.backup_database()
            redis_backup = self.backup_redis()
            self.cleanup_old_backups()
            logger.info(f"Backup completed successfully: {db_backup}, {redis_backup}")
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            sys.exit(1)
    
    def restore(self, db_backup: str, redis_backup: str):
        """Execute restore process"""
        try:
            self.restore_database(db_backup)
            self.restore_redis(redis_backup)
            logger.info("Restore completed successfully")
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            sys.exit(1)

async def main():
    backup_service = BackupService()
    
    # Create a backup
    backup_name = await backup_service.create_backup()
    print(f"Created backup: {backup_name}")
    
    # List backups
    backups = await backup_service.list_backups()
    print("\nAvailable backups:")
    for backup in backups:
        print(f"- {backup['backup_name']} ({backup['timestamp']})")
    
    # Restore backup (example)
    if backups:
        success = await backup_service.restore_backup(backups[0]['backup_name'])
        print(f"\nRestore {'successful' if success else 'failed'}")

if __name__ == "__main__":
    asyncio.run(main()) 