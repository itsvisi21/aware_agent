# Deployment Guide

This guide provides instructions for deploying the Aware Agent system.

## Prerequisites

- Python 3.8 or higher
- PostgreSQL 12 or higher
- Redis 6 or higher
- AWS CLI configured with appropriate credentials
- Docker and Docker Compose (optional)

## Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/aware-agent.git
cd aware-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the environment file:
```bash
cp deploy/.env.example deploy/.env
```

5. Edit the environment file with your configuration:
```bash
nano deploy/.env
```

## Local Deployment

1. Start required services:
```bash
# Start PostgreSQL
docker run --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 -d postgres:12

# Start Redis
docker run --name redis -p 6379:6379 -d redis:6
```

2. Run database migrations:
```bash
alembic upgrade head
```

3. Start the application:
```bash
python deploy/deploy.py
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t aware-agent .
```

2. Run the application with Docker Compose:
```bash
docker-compose up -d
```

## Production Deployment

1. Create an ECS cluster:
```bash
aws ecs create-cluster --cluster-name aware-agent-cluster
```

2. Create an ECR repository:
```bash
aws ecr create-repository --repository-name aware-agent
```

3. Push the Docker image to ECR:
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin your-account-id.dkr.ecr.region.amazonaws.com
docker tag aware-agent:latest your-account-id.dkr.ecr.region.amazonaws.com/aware-agent:latest
docker push your-account-id.dkr.ecr.region.amazonaws.com/aware-agent:latest
```

4. Create a task definition:
```bash
aws ecs register-task-definition --cli-input-json file://deploy/task-definition.json
```

5. Create a service:
```bash
aws ecs create-service --cluster aware-agent-cluster --service-name aware-agent-service --task-definition aware-agent:1 --desired-count 1
```

## Monitoring Setup

1. Setup CloudWatch monitoring:
```bash
python deploy/monitoring.py
```

2. Access the CloudWatch dashboard:
```bash
aws cloudwatch get-dashboard --dashboard-name aware-agent-production
```

## Backup and Recovery

1. Create a backup:
```bash
python deploy/backup.py
```

2. Restore from backup:
```bash
python deploy/backup.py --restore --db-backup backups/db_backup_20230101_120000.sql.gz --redis-backup backups/redis_backup_20230101_120000.rdb.gz
```

## Scaling

1. Enable auto-scaling:
```bash
aws application-autoscaling register-scalable-target --service-namespace ecs --resource-id service/aware-agent-cluster/aware-agent-service --scalable-dimension ecs:service:DesiredCount --min-capacity 1 --max-capacity 10
```

2. Create scaling policies:
```bash
aws application-autoscaling put-scaling-policy --service-namespace ecs --resource-id service/aware-agent-cluster/aware-agent-service --scalable-dimension ecs:service:DesiredCount --policy-name cpu-scaling-policy --policy-type TargetTrackingScaling --target-tracking-scaling-policy-configuration file://deploy/scaling-policy.json
```

## Troubleshooting

1. Check service status:
```bash
aws ecs describe-services --cluster aware-agent-cluster --services aware-agent-service
```

2. Check task status:
```bash
aws ecs list-tasks --cluster aware-agent-cluster --service-name aware-agent-service
```

3. Check logs:
```bash
aws logs get-log-events --log-group-name /aware-agent/production --log-stream-name aware-agent-service
```

## Security

1. SSL/TLS configuration:
```bash
# Generate SSL certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout private.key -out certificate.crt

# Update environment file
SSL_ENABLED=true
SSL_CERT=/path/to/certificate.crt
SSL_KEY=/path/to/private.key
```

2. Network security:
```bash
# Create security group
aws ec2 create-security-group --group-name aware-agent-sg --description "Security group for Aware Agent"

# Add inbound rules
aws ec2 authorize-security-group-ingress --group-name aware-agent-sg --protocol tcp --port 8000 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-name aware-agent-sg --protocol tcp --port 8001 --cidr 0.0.0.0/0
```

## Maintenance

1. Update the application:
```bash
# Pull latest changes
git pull

# Rebuild and push Docker image
docker build -t aware-agent .
docker tag aware-agent:latest your-account-id.dkr.ecr.region.amazonaws.com/aware-agent:latest
docker push your-account-id.dkr.ecr.region.amazonaws.com/aware-agent:latest

# Update service
aws ecs update-service --cluster aware-agent-cluster --service aware-agent-service --force-new-deployment
```

2. Database maintenance:
```bash
# Vacuum database
psql -h localhost -U postgres -d aware_agent -c "VACUUM ANALYZE;"

# Reindex database
psql -h localhost -U postgres -d aware_agent -c "REINDEX DATABASE aware_agent;"
```

3. Cache maintenance:
```bash
# Clear Redis cache
redis-cli FLUSHALL

# Check Redis memory usage
redis-cli INFO MEMORY
``` 