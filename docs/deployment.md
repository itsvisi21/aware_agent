# Deployment Guide

## Prerequisites

- Python 3.8 or higher
- Redis server
- PostgreSQL database
- AWS account (for production deployment)
- Docker and Docker Compose

## Environment Setup

1. **Clone the Repository**
```bash
git clone https://github.com/your-org/aware-agent.git
cd aware-agent/aware-agent-backend
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Variables**
Create `.env` file:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/aware_agent
REDIS_URL=redis://localhost:6379
WEBSOCKET_PORT=8000
LOG_LEVEL=INFO
DEBUG_MODE=false
```

## Local Deployment

1. **Start Redis**
```bash
docker run -d -p 6379:6379 redis
```

2. **Start PostgreSQL**
```bash
docker run -d -p 5432:5432 -e POSTGRES_USER=user -e POSTGRES_PASSWORD=password -e POSTGRES_DB=aware_agent postgres
```

3. **Run Database Migrations**
```bash
alembic upgrade head
```

4. **Start the Application**
```bash
python src/main.py
```

## Docker Deployment

1. **Build Docker Image**
```bash
docker build -t aware-agent:latest .
```

2. **Run with Docker Compose**
```bash
docker-compose up -d
```

## Production Deployment

### AWS Setup

1. **Create ECS Cluster**
```bash
aws ecs create-cluster --cluster-name aware-agent-cluster
```

2. **Create ECR Repository**
```bash
aws ecr create-repository --repository-name aware-agent
```

3. **Push Docker Image**
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag aware-agent:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/aware-agent:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/aware-agent:latest
```

4. **Create Task Definition**
```json
{
    "family": "aware-agent",
    "networkMode": "awsvpc",
    "containerDefinitions": [
        {
            "name": "aware-agent",
            "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/aware-agent:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "DATABASE_URL",
                    "value": "postgresql://user:password@rds-endpoint:5432/aware_agent"
                },
                {
                    "name": "REDIS_URL",
                    "value": "redis://elasticache-endpoint:6379"
                }
            ]
        }
    ]
}
```

5. **Create Service**
```bash
aws ecs create-service --cluster aware-agent-cluster --service-name aware-agent-service --task-definition aware-agent:1 --desired-count 2 --launch-type FARGATE
```

## Monitoring Setup

1. **Configure CloudWatch**
```bash
aws logs create-log-group --log-group-name /ecs/aware-agent
```

2. **Set Up Alarms**
```bash
aws cloudwatch put-metric-alarm --alarm-name aware-agent-health --metric-name HealthCheckStatus --namespace AWS/ECS --statistic Average --period 300 --threshold 1 --comparison-operator LessThanThreshold --evaluation-periods 2
```

## Backup and Recovery

1. **Database Backups**
```bash
# Automated backup
aws rds create-db-snapshot --db-instance-identifier aware-agent-db --db-snapshot-identifier aware-agent-backup-$(date +%Y%m%d)
```

2. **Restore from Backup**
```bash
aws rds restore-db-instance-from-db-snapshot --db-instance-identifier aware-agent-db-new --db-snapshot-identifier aware-agent-backup-20240101
```

## Scaling

1. **Auto Scaling Configuration**
```bash
aws application-autoscaling register-scalable-target --service-namespace ecs --resource-id service/aware-agent-cluster/aware-agent-service --scalable-dimension ecs:service:DesiredCount --min-capacity 2 --max-capacity 10
```

2. **Scaling Policies**
```bash
aws application-autoscaling put-scaling-policy --policy-name aware-agent-scaling --service-namespace ecs --resource-id service/aware-agent-cluster/aware-agent-service --scalable-dimension ecs:service:DesiredCount --policy-type TargetTrackingScaling --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

## Troubleshooting

1. **Check Logs**
```bash
aws logs get-log-events --log-group-name /ecs/aware-agent --log-stream-name aware-agent/$(aws logs describe-log-streams --log-group-name /ecs/aware-agent --query 'logStreams[0].logStreamName' --output text)
```

2. **Service Status**
```bash
aws ecs describe-services --cluster aware-agent-cluster --services aware-agent-service
```

3. **Task Status**
```bash
aws ecs describe-tasks --cluster aware-agent-cluster --tasks $(aws ecs list-tasks --cluster aware-agent-cluster --service-name aware-agent-service --query 'taskArns[0]' --output text)
```

## Security

1. **SSL/TLS Configuration**
```bash
aws acm request-certificate --domain-name api.aware-agent.com --validation-method DNS
```

2. **Network Security**
```bash
aws ec2 create-security-group --group-name aware-agent-sg --description "Security group for Aware Agent"
aws ec2 authorize-security-group-ingress --group-name aware-agent-sg --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

## Maintenance

1. **Update Application**
```bash
# Build and push new image
docker build -t aware-agent:new .
docker tag aware-agent:new <account-id>.dkr.ecr.us-east-1.amazonaws.com/aware-agent:new
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/aware-agent:new

# Update service
aws ecs update-service --cluster aware-agent-cluster --service aware-agent-service --force-new-deployment
```

2. **Database Maintenance**
```bash
# Vacuum database
psql -h rds-endpoint -U user -d aware_agent -c "VACUUM ANALYZE;"
```

3. **Cache Maintenance**
```bash
# Clear Redis cache
redis-cli -h elasticache-endpoint FLUSHALL
``` 