# Monitoring Guide

## Overview

The Aware Agent system includes comprehensive monitoring capabilities through various tools and services. This guide covers monitoring setup, configuration, and best practices.

## Monitoring Components

### 1. System Metrics

#### Health Checks
- **Endpoint**: `GET /health`
- **Frequency**: Every 5 minutes
- **Metrics**:
  - Service status
  - Component health
  - Response time
  - Error rate

#### Performance Metrics
- **Endpoint**: `GET /metrics`
- **Frequency**: Every 1 minute
- **Metrics**:
  - Message processing rate
  - Active connections
  - Cache hit/miss ratio
  - Average response time
  - Memory usage
  - CPU utilization

### 2. Logging

#### Log Levels
- **DEBUG**: Detailed information for debugging
- **INFO**: General operational information
- **WARNING**: Warning conditions
- **ERROR**: Error conditions
- **CRITICAL**: Critical conditions

#### Log Format
```json
{
    "timestamp": "ISO-8601 timestamp",
    "level": "log_level",
    "service": "service_name",
    "message": "log_message",
    "context": {
        "request_id": "uuid",
        "user_id": "user_identifier",
        "agent": "agent_type"
    }
}
```

### 3. Alerts

#### Alert Types
1. **Critical Alerts**
   - Service down
   - Database connection failure
   - High error rate
   - Memory exhaustion

2. **Warning Alerts**
   - High latency
   - Cache miss rate increase
   - Connection pool exhaustion
   - Disk space low

3. **Info Alerts**
   - Service restarts
   - Configuration changes
   - Deployment events

## Setup Instructions

### 1. CloudWatch Configuration

```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/aware-agent

# Create metric filter
aws logs put-metric-filter \
    --log-group-name /ecs/aware-agent \
    --filter-name "ErrorFilter" \
    --filter-pattern "{ $.level = \"ERROR\" }" \
    --metric-transformations \
        metricName=ErrorCount,metricNamespace=AwareAgent,metricValue=1
```

### 2. Alert Configuration

```bash
# Create SNS topic
aws sns create-topic --name aware-agent-alerts

# Create CloudWatch alarm
aws cloudwatch put-metric-alarm \
    --alarm-name aware-agent-error-rate \
    --metric-name ErrorCount \
    --namespace AwareAgent \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:aware-agent-alerts
```

### 3. Dashboard Setup

```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
    --dashboard-name aware-agent-dashboard \
    --dashboard-body file://dashboard.json
```

## Monitoring Tools

### 1. CloudWatch Dashboard

#### Metrics to Monitor
- **System Health**
  - Service uptime
  - Health check status
  - Error rate
  - Response time

- **Performance**
  - Message throughput
  - Active connections
  - Cache performance
  - Database latency

- **Resource Usage**
  - CPU utilization
  - Memory usage
  - Disk space
  - Network traffic

### 2. Prometheus Integration

#### Configuration
```yaml
scrape_configs:
  - job_name: 'aware-agent'
    static_configs:
      - targets: ['localhost:8000']
```

#### Key Metrics
- `aware_agent_messages_total`
- `aware_agent_connections_active`
- `aware_agent_response_time_seconds`
- `aware_agent_cache_hits_total`
- `aware_agent_cache_misses_total`

### 3. Grafana Dashboards

#### Recommended Dashboards
1. **System Overview**
   - Service status
   - Resource usage
   - Error rates
   - Response times

2. **Performance Analysis**
   - Message throughput
   - Cache performance
   - Database metrics
   - Connection stats

3. **Business Metrics**
   - User activity
   - Agent usage
   - Feature adoption
   - Error patterns

## Best Practices

### 1. Alert Management

- Set appropriate thresholds
- Use multiple notification channels
- Implement alert deduplication
- Create runbooks for common issues

### 2. Log Management

- Use structured logging
- Implement log rotation
- Set appropriate log levels
- Include relevant context

### 3. Performance Monitoring

- Monitor key metrics continuously
- Set up baseline measurements
- Track trends over time
- Implement automated scaling

### 4. Security Monitoring

- Monitor authentication attempts
- Track access patterns
- Log security events
- Set up intrusion detection

## Troubleshooting

### 1. Common Issues

#### High Latency
- Check database performance
- Monitor cache hit ratio
- Review connection pool usage
- Analyze message queue length

#### High Error Rate
- Check service dependencies
- Review recent deployments
- Analyze error patterns
- Check resource limits

#### Service Unavailability
- Verify health checks
- Check network connectivity
- Review system logs
- Check resource utilization

### 2. Diagnostic Tools

```bash
# Check service status
curl -f https://api.aware-agent.com/health

# Get metrics
curl https://api.aware-agent.com/metrics

# Check logs
aws logs get-log-events \
    --log-group-name /ecs/aware-agent \
    --log-stream-name aware-agent/$(aws logs describe-log-streams \
        --log-group-name /ecs/aware-agent \
        --query 'logStreams[0].logStreamName' \
        --output text)
```

### 3. Recovery Procedures

1. **Service Restart**
```bash
aws ecs update-service \
    --cluster aware-agent-cluster \
    --service aware-agent-service \
    --force-new-deployment
```

2. **Database Recovery**
```bash
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier aware-agent-db-new \
    --db-snapshot-identifier aware-agent-backup-latest
```

3. **Cache Reset**
```bash
redis-cli -h elasticache-endpoint FLUSHALL
``` 