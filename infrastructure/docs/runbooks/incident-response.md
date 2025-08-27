# Incident Response Runbook
## IPO Valuation Platform

This runbook provides step-by-step procedures for responding to incidents in the IPO Valuation Platform.

## 🚨 Emergency Contacts

### Primary On-Call (24/7)
- **DevOps Team**: +1-555-DEVOPS (ext. 1)
- **Backend Team**: +1-555-BACKEND (ext. 2)
- **Platform Team**: +1-555-PLATFORM (ext. 3)

### Secondary Contacts
- **CTO**: +1-555-CTO-HELP
- **VP Engineering**: +1-555-VP-ENG
- **Security Team**: +1-555-SEC-TEAM

### Escalation Matrix
1. **P0 (Critical)**: Immediate escalation to CTO
2. **P1 (High)**: Escalate to VP Engineering within 30 minutes
3. **P2 (Medium)**: Escalate to Team Lead within 2 hours
4. **P3 (Low)**: Handle within normal business hours

## 📋 Incident Classification

### Severity Levels

#### P0 - Critical (< 15 minutes response)
- Complete service outage
- Data breach or security incident
- Financial data corruption
- Mass user impact (>90% users affected)

#### P1 - High (< 1 hour response)
- Partial service outage
- Performance degradation affecting >50% users
- Payment processing issues
- Third-party integration failures

#### P2 - Medium (< 4 hours response)
- Minor feature outages
- Performance issues affecting <50% users
- Non-critical third-party service issues

#### P3 - Low (< 24 hours response)
- Minor bugs or cosmetic issues
- Documentation updates
- Non-urgent maintenance

## 🔧 Common Incident Scenarios

### 1. Application Down (P0)

#### Symptoms
- Health check endpoints returning 5xx errors
- No response from application
- Monitoring alerts firing

#### Immediate Actions
```bash
# 1. Check application status
kubectl get pods -n uprez-valuation
kubectl describe pod <failing-pod> -n uprez-valuation

# 2. Check recent deployments
kubectl rollout history deployment/backend-deployment -n uprez-valuation
kubectl rollout history deployment/frontend-deployment -n uprez-valuation

# 3. Check logs
kubectl logs -f deployment/backend-deployment -n uprez-valuation --tail=100
kubectl logs -f deployment/frontend-deployment -n uprez-valuation --tail=100

# 4. If recent deployment caused issue, rollback
kubectl rollout undo deployment/backend-deployment -n uprez-valuation
kubectl rollout undo deployment/frontend-deployment -n uprez-valuation

# 5. Monitor rollback
kubectl rollout status deployment/backend-deployment -n uprez-valuation
```

#### Investigation Steps
1. Check Grafana dashboards for anomalies
2. Review application logs for errors
3. Check database connectivity
4. Verify external service dependencies
5. Review recent changes in Git

#### Communication Template
```
🚨 **INCIDENT ALERT - P0**
Service: IPO Valuation Platform
Status: Investigating
Impact: Complete service outage
ETA: Under investigation
Next Update: 15 minutes

Actions Taken:
- Incident declared at [TIME]
- War room established
- Initial investigation started
- Rollback initiated (if applicable)
```

### 2. High Error Rate (P1)

#### Symptoms
- Error rate >5% in monitoring dashboards
- Users reporting application errors
- Increased 5xx responses

#### Immediate Actions
```bash
# 1. Check error rates
curl -s "http://prometheus:9090/api/v1/query?query=rate(http_requests_total{status=~'5..'}[5m])"

# 2. Identify error patterns
kubectl logs -f deployment/backend-deployment -n uprez-valuation | grep ERROR

# 3. Check database status
kubectl exec -it deployment/backend-deployment -n uprez-valuation -- \
  psql $DATABASE_URL -c "SELECT version();"

# 4. Check external services
curl -f https://api.alpha-vantage.com/status
curl -f https://finance.yahoo.com/status

# 5. Scale up if needed
kubectl scale deployment backend-deployment --replicas=6 -n uprez-valuation
```

### 3. Database Issues (P0/P1)

#### Symptoms
- Database connection timeouts
- Slow query performance
- Database unavailable errors

#### Immediate Actions
```bash
# 1. Check database status
gcloud sql instances describe uprez-valuation-db-prod

# 2. Check connections
gcloud sql operations list --instance=uprez-valuation-db-prod --limit=10

# 3. Monitor slow queries
# Connect to database and run:
# SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;

# 4. Check replica lag (if applicable)
gcloud sql instances describe uprez-valuation-db-prod-replica

# 5. Scale read replicas if needed
gcloud sql instances patch uprez-valuation-db-prod --replica-count=3
```

### 4. Performance Degradation (P1/P2)

#### Symptoms
- Response times >2 seconds
- High CPU/Memory usage
- User complaints about slow loading

#### Immediate Actions
```bash
# 1. Check application metrics
kubectl top pods -n uprez-valuation

# 2. Check HPA status
kubectl get hpa -n uprez-valuation

# 3. Manual scaling if HPA not responding
kubectl scale deployment backend-deployment --replicas=10 -n uprez-valuation

# 4. Check database performance
# Run slow query analysis

# 5. Check cache hit rate
redis-cli info stats | grep keyspace
```

### 5. Security Incident (P0)

#### Symptoms
- Unauthorized access alerts
- DDoS attack indicators
- Data breach notifications

#### Immediate Actions
```bash
# 1. Enable enhanced monitoring
kubectl apply -f infrastructure/kubernetes/security/incident-monitoring.yaml

# 2. Check failed authentication attempts
kubectl logs -f deployment/backend-deployment -n uprez-valuation | grep "401\|403"

# 3. Review access logs
gcloud logging read 'resource.type="gce_instance" AND severity>=WARNING' --limit=100

# 4. Implement emergency measures
# Block suspicious IPs via Cloud Armor
gcloud compute security-policies rules create 1000 \
  --security-policy uprez-valuation-security-policy \
  --expression "origin.ip == 'SUSPICIOUS_IP'" \
  --action "deny-403"

# 5. Notify security team immediately
```

## 📊 Monitoring & Dashboards

### Key URLs
- **Grafana**: https://grafana.uprez-valuation.com
- **Prometheus**: https://prometheus.uprez-valuation.com
- **Kibana**: https://kibana.uprez-valuation.com
- **GCP Console**: https://console.cloud.google.com
- **GitHub Actions**: https://github.com/uprez/uprez-valuation/actions

### Critical Dashboards
1. **Application Overview**: https://grafana.uprez-valuation.com/d/app-overview
2. **Infrastructure Health**: https://grafana.uprez-valuation.com/d/infra-health
3. **Database Performance**: https://grafana.uprez-valuation.com/d/db-perf
4. **Security Monitoring**: https://grafana.uprez-valuation.com/d/security

### Key Metrics to Monitor
- **Availability**: `up{job="uprez-backend"}`
- **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
- **Response Time**: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`
- **CPU Usage**: `rate(container_cpu_usage_seconds_total[5m])`
- **Memory Usage**: `container_memory_usage_bytes`

## 🔧 Useful Commands

### Kubernetes Debugging
```bash
# Get all resources
kubectl get all -n uprez-valuation

# Describe problematic pod
kubectl describe pod <pod-name> -n uprez-valuation

# Get logs with previous instance
kubectl logs <pod-name> -n uprez-valuation --previous

# Port forward for debugging
kubectl port-forward deployment/backend-deployment 8080:8000 -n uprez-valuation

# Execute commands in pod
kubectl exec -it <pod-name> -n uprez-valuation -- /bin/bash

# Check events
kubectl get events -n uprez-valuation --sort-by=.metadata.creationTimestamp
```

### Database Debugging
```bash
# Connect to database
kubectl exec -it deployment/backend-deployment -n uprez-valuation -- \
  psql $DATABASE_URL

# Check database size
SELECT pg_size_pretty(pg_database_size('uprez_valuation'));

# Check active connections
SELECT count(*) FROM pg_stat_activity;

# Check slow queries
SELECT query, mean_time, calls, rows, 100.0 * shared_blks_hit / 
nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 5;

# Check locks
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement,
       blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

### GCP Debugging
```bash
# Check cluster status
gcloud container clusters describe uprez-valuation-prod-cluster --region=us-central1

# Check Cloud SQL
gcloud sql instances describe uprez-valuation-db-prod

# Check load balancer
gcloud compute backend-services describe backend-service-name --global

# Check logs
gcloud logging read 'resource.type="k8s_container" 
resource.labels.namespace_name="uprez-valuation"' --limit=50

# Check metrics
gcloud monitoring metrics list --filter="metric.type:compute"
```

## 📝 Post-Incident Actions

### Immediate (Within 2 hours)
1. **Service Recovery**: Ensure full service restoration
2. **Initial Communication**: Update stakeholders on resolution
3. **Preserve Evidence**: Save logs, metrics, and configurations
4. **Document Timeline**: Record all actions taken during incident

### Short-term (Within 24 hours)
1. **Incident Report**: Create detailed incident report
2. **Root Cause Analysis**: Identify primary and contributing causes
3. **Immediate Fixes**: Implement quick fixes to prevent recurrence
4. **Customer Communication**: Send post-incident communication

### Long-term (Within 1 week)
1. **Post-Mortem Meeting**: Conduct blameless post-mortem
2. **Action Items**: Create and assign improvement tasks
3. **Process Updates**: Update runbooks and procedures
4. **Preventive Measures**: Implement monitoring and alerting improvements

## 📋 Incident Report Template

```markdown
# Incident Report: [INCIDENT_ID]

## Summary
- **Date**: [DATE]
- **Duration**: [START_TIME] - [END_TIME] ([DURATION])
- **Severity**: [P0/P1/P2/P3]
- **Status**: Resolved
- **Incident Commander**: [NAME]

## Impact
- **Users Affected**: [NUMBER/PERCENTAGE]
- **Services Affected**: [LIST]
- **Business Impact**: [DESCRIPTION]

## Root Cause
[DETAILED DESCRIPTION OF ROOT CAUSE]

## Timeline
- [TIME]: Issue first detected
- [TIME]: Incident declared
- [TIME]: War room established
- [TIME]: Root cause identified
- [TIME]: Fix applied
- [TIME]: Service restored
- [TIME]: Incident resolved

## Resolution
[DESCRIPTION OF HOW THE ISSUE WAS RESOLVED]

## Action Items
1. [ ] [ACTION ITEM 1] - Owner: [NAME] - Due: [DATE]
2. [ ] [ACTION ITEM 2] - Owner: [NAME] - Due: [DATE]

## Lessons Learned
- **What Went Well**: [LIST]
- **What Could Be Improved**: [LIST]
- **Preventive Measures**: [LIST]
```

## 🔄 Continuous Improvement

### Monthly Review
- Review all incidents from the past month
- Identify recurring patterns
- Update runbooks based on lessons learned
- Conduct tabletop exercises

### Quarterly Assessment
- Evaluate incident response effectiveness
- Update emergency contacts and escalation procedures
- Review and update monitoring and alerting
- Conduct disaster recovery drills

### Annual Planning
- Comprehensive review of incident response procedures
- Update business continuity plans
- Conduct major disaster recovery exercises
- Review and update SLAs/SLOs

---

**Remember**: Stay calm, communicate clearly, and focus on resolution first, investigation second.