# IPO Valuation Platform - DevOps Infrastructure

This directory contains the complete DevOps infrastructure for the IPO Valuation Platform, designed for production deployment on Google Cloud Platform (GCP).

## 🏗️ Architecture Overview

The infrastructure implements a cloud-native, microservices architecture with the following components:

- **Container Orchestration**: Google Kubernetes Engine (GKE)
- **Infrastructure as Code**: Terraform with modular design
- **CI/CD Pipeline**: GitHub Actions with automated testing and deployment
- **Monitoring**: Prometheus + Grafana + Jaeger for comprehensive observability
- **Security**: Multi-layered security with Workload Identity, Network Policies, and Binary Authorization
- **Database**: Cloud SQL PostgreSQL with high availability and automated backups
- **Storage**: Cloud Storage with lifecycle management
- **Networking**: VPC with private subnets, Cloud NAT, and Load Balancers

## 📁 Directory Structure

```
infrastructure/
├── terraform/                    # Infrastructure as Code
│   ├── modules/                 # Reusable Terraform modules
│   │   ├── compute/            # GKE cluster configuration
│   │   ├── networking/         # VPC, subnets, load balancers
│   │   ├── database/           # Cloud SQL setup
│   │   ├── monitoring/         # Cloud Monitoring setup
│   │   ├── security/           # IAM, secrets, security policies
│   │   └── storage/            # Cloud Storage buckets
│   ├── environments/           # Environment-specific configurations
│   │   ├── dev/
│   │   ├── staging/
│   │   └── prod/
│   ├── main.tf                 # Main Terraform configuration
│   ├── variables.tf            # Variable definitions
│   └── outputs.tf              # Output definitions
├── kubernetes/                 # Kubernetes manifests
│   ├── base/                   # Base Kubernetes resources
│   └── overlays/               # Environment-specific overlays
│       ├── dev/
│       ├── staging/
│       └── prod/
├── docker/                     # Docker configurations
│   ├── Dockerfile.backend      # Backend API container
│   ├── Dockerfile.frontend     # Frontend React app container
│   └── docker-compose.yml      # Local development environment
├── helm/                       # Helm charts
│   ├── charts/                 # Custom Helm charts
│   └── values/                 # Environment-specific values
├── monitoring/                 # Monitoring configuration
│   ├── prometheus/             # Prometheus configuration
│   ├── grafana/                # Grafana dashboards
│   └── jaeger/                 # Distributed tracing
├── github-actions/             # CI/CD workflows
│   └── workflows/              # GitHub Actions workflows
├── scripts/                    # Deployment and utility scripts
├── docs/                       # Documentation and runbooks
│   ├── runbooks/              # Operational runbooks
│   └── architecture/          # Architecture documentation
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- **Google Cloud SDK** (gcloud CLI)
- **Terraform** >= 1.0
- **kubectl**
- **Docker**
- **Node.js** >= 18
- **Helm** >= 3.0

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/uprez/uprez-valuation.git
   cd uprez-valuation/infrastructure
   ```

2. **Set up GCP authentication**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Create GCP project and enable APIs**:
   ```bash
   gcloud projects create uprez-valuation-dev --name="Uprez Valuation Dev"
   gcloud config set project uprez-valuation-dev
   
   # Enable required APIs
   gcloud services enable container.googleapis.com
   gcloud services enable compute.googleapis.com
   gcloud services enable sql-component.googleapis.com
   gcloud services enable monitoring.googleapis.com
   gcloud services enable logging.googleapis.com
   ```

4. **Create Terraform state bucket**:
   ```bash
   gsutil mb gs://uprez-valuation-dev-terraform-state
   gsutil versioning set on gs://uprez-valuation-dev-terraform-state
   ```

### Environment Deployment

#### Development Environment

```bash
# Initialize Terraform
cd terraform
terraform init -backend-config="bucket=uprez-valuation-dev-terraform-state"

# Plan deployment
terraform plan -var-file="environments/dev/terraform.tfvars"

# Apply infrastructure
terraform apply -var-file="environments/dev/terraform.tfvars"

# Deploy applications
cd ../scripts
./deploy.sh -e dev -p uprez-valuation-dev
```

#### Staging Environment

```bash
# Deploy to staging
./deploy.sh -e staging -p uprez-valuation-staging
```

#### Production Environment

```bash
# Deploy to production (requires confirmation)
./deploy.sh -e prod -p uprez-valuation-prod

# Or with force flag (use with caution)
./deploy.sh -e prod -p uprez-valuation-prod --force
```

## 🔧 Infrastructure Components

### Terraform Modules

#### Compute Module (GKE)
- **Multi-node pools** with different machine types
- **Workload Identity** for secure pod-to-GCP authentication
- **Network Policies** for micro-segmentation
- **Horizontal Pod Autoscaling** for dynamic scaling
- **Vertical Pod Autoscaling** for resource optimization

#### Networking Module
- **VPC** with public and private subnets
- **Cloud NAT** for egress traffic from private instances
- **Load Balancers** with SSL termination
- **Cloud Armor** for DDoS protection
- **Firewall rules** following least privilege principle

#### Database Module
- **Cloud SQL PostgreSQL** with high availability
- **Automated backups** with point-in-time recovery
- **Read replicas** for performance optimization
- **Connection pooling** and security configurations

#### Security Module
- **IAM roles and policies** with minimal required permissions
- **Service accounts** for applications and services
- **Secret Manager** integration for sensitive data
- **Binary Authorization** for container image security

#### Monitoring Module
- **Cloud Monitoring** integration
- **Custom dashboards** and alerting rules
- **Log-based metrics** for business intelligence
- **Uptime checks** for service availability

### Kubernetes Configuration

#### Base Resources
- **Namespace** with resource quotas and limit ranges
- **ConfigMaps** for application configuration
- **Secrets** for sensitive data (integrated with Secret Manager)
- **Services** with appropriate service types
- **Ingress** with SSL termination and routing rules

#### Application Deployments
- **Backend API** with health checks and resource limits
- **Frontend React app** served by Nginx
- **Admin dashboard** for operational management
- **Database migrations** as Kubernetes jobs

#### Auto-scaling and High Availability
- **Horizontal Pod Autoscaler** based on CPU, memory, and custom metrics
- **Vertical Pod Autoscaler** for resource optimization
- **Pod Disruption Budgets** for maintenance operations
- **Anti-affinity rules** for optimal pod distribution

## 📊 Monitoring and Observability

### Metrics Collection
- **Prometheus** for metrics collection and storage
- **Grafana** for visualization and dashboards
- **Application metrics** instrumented in code
- **Business metrics** for KPI tracking

### Logging
- **Cloud Logging** for centralized log aggregation
- **Structured logging** with consistent formats
- **Log-based alerting** for proactive issue detection
- **Log retention policies** for compliance

### Tracing
- **Jaeger** for distributed tracing
- **Request correlation** across services
- **Performance bottleneck identification**
- **Service dependency mapping**

### Alerting
- **Multi-channel alerting** (email, Slack, PagerDuty)
- **Severity-based escalation** procedures
- **SLO/SLI monitoring** with error budgets
- **Runbooks** linked to alerts

## 🔒 Security

### Network Security
- **Private GKE cluster** with authorized networks
- **Network policies** for pod-to-pod communication
- **VPC firewall rules** with minimal required access
- **Cloud Armor** for application-layer security

### Identity and Access Management
- **Workload Identity** for pod-to-GCP authentication
- **Service accounts** with minimal required permissions
- **IAM policies** following principle of least privilege
- **Regular access reviews** and rotations

### Data Protection
- **Encryption at rest** for all data stores
- **Encryption in transit** with TLS 1.2+
- **Secret Manager** for sensitive configuration
- **Database access controls** and audit logging

### Container Security
- **Binary Authorization** for trusted container images
- **Container image scanning** in CI/CD pipeline
- **Security patches** automated deployment
- **Runtime security** monitoring

## 💰 Cost Optimization

### Resource Optimization
- **Right-sizing** based on actual usage metrics
- **Spot instances** for non-critical workloads
- **Scheduled scaling** for predictable traffic patterns
- **Resource quotas** to prevent over-provisioning

### Storage Optimization
- **Lifecycle policies** for automated data archiving
- **Compression** for log data and backups
- **Regional storage** for frequently accessed data
- **Archive storage** for long-term retention

### Monitoring and Alerts
- **Cost monitoring** dashboards
- **Budget alerts** for spend management
- **Resource utilization** tracking
- **Waste detection** automation

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows
- **Multi-stage pipeline** with parallel execution
- **Automated testing** (unit, integration, e2e)
- **Security scanning** for vulnerabilities
- **Blue-green deployments** for zero-downtime

### Deployment Strategies
- **Environment promotion** (dev → staging → prod)
- **Feature flags** for gradual rollouts
- **Canary deployments** for risk mitigation
- **Automatic rollbacks** on failure detection

### Quality Gates
- **Code coverage** requirements (>80%)
- **Security scan** passing requirements
- **Performance benchmarks** validation
- **Manual approval** for production deployments

## 📚 Operations

### Runbooks
- **Incident response** procedures
- **Troubleshooting guides** for common issues
- **Maintenance procedures** for routine operations
- **Disaster recovery** playbooks

### Backup and Recovery
- **Automated database backups** with cross-region replication
- **Application data backups** to Cloud Storage
- **Infrastructure state backups** (Terraform state)
- **Regular recovery testing** and validation

### Capacity Planning
- **Resource utilization** monitoring
- **Growth projections** based on business metrics
- **Scalability testing** for peak loads
- **Performance benchmarking** and optimization

## 🌐 Multi-Environment Support

### Environment Isolation
- **Separate GCP projects** for complete isolation
- **Environment-specific configurations** via Terraform variables
- **Namespace-based separation** within clusters
- **Resource tagging** for cost allocation

### Configuration Management
- **Environment-specific** Terraform variable files
- **Kubernetes overlays** with Kustomize
- **Secret management** per environment
- **Feature flag** configuration

### Promotion Pipeline
- **Automated promotion** from dev to staging
- **Manual approval** for production deployment
- **Environment synchronization** tooling
- **Configuration drift detection**

## 📞 Support and Contacts

### Emergency Contacts
- **DevOps Team**: +1-555-DEVOPS
- **Platform Team**: +1-555-PLATFORM  
- **Security Team**: +1-555-SECURITY
- **On-Call Engineer**: +1-555-ONCALL

### Escalation Matrix
1. **P0 (Critical)**: Immediate escalation to CTO
2. **P1 (High)**: Escalate to VP Engineering within 30 minutes
3. **P2 (Medium)**: Escalate to Team Lead within 2 hours
4. **P3 (Low)**: Handle within normal business hours

### Documentation
- **Architecture Docs**: `/infrastructure/docs/architecture/`
- **Runbooks**: `/infrastructure/docs/runbooks/`
- **API Documentation**: `https://api-docs.uprez-valuation.com`
- **Monitoring Dashboards**: `https://grafana.uprez-valuation.com`

## 🧪 Testing

### Infrastructure Testing
```bash
# Terraform validation
terraform validate
terraform plan -detailed-exitcode

# Kubernetes manifest validation  
kubectl apply --dry-run=client -k kubernetes/overlays/dev/

# Security scanning
trivy fs --security-checks vuln,config .
```

### Application Testing
```bash
# Unit tests
npm test -- --coverage

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e

# Load testing
k6 run tests/load/basic-load-test.js
```

### Disaster Recovery Testing
```bash
# Database backup restoration
./scripts/test-backup-restore.sh

# Cluster failover
./scripts/test-cluster-failover.sh

# Network partition testing
./scripts/test-network-partition.sh
```

## 🔄 Maintenance

### Regular Maintenance Tasks
- **Weekly**: Security patches and updates
- **Monthly**: Performance optimization and cost review
- **Quarterly**: Disaster recovery testing and documentation updates
- **Annually**: Architecture review and technology refresh

### Automated Maintenance
- **Certificate renewal** via cert-manager
- **Database maintenance** windows
- **Security updates** for container images
- **Log rotation** and archiving

### Change Management
- **Change approval** process for production
- **Maintenance windows** scheduling
- **Rollback procedures** for failed changes
- **Change documentation** and tracking

---

## 📈 Getting Started Checklist

- [ ] **Prerequisites installed** (gcloud, terraform, kubectl, docker)
- [ ] **GCP project created** and APIs enabled
- [ ] **Service account** created with necessary permissions
- [ ] **Terraform state bucket** created
- [ ] **Environment variables** configured
- [ ] **Secrets** configured in Secret Manager
- [ ] **DNS domains** configured (if using custom domains)
- [ ] **SSL certificates** requested
- [ ] **Monitoring** configured and tested
- [ ] **Backup procedures** tested
- [ ] **CI/CD pipeline** configured
- [ ] **Team access** configured
- [ ] **Documentation** reviewed and updated

For detailed deployment instructions, see the [Production Readiness Checklist](docs/production-readiness-checklist.md).

For operational procedures, see the [Runbooks](docs/runbooks/) directory.

For troubleshooting, see the [Incident Response Runbook](docs/runbooks/incident-response.md).