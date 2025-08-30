# GCP AI/ML Services Deployment Guide

## Overview

This comprehensive deployment guide provides step-by-step instructions for deploying the complete Google Cloud AI/ML integration for the IPO valuation platform, including infrastructure setup, service configuration, and production deployment.

## Prerequisites

### Required Tools

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud --version

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
terraform --version

# Install Python dependencies
pip install -r requirements.txt

# Install kubectl (if using GKE)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
```

### GCP Project Setup

```bash
# Set up authentication
gcloud auth login
gcloud auth application-default login

# Create project (if new)
gcloud projects create uprez-valuation-prod --name="Uprez IPO Valuation Platform"

# Set project and region
export PROJECT_ID="uprez-valuation-prod"
export REGION="australia-southeast1"
export ZONE="australia-southeast1-a"

gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

# Enable billing (replace with your billing account)
gcloud billing projects link $PROJECT_ID --billing-account=YOUR-BILLING-ACCOUNT-ID
```

## Phase 1: Infrastructure Deployment with Terraform

### 1.1 Initialize Terraform Backend

```bash
# Create Terraform state bucket
gsutil mb gs://${PROJECT_ID}-terraform-state
gsutil versioning set on gs://${PROJECT_ID}-terraform-state

# Navigate to terraform directory
cd docs/dev-notes/gcp-integration/terraform/

# Initialize Terraform
terraform init
```

### 1.2 Configure Variables

```bash
# Copy example variables file
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your values
nano terraform.tfvars
```

Required variables to update:
```hcl
project_id = "your-actual-project-id"
alert_email = "your-email@company.com"
slack_webhook_url = "your-slack-webhook-url"
github_owner = "your-github-org"
github_repo = "your-repo-name"
```

### 1.3 Plan and Deploy Infrastructure

```bash
# Validate configuration
terraform validate

# Plan deployment
terraform plan -out=tfplan

# Apply infrastructure
terraform apply tfplan
```

This will create:
- Service accounts with appropriate IAM roles
- Cloud Storage buckets for ML artifacts
- BigQuery datasets for data processing
- Vertex AI Feature Store
- Pub/Sub topics for event processing
- Monitoring and alerting infrastructure
- Security and networking components

### 1.4 Verify Infrastructure Deployment

```bash
# Check service accounts
gcloud iam service-accounts list --filter="email:uprez-*"

# Check storage buckets
gsutil ls -p $PROJECT_ID

# Check BigQuery datasets
bq ls --project_id $PROJECT_ID

# Check Vertex AI Feature Store
gcloud ai feature-stores list --region=$REGION
```

## Phase 2: Service Account Setup and Authentication

### 2.1 Create Service Account Keys (Development Only)

```bash
# For development environments only - use Workload Identity in production
export SA_NAME="uprez-production-ml-training"

gcloud iam service-accounts keys create ${SA_NAME}-key.json \
  --iam-account=${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/${SA_NAME}-key.json"
```

### 2.2 Verify Service Account Permissions

```python
# Test service account authentication
python3 << EOF
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import documentai

project_id = "${PROJECT_ID}"
region = "${REGION}"

# Test Vertex AI access
try:
    aiplatform.init(project=project_id, location=region)
    models = aiplatform.Model.list()
    print(f"✓ Vertex AI access verified. Found {len(models)} models.")
except Exception as e:
    print(f"✗ Vertex AI access failed: {e}")

# Test BigQuery access
try:
    client = bigquery.Client(project=project_id)
    datasets = list(client.list_datasets())
    print(f"✓ BigQuery access verified. Found {len(datasets)} datasets.")
except Exception as e:
    print(f"✗ BigQuery access failed: {e}")

# Test Document AI access
try:
    client = documentai.DocumentProcessorServiceClient()
    print("✓ Document AI access verified.")
except Exception as e:
    print(f"✗ Document AI access failed: {e}")
EOF
```

## Phase 3: Feature Store Setup

### 3.1 Initialize Feature Store

```python
# Run Feature Store setup
python3 << EOF
from src.gcp.vertex_ai.pipeline_orchestration import VertexAIPipelineOrchestrator

project_id = "${PROJECT_ID}"
region = "${REGION}"

# Initialize orchestrator
orchestrator = VertexAIPipelineOrchestrator(project_id, region)

# Set up complete Feature Store
print("Setting up Feature Store...")
setup_results = orchestrator.setup_complete_feature_store()

if setup_results["setup_status"] == "completed":
    print("✓ Feature Store setup completed successfully")
    print(f"  - Feature Store: {setup_results['components_created']['featurestore']}")
    print(f"  - Entity Types: {len(setup_results['components_created']['entity_types'])}")
    print(f"  - Features: {sum(len(features) for features in setup_results['components_created']['features'].values())}")
else:
    print(f"✗ Feature Store setup failed: {setup_results.get('error', 'Unknown error')}")
EOF
```

### 3.2 Populate Feature Store with Sample Data

```sql
-- Create sample data in BigQuery (run in BigQuery console)
-- Create sample company data
CREATE OR REPLACE TABLE `${PROJECT_ID}.ml_features.sample_companies` AS
SELECT 
    'TCL' as asx_code,
    'TechCorp Limited' as company_name,
    'Technology' as sector,
    'ASX 300' as listing_tier,
    250000000.0 as market_cap,
    50 as employee_count,
    5 as company_age_years,
    'Major' as headquarters_location,
    CURRENT_TIMESTAMP() as feature_timestamp
UNION ALL
SELECT 
    'BHP' as asx_code,
    'BHP Group Limited' as company_name,
    'Resources' as sector,
    'ASX 200' as listing_tier,
    180000000000.0 as market_cap,
    80000 as employee_count,
    137 as company_age_years,
    'Major' as headquarters_location,
    CURRENT_TIMESTAMP() as feature_timestamp;

-- Create sample financial data
CREATE OR REPLACE TABLE `${PROJECT_ID}.ml_features.sample_financial` AS
SELECT 
    'TCL' as asx_code,
    75000000.0 as revenue_current,
    65000000.0 as revenue_previous,
    12000000.0 as net_profit_current,
    8000000.0 as net_profit_previous,
    180000000.0 as total_assets,
    45000000.0 as total_debt,
    135000000.0 as shareholders_equity,
    25000000.0 as cash_and_equivalents,
    0.33 as debt_to_equity_ratio,
    0.089 as return_on_equity,
    0.16 as profit_margin,
    0.154 as revenue_growth_rate,
    CURRENT_TIMESTAMP() as feature_timestamp;
```

## Phase 4: Document AI Processor Setup

### 4.1 Create Custom Document Processors

```bash
# Create Document AI processors
gcloud documentai processors create \
  --location=$REGION \
  --display-name="IPO Prospectus Parser" \
  --type="FORM_PARSER_PROCESSOR"

gcloud documentai processors create \
  --location=$REGION \
  --display-name="Financial Statement Parser" \
  --type="FORM_PARSER_PROCESSOR"

gcloud documentai processors create \
  --location=$REGION \
  --display-name="Annual Report Parser" \
  --type="FORM_PARSER_PROCESSOR"

# List created processors
gcloud documentai processors list --location=$REGION
```

### 4.2 Test Document Processing

```python
# Test document processing
python3 << EOF
from src.gcp.document_ai.processor_config import ProspectusProcessor

project_id = "${PROJECT_ID}"
region = "${REGION}"

# Initialize processor
processor = ProspectusProcessor(project_id, region)

# Test with sample text (in production, use actual PDF documents)
sample_text = """
TechCorp Limited (ASX: TCL) Initial Public Offering
Issue Price: A$2.50 - A$3.50 per share
Shares Offered: 20,000,000
Listing Date: March 15, 2024
Sector: Technology
Lead Manager: Goldman Sachs Australia
"""

print("Testing document processing...")
try:
    # Note: This is a simplified test with text instead of actual PDF
    # In production, you would use actual PDF documents
    entities = processor.extract_ipo_data_from_text(sample_text)
    print("✓ Document processing test successful")
    print(f"  - Extracted entities: {len(entities)}")
except Exception as e:
    print(f"✗ Document processing test failed: {e}")
EOF
```

## Phase 5: BigQuery ML Model Creation

### 5.1 Create Time Series Forecasting Models

```sql
-- Run in BigQuery console
-- Create sample market data
CREATE OR REPLACE TABLE `${PROJECT_ID}.market_data.sample_daily_prices` AS
WITH dates AS (
  SELECT date
  FROM UNNEST(GENERATE_DATE_ARRAY('2020-01-01', CURRENT_DATE(), INTERVAL 1 DAY)) AS date
),
sample_data AS (
  SELECT 
    date,
    'ASX200' as index_name,
    7000 + (RAND() - 0.5) * 1000 as index_value,
    0.15 + (RAND() - 0.5) * 0.1 as volatility_30d
  FROM dates
  WHERE EXTRACT(DAYOFWEEK FROM date) NOT IN (1, 7) -- Exclude weekends
)
SELECT * FROM sample_data;

-- Create ARIMA model for market forecasting
CREATE OR REPLACE MODEL `${PROJECT_ID}.ml_models.asx_market_forecast`
OPTIONS(
  MODEL_TYPE='ARIMA_PLUS',
  TIME_SERIES_TIMESTAMP_COL='date',
  TIME_SERIES_DATA_COL='index_value',
  HORIZON=30,
  AUTO_ARIMA=TRUE,
  DATA_FREQUENCY='DAILY',
  HOLIDAY_REGION='AU'
) AS
SELECT
  date,
  index_value
FROM `${PROJECT_ID}.market_data.sample_daily_prices`
WHERE index_name = 'ASX200'
AND date >= '2022-01-01';

-- Test the model
SELECT *
FROM ML.FORECAST(
  MODEL `${PROJECT_ID}.ml_models.asx_market_forecast`,
  STRUCT(30 as horizon, 0.9 as confidence_level)
)
LIMIT 10;
```

### 5.2 Verify BigQuery ML Models

```bash
# List BigQuery ML models
bq ls --format=pretty ${PROJECT_ID}:ml_models

# Get model information
bq show --format=pretty ${PROJECT_ID}:ml_models.asx_market_forecast
```

## Phase 6: Monitoring and Alerting Setup

### 6.1 Create Custom Metrics and Dashboards

```python
# Set up monitoring infrastructure
python3 << EOF
from src.gcp.monitoring.monitoring_setup import IPOValuationMonitoring

project_id = "${PROJECT_ID}"

# Initialize monitoring
monitoring = IPOValuationMonitoring(project_id)

print("Setting up monitoring infrastructure...")

# Create custom metrics
print("Creating custom metrics...")
metrics_result = monitoring.create_custom_metrics()
print(f"✓ Created {len(metrics_result['created_metrics'])} custom metrics")

# Create notification channels
print("Creating notification channels...")
channels_result = monitoring.create_notification_channels()
print(f"✓ Created {len(channels_result['created_channels'])} notification channels")

# Create alert policies
print("Creating alert policies...")
notification_channels = [ch["resource_name"] for ch in channels_result["created_channels"]]
alerts_result = monitoring.create_alert_policies(notification_channels)
print(f"✓ Created {len(alerts_result['created_policies'])} alert policies")

print("Monitoring setup completed successfully!")
EOF
```

### 6.2 Create Monitoring Dashboards

```python
# Create monitoring dashboards
python3 << EOF
from src.gcp.monitoring.monitoring_setup import DashboardManager

project_id = "${PROJECT_ID}"

# Initialize dashboard manager
dashboard_manager = DashboardManager(project_id)

print("Creating monitoring dashboards...")

# Create ML performance dashboard
ml_dashboard = dashboard_manager.create_ml_performance_dashboard()
if ml_dashboard["status"] == "created":
    print(f"✓ Created ML Performance Dashboard: {ml_dashboard['display_name']}")
else:
    print(f"✗ Failed to create ML dashboard: {ml_dashboard.get('error')}")

# Create business KPI dashboard
business_dashboard = dashboard_manager.create_business_kpi_dashboard()
if business_dashboard["status"] == "created":
    print(f"✓ Created Business KPI Dashboard: {business_dashboard['display_name']}")
else:
    print(f"✗ Failed to create business dashboard: {business_dashboard.get('error')}")

print("Dashboard creation completed!")
EOF
```

## Phase 7: Model Training and Deployment

### 7.1 Train Sample Models

```python
# Train sample models
python3 << EOF
from src.gcp.vertex_ai.automl_config import IPOAutoMLConfig, AutoMLTrainingManager, AutoMLModelType
import pandas as pd

project_id = "${PROJECT_ID}"
region = "${REGION}"

# Initialize AutoML
config = IPOAutoMLConfig(project_id, region)
manager = AutoMLTrainingManager(config)

print("Training sample AutoML model...")

# Create sample training data
sample_data = {
    'market_cap': [150000000, 200000000, 300000000, 180000000, 250000000],
    'revenue': [50000000, 75000000, 120000000, 65000000, 95000000],
    'sector': ['Technology', 'Healthcare', 'Financial Services', 'Resources', 'Technology'],
    'listing_tier': ['ASX 300', 'ASX 200', 'ASX 200', 'ASX 300', 'ASX 200'],
    'final_ipo_valuation': [160000000, 210000000, 320000000, 185000000, 265000000],
    'ml_use': ['TRAIN', 'TRAIN', 'TRAIN', 'VALIDATE', 'VALIDATE']
}

df = pd.DataFrame(sample_data)

# Save to temporary CSV (in production, this would be in GCS)
df.to_csv('/tmp/sample_training_data.csv', index=False)

print("✓ Sample training data created")
print("Note: In production, upload this data to GCS and create proper AutoML datasets")

# For this example, we'll skip actual model training as it requires substantial data
# and takes significant time. In production deployment, you would:
# 1. Upload training data to GCS
# 2. Create AutoML dataset
# 3. Train models
# 4. Deploy to endpoints

print("Training setup completed. Ready for production data and training.")
EOF
```

### 7.2 Deploy Pre-trained Models (if available)

```bash
# If you have pre-trained models, deploy them
# This is a placeholder - replace with actual model URIs

export MODEL_URI="projects/${PROJECT_ID}/locations/${REGION}/models/your-trained-model"
export ENDPOINT_NAME="ipo-valuation-endpoint"

# Create endpoint
gcloud ai endpoints create \
  --display-name=$ENDPOINT_NAME \
  --region=$REGION

# Deploy model to endpoint (replace with actual values)
# gcloud ai endpoints deploy-model ENDPOINT_ID \
#   --region=$REGION \
#   --model=$MODEL_URI \
#   --display-name="ipo-valuation-deployment" \
#   --machine-type="n1-standard-4" \
#   --min-replica-count=1 \
#   --max-replica-count=5

echo "Model deployment commands prepared (update with actual model URIs)"
```

## Phase 8: End-to-End Testing

### 8.1 Run Complete Integration Test

```python
# Run comprehensive integration test
python3 examples/complete-integration-example.py
```

### 8.2 Verify System Health

```python
# Check system health
python3 << EOF
from src.gcp.monitoring.monitoring_setup import MetricWriter

project_id = "${PROJECT_ID}"

# Initialize metric writer
metric_writer = MetricWriter(project_id)

print("Testing metric writing...")

# Write test metrics
try:
    metric_writer.write_model_accuracy_metric(
        model_name="test-model",
        model_version="1.0.0", 
        accuracy=0.95,
        environment="production"
    )
    print("✓ Model accuracy metric written successfully")
except Exception as e:
    print(f"✗ Failed to write model metric: {e}")

try:
    metric_writer.write_business_metric(
        "system_health_check",
        1,
        {"test_type": "deployment_verification"}
    )
    print("✓ Business metric written successfully")
except Exception as e:
    print(f"✗ Failed to write business metric: {e}")

print("Health check completed!")
EOF
```

## Phase 9: Production Configuration

### 9.1 Security Hardening

```bash
# Enable audit logging
gcloud logging sinks create audit-logs-sink \
  storage.googleapis.com/${PROJECT_ID}-audit-logs \
  --log-filter='protoPayload.@type="type.googleapis.com/google.cloud.audit.AuditLog"'

# Set up VPC firewall rules (if using custom VPC)
gcloud compute firewall-rules create allow-ml-services \
  --allow tcp:443,tcp:80 \
  --source-ranges 10.0.0.0/8 \
  --description "Allow ML services communication"

# Enable private Google access (if using private subnets)
gcloud compute networks subnets update default \
  --region=$REGION \
  --enable-private-ip-google-access
```

### 9.2 Backup and Disaster Recovery

```bash
# Set up automated backups
gsutil lifecycle set - gs://${PROJECT_ID}-ml-models << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {"age": 30}
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {"age": 365}
      }
    ]
  }
}
EOF

# Export BigQuery datasets for backup
bq extract \
  --destination_format=AVRO \
  ${PROJECT_ID}:ml_features.sample_companies \
  gs://${PROJECT_ID}-backups/ml_features/companies/companies-$(date +%Y%m%d).avro

echo "Backup configuration completed"
```

### 9.3 Cost Optimization

```bash
# Set up budget alerts
gcloud billing budgets create \
  --billing-account=YOUR-BILLING-ACCOUNT-ID \
  --display-name="IPO Valuation Platform Budget" \
  --budget-amount=5000USD \
  --threshold-percent=50,80,100

# Enable cost optimization recommendations
gcloud recommender recommendations list \
  --project=$PROJECT_ID \
  --recommender=google.compute.instance.MachineTypeRecommender \
  --location=global
```

## Phase 10: Go-Live Checklist

### Pre-Production Checklist

- [ ] All infrastructure deployed successfully via Terraform
- [ ] Service accounts created with minimal required permissions
- [ ] Feature Store populated with production data
- [ ] Document AI processors trained and tested
- [ ] BigQuery ML models trained and validated
- [ ] Monitoring dashboards and alerts configured
- [ ] Security policies and access controls implemented
- [ ] Backup and disaster recovery procedures tested
- [ ] Performance testing completed
- [ ] Cost monitoring and budgets configured

### Production Deployment Commands

```bash
# Final verification before go-live
echo "Running pre-production verification..."

# Check all services are healthy
gcloud services list --enabled --filter="name:aiplatform.googleapis.com OR name:documentai.googleapis.com OR name:bigquery.googleapis.com"

# Verify IAM configuration
gcloud projects get-iam-policy $PROJECT_ID --format="table(bindings.role,bindings.members)"

# Check monitoring
gcloud alpha monitoring policies list

# Verify storage buckets
gsutil ls -L gs://${PROJECT_ID}-*

echo "✓ Pre-production verification completed"
echo "🚀 System ready for production deployment!"
```

### Post-Deployment Monitoring

```bash
# Monitor system metrics for first 24 hours
gcloud logging read 'resource.type="vertex_ai_model" OR resource.type="cloud_function"' \
  --freshness=1h \
  --format="table(timestamp,severity,textPayload)"

# Check error rates
gcloud logging read 'severity="ERROR" AND timestamp>="2024-01-01T00:00:00Z"' \
  --limit=50 \
  --format="table(timestamp,severity,resource.type,textPayload)"
```

## Troubleshooting

### Common Issues and Solutions

1. **Permission Denied Errors**
   ```bash
   # Check service account permissions
   gcloud projects get-iam-policy $PROJECT_ID \
     --flatten="bindings[].members" \
     --filter="bindings.members:serviceAccount:*" \
     --format="table(bindings.role,bindings.members)"
   ```

2. **API Not Enabled Errors**
   ```bash
   # Enable required APIs
   gcloud services enable aiplatform.googleapis.com documentai.googleapis.com
   ```

3. **Quota Exceeded Errors**
   ```bash
   # Check quotas
   gcloud compute regions describe $REGION --format="table(quotas.metric,quotas.limit,quotas.usage)"
   ```

4. **Model Training Failures**
   ```bash
   # Check training job logs
   gcloud ai custom-jobs list --region=$REGION
   gcloud ai custom-jobs describe JOB_ID --region=$REGION
   ```

### Support and Maintenance

- **Monitoring**: Check Cloud Monitoring dashboards daily
- **Logs**: Review error logs in Cloud Logging
- **Updates**: Plan monthly updates for dependencies and models
- **Backup**: Verify backup procedures weekly
- **Security**: Run security audits quarterly

## Conclusion

This deployment guide provides comprehensive instructions for setting up the complete GCP AI/ML integration for the IPO valuation platform. Follow each phase carefully and verify functionality at each step to ensure a successful production deployment.

For ongoing operations and maintenance, refer to the monitoring documentation and establish regular operational procedures for the platform.