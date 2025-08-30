# Authentication and Service Account Configuration Guide

## Overview

This guide provides comprehensive setup instructions for Google Cloud authentication, service accounts, and IAM configurations for the IPO valuation platform's AI/ML services integration.

## Service Account Architecture

### Core Service Accounts

```yaml
# Service account structure
service_accounts:
  # ML/AI Services
  ml-service-account:
    name: "ml-service-account@PROJECT_ID.iam.gserviceaccount.com"
    description: "Service account for ML model training and serving"
    roles:
      - "roles/aiplatform.user"
      - "roles/storage.objectAdmin"
      - "roles/bigquery.dataEditor"
      - "roles/bigquery.jobUser"
    
  # Document processing
  document-processor:
    name: "document-processor@PROJECT_ID.iam.gserviceaccount.com"
    description: "Service account for Document AI processing"
    roles:
      - "roles/documentai.apiUser"
      - "roles/storage.objectViewer"
      - "roles/bigquery.dataEditor"
    
  # Application backend
  backend-service:
    name: "backend-service@PROJECT_ID.iam.gserviceaccount.com"
    description: "Service account for backend application"
    roles:
      - "roles/aiplatform.predictor"
      - "roles/bigquery.dataViewer"
      - "roles/storage.objectViewer"
    
  # CI/CD Pipeline
  cicd-pipeline:
    name: "cicd-pipeline@PROJECT_ID.iam.gserviceaccount.com"
    description: "Service account for CI/CD operations"
    roles:
      - "roles/aiplatform.admin"
      - "roles/storage.admin"
      - "roles/cloudbuild.builds.editor"
```

## Terraform Configuration

### Service Account Setup

```hcl
# terraform/service_accounts.tf
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

# ML Service Account
resource "google_service_account" "ml_service_account" {
  account_id   = "ml-service-account"
  display_name = "ML Service Account"
  description  = "Service account for ML model training and serving"
  project      = var.project_id
}

# Document AI Service Account
resource "google_service_account" "document_processor" {
  account_id   = "document-processor"
  display_name = "Document Processor"
  description  = "Service account for Document AI processing"
  project      = var.project_id
}

# Backend Service Account
resource "google_service_account" "backend_service" {
  account_id   = "backend-service"
  display_name = "Backend Service"
  description  = "Service account for backend application"
  project      = var.project_id
}

# CI/CD Service Account
resource "google_service_account" "cicd_pipeline" {
  account_id   = "cicd-pipeline"
  display_name = "CI/CD Pipeline"
  description  = "Service account for CI/CD operations"
  project      = var.project_id
}

# IAM Bindings for ML Service Account
resource "google_project_iam_member" "ml_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.ml_service_account.email}"
}

resource "google_project_iam_member" "ml_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.ml_service_account.email}"
}

resource "google_project_iam_member" "ml_bigquery_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.ml_service_account.email}"
}

resource "google_project_iam_member" "ml_bigquery_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.ml_service_account.email}"
}

# Custom IAM role for specific ML operations
resource "google_project_iam_custom_role" "ml_custom_role" {
  role_id     = "customMLRole"
  title       = "Custom ML Role"
  description = "Custom role for ML operations"
  project     = var.project_id
  
  permissions = [
    "aiplatform.models.create",
    "aiplatform.models.deploy",
    "aiplatform.models.get",
    "aiplatform.models.list",
    "aiplatform.endpoints.create",
    "aiplatform.endpoints.deploy",
    "aiplatform.endpoints.predict",
    "aiplatform.featurestores.create",
    "aiplatform.featurestores.get"
  ]
}

# Service Account Keys (for development only)
resource "google_service_account_key" "ml_service_key" {
  service_account_id = google_service_account.ml_service_account.name
  public_key_type    = "TYPE_X509_PEM_FILE"
  
  # Only create in development
  count = var.environment == "development" ? 1 : 0
}

# Workload Identity for GKE (production)
resource "google_service_account_iam_member" "workload_identity_ml" {
  service_account_id = google_service_account.ml_service_account.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[ml-namespace/ml-service-account]"
  
  count = var.environment != "development" ? 1 : 0
}
```

### KMS Encryption Setup

```hcl
# terraform/encryption.tf
# KMS Key Ring for ML services
resource "google_kms_key_ring" "ml_key_ring" {
  name     = "ml-encryption-keys"
  location = var.region
  project  = var.project_id
}

# Model encryption key
resource "google_kms_crypto_key" "model_encryption_key" {
  name     = "model-encryption-key"
  key_ring = google_kms_key_ring.ml_key_ring.id
  
  lifecycle {
    prevent_destroy = true
  }
  
  version_template {
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }
}

# Data encryption key
resource "google_kms_crypto_key" "data_encryption_key" {
  name     = "data-encryption-key"
  key_ring = google_kms_key_ring.ml_key_ring.id
  
  lifecycle {
    prevent_destroy = true
  }
}

# Grant access to ML service account
resource "google_kms_crypto_key_iam_member" "ml_encrypt_decrypt" {
  crypto_key_id = google_kms_crypto_key.model_encryption_key.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${google_service_account.ml_service_account.email}"
}
```

## Authentication Implementation

### 1. Application Default Credentials Setup

```python
# auth/gcp_auth_manager.py
import os
import json
from typing import Optional, Dict, Any
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth.exceptions

class GCPAuthManager:
    """Manage GCP authentication for the application"""
    
    def __init__(self, project_id: str, environment: str = "production"):
        self.project_id = project_id
        self.environment = environment
        self.credentials = None
        self.service_account_info = None
    
    def initialize_auth(self, service_account_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize GCP authentication
        
        Args:
            service_account_path: Path to service account key file (development only)
            
        Returns:
            Authentication status and metadata
        """
        
        auth_status = {
            'authenticated': False,
            'auth_method': None,
            'project_id': self.project_id,
            'environment': self.environment
        }
        
        try:
            if self.environment == "development" and service_account_path:
                # Development: Use service account key file
                self.credentials = service_account.Credentials.from_service_account_file(
                    service_account_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                auth_status['auth_method'] = 'service_account_file'
                
                # Load service account info
                with open(service_account_path, 'r') as f:
                    self.service_account_info = json.load(f)
                
            elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                # Environment variable pointing to service account
                self.credentials, project = default()
                auth_status['auth_method'] = 'application_default_credentials'
                auth_status['detected_project'] = project
                
            else:
                # Production: Use workload identity or metadata service
                self.credentials, project = default()
                auth_status['auth_method'] = 'workload_identity_or_metadata'
                auth_status['detected_project'] = project
            
            # Refresh credentials to verify they work
            self.credentials.refresh(Request())
            auth_status['authenticated'] = True
            auth_status['service_account_email'] = getattr(
                self.credentials, 'service_account_email', 'unknown'
            )
            
        except google.auth.exceptions.DefaultCredentialsError as e:
            auth_status['error'] = f"Credentials error: {str(e)}"
        except Exception as e:
            auth_status['error'] = f"Authentication failed: {str(e)}"
        
        return auth_status
    
    def get_authenticated_client(self, service_name: str):
        """Get authenticated client for specific GCP service"""
        
        if not self.credentials:
            raise ValueError("Authentication not initialized")
        
        # Import appropriate client based on service
        if service_name == 'aiplatform':
            from google.cloud import aiplatform
            return aiplatform.gapic.ModelServiceClient(credentials=self.credentials)
        
        elif service_name == 'documentai':
            from google.cloud import documentai
            return documentai.DocumentProcessorServiceClient(credentials=self.credentials)
        
        elif service_name == 'language':
            from google.cloud import language_v1
            return language_v1.LanguageServiceClient(credentials=self.credentials)
        
        elif service_name == 'bigquery':
            from google.cloud import bigquery
            return bigquery.Client(credentials=self.credentials, project=self.project_id)
        
        elif service_name == 'storage':
            from google.cloud import storage
            return storage.Client(credentials=self.credentials, project=self.project_id)
        
        else:
            raise ValueError(f"Unsupported service: {service_name}")
    
    def verify_permissions(self, required_permissions: List[str]) -> Dict[str, bool]:
        """Verify that the service account has required permissions"""
        
        from google.cloud import asset_v1
        
        # This is a simplified check - in production you'd use IAM policy analyzer
        permission_status = {}
        
        for permission in required_permissions:
            # Check if permission is granted (simplified)
            try:
                # Test permission by attempting a relevant operation
                permission_status[permission] = self._test_permission(permission)
            except Exception:
                permission_status[permission] = False
        
        return permission_status
    
    def _test_permission(self, permission: str) -> bool:
        """Test if specific permission is granted"""
        
        # Map permissions to test operations
        permission_tests = {
            'aiplatform.models.list': self._test_aiplatform_access,
            'bigquery.datasets.get': self._test_bigquery_access,
            'storage.buckets.list': self._test_storage_access,
            'documentai.processors.list': self._test_documentai_access
        }
        
        test_func = permission_tests.get(permission)
        if test_func:
            try:
                test_func()
                return True
            except Exception:
                return False
        
        return False
    
    def _test_aiplatform_access(self):
        """Test Vertex AI access"""
        client = self.get_authenticated_client('aiplatform')
        parent = f"projects/{self.project_id}/locations/us-central1"
        request = {"parent": parent}
        list(client.list_models(request=request))
    
    def _test_bigquery_access(self):
        """Test BigQuery access"""
        client = self.get_authenticated_client('bigquery')
        list(client.list_datasets())
    
    def _test_storage_access(self):
        """Test Cloud Storage access"""
        client = self.get_authenticated_client('storage')
        list(client.list_buckets())
    
    def _test_documentai_access(self):
        """Test Document AI access"""
        client = self.get_authenticated_client('documentai')
        parent = f"projects/{self.project_id}/locations/us"
        request = {"parent": parent}
        list(client.list_processors(request=request))
```

### 2. Workload Identity Configuration (GKE)

```yaml
# kubernetes/auth/workload-identity.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-service-account
  namespace: ml-namespace
  annotations:
    iam.gke.io/gcp-service-account: ml-service-account@PROJECT_ID.iam.gserviceaccount.com
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: document-processor
  namespace: document-processing
  annotations:
    iam.gke.io/gcp-service-account: document-processor@PROJECT_ID.iam.gserviceaccount.com
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: backend-service
  namespace: default
  annotations:
    iam.gke.io/gcp-service-account: backend-service@PROJECT_ID.iam.gserviceaccount.com
```

### 3. Application Configuration

```python
# config/auth_config.py
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AuthConfig:
    """Authentication configuration"""
    project_id: str
    region: str
    environment: str
    service_account_path: Optional[str] = None
    use_workload_identity: bool = True
    
    # Service-specific configurations
    vertex_ai_config: Optional[Dict[str, Any]] = None
    document_ai_config: Optional[Dict[str, Any]] = None
    bigquery_config: Optional[Dict[str, Any]] = None

def load_auth_config() -> AuthConfig:
    """Load authentication configuration from environment"""
    
    return AuthConfig(
        project_id=os.getenv('GCP_PROJECT_ID', 'your-project-id'),
        region=os.getenv('GCP_REGION', 'us-central1'),
        environment=os.getenv('ENVIRONMENT', 'production'),
        service_account_path=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        use_workload_identity=os.getenv('USE_WORKLOAD_IDENTITY', 'true').lower() == 'true',
        
        # Service-specific configs
        vertex_ai_config={
            'api_endpoint': f"{os.getenv('GCP_REGION', 'us-central1')}-aiplatform.googleapis.com",
            'default_machine_type': os.getenv('VERTEX_AI_MACHINE_TYPE', 'n1-standard-4'),
            'max_replicas': int(os.getenv('VERTEX_AI_MAX_REPLICAS', '5'))
        },
        
        document_ai_config={
            'location': os.getenv('DOCUMENT_AI_LOCATION', 'us'),
            'processors': {
                'general_ocr': os.getenv('DOC_AI_OCR_PROCESSOR_ID'),
                'form_parser': os.getenv('DOC_AI_FORM_PROCESSOR_ID'),
                'prospectus_parser': os.getenv('DOC_AI_PROSPECTUS_PROCESSOR_ID')
            }
        },
        
        bigquery_config={
            'dataset_id': os.getenv('BIGQUERY_DATASET_ID', 'ipo_valuation'),
            'location': os.getenv('BIGQUERY_LOCATION', 'US'),
            'default_job_config': {
                'use_query_cache': True,
                'use_legacy_sql': False
            }
        }
    )

# Example usage in application
async def initialize_gcp_services() -> Dict[str, Any]:
    """Initialize all GCP services with proper authentication"""
    
    config = load_auth_config()
    auth_manager = GCPAuthManager(config.project_id, config.environment)
    
    # Initialize authentication
    auth_status = auth_manager.initialize_auth(config.service_account_path)
    
    if not auth_status['authenticated']:
        raise Exception(f"Authentication failed: {auth_status.get('error')}")
    
    # Verify required permissions
    required_permissions = [
        'aiplatform.models.list',
        'bigquery.datasets.get',
        'storage.buckets.list',
        'documentai.processors.list'
    ]
    
    permission_status = auth_manager.verify_permissions(required_permissions)
    missing_permissions = [perm for perm, granted in permission_status.items() if not granted]
    
    if missing_permissions:
        raise Exception(f"Missing permissions: {missing_permissions}")
    
    # Initialize service clients
    services = {
        'vertex_ai': auth_manager.get_authenticated_client('aiplatform'),
        'document_ai': auth_manager.get_authenticated_client('documentai'),
        'natural_language': auth_manager.get_authenticated_client('language'),
        'bigquery': auth_manager.get_authenticated_client('bigquery'),
        'storage': auth_manager.get_authenticated_client('storage')
    }
    
    return {
        'auth_status': auth_status,
        'permission_status': permission_status,
        'services': services,
        'config': config
    }
```

## Security Best Practices

### 1. Least Privilege Access

```python
# security/iam_policy_manager.py
from google.cloud import resourcemanager
from typing import Dict, List, Any

class IAMPolicyManager:
    """Manage IAM policies with least privilege principle"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = resourcemanager.ProjectsClient()
    
    def create_minimal_ml_policy(self) -> Dict[str, Any]:
        """Create minimal IAM policy for ML operations"""
        
        # Define minimal permissions for each operation
        ml_permissions = {
            'model_training': [
                'aiplatform.customJobs.create',
                'aiplatform.models.upload',
                'storage.objects.get',
                'storage.objects.create',
                'bigquery.jobs.create',
                'bigquery.tables.get'
            ],
            'model_serving': [
                'aiplatform.endpoints.predict',
                'aiplatform.models.get',
                'monitoring.timeSeries.create'
            ],
            'document_processing': [
                'documentai.documents.process',
                'storage.objects.get',
                'bigquery.tables.create',
                'bigquery.rows.insert'
            ]
        }
        
        return ml_permissions
    
    def audit_service_account_permissions(self, service_account_email: str) -> Dict[str, Any]:
        """Audit permissions for service account"""
        
        # Get current IAM policy
        request = resourcemanager.GetIamPolicyRequest(
            resource=f"projects/{self.project_id}"
        )
        
        policy = self.client.get_iam_policy(request=request)
        
        # Find bindings for the service account
        sa_bindings = []
        for binding in policy.bindings:
            if f"serviceAccount:{service_account_email}" in binding.members:
                sa_bindings.append({
                    'role': binding.role,
                    'condition': binding.condition.expression if binding.condition else None
                })
        
        return {
            'service_account': service_account_email,
            'bindings': sa_bindings,
            'total_roles': len(sa_bindings),
            'audit_timestamp': datetime.utcnow().isoformat()
        }
```

### 2. Secrets Management

```python
# security/secrets_manager.py
from google.cloud import secretmanager
from typing import Dict, Any, Optional
import json

class SecretsManager:
    """Manage secrets using Google Secret Manager"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()
    
    async def store_api_keys(self, api_keys: Dict[str, str]) -> Dict[str, str]:
        """Store API keys in Secret Manager"""
        
        secret_versions = {}
        
        for key_name, key_value in api_keys.items():
            # Create secret
            parent = f"projects/{self.project_id}"
            secret_id = f"api-key-{key_name.replace('_', '-')}"
            
            try:
                # Try to create secret
                secret = self.client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}}
                    }
                )
            except Exception:
                # Secret might already exist
                secret = self.client.get_secret(
                    request={"name": f"{parent}/secrets/{secret_id}"}
                )
            
            # Add secret version
            version = self.client.add_secret_version(
                request={
                    "parent": secret.name,
                    "payload": {"data": key_value.encode("UTF-8")}
                }
            )
            
            secret_versions[key_name] = version.name
        
        return secret_versions
    
    async def get_secret(self, secret_name: str) -> str:
        """Retrieve secret value"""
        
        name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
        
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    
    async def rotate_secrets(self, secret_names: List[str]) -> Dict[str, str]:
        """Rotate secrets (placeholder for implementation)"""
        
        rotation_results = {}
        
        for secret_name in secret_names:
            # In practice, this would:
            # 1. Generate new secret value
            # 2. Update external services
            # 3. Add new version to Secret Manager
            # 4. Update applications to use new version
            # 5. Disable old version after grace period
            
            rotation_results[secret_name] = "rotation_scheduled"
        
        return rotation_results
```

## Environment-Specific Configurations

### 1. Development Environment

```bash
#!/bin/bash
# scripts/setup_dev_auth.sh

# Development environment authentication setup

PROJECT_ID="your-project-dev"
REGION="us-central1"

echo "Setting up development authentication..."

# Create service account for development
gcloud iam service-accounts create ml-dev-service-account \
    --description="Development ML service account" \
    --display-name="ML Dev Service Account" \
    --project=$PROJECT_ID

# Grant necessary roles
ROLES=(
    "roles/aiplatform.user"
    "roles/storage.objectAdmin"
    "roles/bigquery.dataEditor"
    "roles/bigquery.jobUser"
    "roles/documentai.apiUser"
    "roles/secretmanager.secretAccessor"
)

for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:ml-dev-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="$role"
done

# Create and download key file
gcloud iam service-accounts keys create ./keys/ml-dev-service-account.json \
    --iam-account=ml-dev-service-account@$PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="./keys/ml-dev-service-account.json"

echo "Development authentication setup complete!"
echo "Service account: ml-dev-service-account@$PROJECT_ID.iam.gserviceaccount.com"
echo "Key file: ./keys/ml-dev-service-account.json"
```

### 2. Production Environment

```yaml
# kubernetes/production/auth-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gcp-auth-config
  namespace: production
data:
  project_id: "your-project-prod"
  region: "us-central1"
  environment: "production"
  use_workload_identity: "true"
  
  # Service configurations
  vertex_ai_endpoint: "us-central1-aiplatform.googleapis.com"
  document_ai_location: "us"
  bigquery_dataset: "ipo_valuation_prod"
  
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: production
spec:
  template:
    spec:
      serviceAccountName: ml-service-account
      containers:
      - name: ml-service
        image: gcr.io/PROJECT_ID/ml-service:latest
        env:
        - name: GCP_PROJECT_ID
          valueFrom:
            configMapKeyRef:
              name: gcp-auth-config
              key: project_id
        - name: GCP_REGION
          valueFrom:
            configMapKeyRef:
              name: gcp-auth-config
              key: region
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: gcp-auth-config
              key: environment
```

## Testing Authentication

### Authentication Test Suite

```python
# tests/test_authentication.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from auth.gcp_auth_manager import GCPAuthManager

class TestGCPAuthentication:
    """Test suite for GCP authentication"""
    
    @pytest.fixture
    def auth_manager(self):
        return GCPAuthManager("test-project", "development")
    
    @pytest.mark.asyncio
    async def test_service_account_auth(self, auth_manager):
        """Test service account authentication"""
        
        with patch('google.oauth2.service_account.Credentials.from_service_account_file') as mock_creds:
            mock_creds.return_value = Mock()
            
            auth_status = auth_manager.initialize_auth("./test-key.json")
            
            assert auth_status['authenticated'] == True
            assert auth_status['auth_method'] == 'service_account_file'
    
    @pytest.mark.asyncio
    async def test_workload_identity_auth(self, auth_manager):
        """Test workload identity authentication"""
        
        with patch('google.auth.default') as mock_default:
            mock_creds = Mock()
            mock_default.return_value = (mock_creds, "test-project")
            
            auth_status = auth_manager.initialize_auth()
            
            assert auth_status['authenticated'] == True
            assert 'workload_identity' in auth_status['auth_method']
    
    @pytest.mark.asyncio
    async def test_permission_verification(self, auth_manager):
        """Test permission verification"""
        
        # Mock successful authentication
        auth_manager.credentials = Mock()
        
        with patch.object(auth_manager, '_test_permission', return_value=True):
            permissions = auth_manager.verify_permissions(['aiplatform.models.list'])
            
            assert permissions['aiplatform.models.list'] == True
    
    @pytest.mark.asyncio
    async def test_client_creation(self, auth_manager):
        """Test authenticated client creation"""
        
        auth_manager.credentials = Mock()
        
        with patch('google.cloud.aiplatform.gapic.ModelServiceClient') as mock_client:
            client = auth_manager.get_authenticated_client('aiplatform')
            
            mock_client.assert_called_once_with(credentials=auth_manager.credentials)

# Integration test
@pytest.mark.integration
async def test_end_to_end_auth_flow():
    """Test complete authentication flow"""
    
    # This test requires actual GCP credentials
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        pytest.skip("No GCP credentials available for integration test")
    
    config = load_auth_config()
    services = await initialize_gcp_services()
    
    assert services['auth_status']['authenticated'] == True
    assert 'vertex_ai' in services['services']
    assert 'bigquery' in services['services']
```

## Monitoring and Auditing

### 1. Authentication Audit Logger

```python
# monitoring/auth_audit_logger.py
from google.cloud import logging
import json
from typing import Dict, Any

class AuthAuditLogger:
    """Log authentication events for audit trails"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.logging_client = logging.Client()
        self.logger = self.logging_client.logger("auth-audit")
    
    def log_auth_event(
        self,
        event_type: str,
        service_account: str,
        operation: str,
        success: bool,
        metadata: Dict[str, Any]
    ):
        """Log authentication event"""
        
        log_entry = {
            'event_type': event_type,
            'service_account': service_account,
            'operation': operation,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata,
            'severity': 'INFO' if success else 'WARNING'
        }
        
        self.logger.log_struct(
            log_entry,
            severity=log_entry['severity']
        )
    
    def log_permission_check(
        self,
        service_account: str,
        permission: str,
        granted: bool,
        resource: str
    ):
        """Log permission check result"""
        
        self.log_auth_event(
            event_type='permission_check',
            service_account=service_account,
            operation=f"check_{permission}",
            success=granted,
            metadata={
                'permission': permission,
                'resource': resource,
                'granted': granted
            }
        )
```

### 2. Security Monitoring

```python
# security/security_monitor.py
from google.cloud import securitycenter
from typing import Dict, List, Any

class SecurityMonitor:
    """Monitor security events and compliance"""
    
    def __init__(self, organization_id: str, project_id: str):
        self.organization_id = organization_id
        self.project_id = project_id
        self.client = securitycenter.SecurityCenterClient()
    
    async def check_service_account_security(
        self,
        service_account_email: str
    ) -> Dict[str, Any]:
        """Check service account security posture"""
        
        security_findings = {
            'service_account': service_account_email,
            'security_score': 0,
            'findings': [],
            'recommendations': []
        }
        
        # Check key rotation status
        key_age = await self._check_key_age(service_account_email)
        if key_age > 90:  # Days
            security_findings['findings'].append({
                'type': 'stale_key',
                'severity': 'HIGH',
                'description': f'Service account key is {key_age} days old'
            })
            security_findings['recommendations'].append('Rotate service account key')
        
        # Check for overprivileged access
        excessive_permissions = await self._check_excessive_permissions(service_account_email)
        if excessive_permissions:
            security_findings['findings'].append({
                'type': 'excessive_permissions',
                'severity': 'MEDIUM',
                'description': f'Service account has {len(excessive_permissions)} excessive permissions'
            })
            security_findings['recommendations'].append('Remove unnecessary permissions')
        
        # Calculate security score
        total_checks = 10
        failed_checks = len(security_findings['findings'])
        security_findings['security_score'] = max(0, (total_checks - failed_checks) / total_checks * 100)
        
        return security_findings
```

## Deployment Scripts

### 1. Environment Setup Script

```bash
#!/bin/bash
# scripts/setup_gcp_auth.sh

set -e

PROJECT_ID=$1
ENVIRONMENT=$2
REGION=${3:-"us-central1"}

if [ -z "$PROJECT_ID" ] || [ -z "$ENVIRONMENT" ]; then
    echo "Usage: $0 <project_id> <environment> [region]"
    exit 1
fi

echo "Setting up GCP authentication for $ENVIRONMENT environment..."

# Enable required APIs
echo "Enabling GCP APIs..."
gcloud services enable aiplatform.googleapis.com \
    documentai.googleapis.com \
    language.googleapis.com \
    bigquery.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    monitoring.googleapis.com \
    --project=$PROJECT_ID

# Create service accounts
echo "Creating service accounts..."

# ML Service Account
gcloud iam service-accounts create ml-service-account-$ENVIRONMENT \
    --description="ML service account for $ENVIRONMENT" \
    --display-name="ML Service Account ($ENVIRONMENT)" \
    --project=$PROJECT_ID

# Document AI Service Account
gcloud iam service-accounts create document-processor-$ENVIRONMENT \
    --description="Document AI service account for $ENVIRONMENT" \
    --display-name="Document Processor ($ENVIRONMENT)" \
    --project=$PROJECT_ID

# Backend Service Account
gcloud iam service-accounts create backend-service-$ENVIRONMENT \
    --description="Backend service account for $ENVIRONMENT" \
    --display-name="Backend Service ($ENVIRONMENT)" \
    --project=$PROJECT_ID

# Grant roles
echo "Granting IAM roles..."

# ML Service Account roles
ML_SA="ml-service-account-$ENVIRONMENT@$PROJECT_ID.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$ML_SA" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$ML_SA" \
    --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$ML_SA" \
    --role="roles/bigquery.dataEditor"

# Document AI Service Account roles
DOC_SA="document-processor-$ENVIRONMENT@$PROJECT_ID.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$DOC_SA" \
    --role="roles/documentai.apiUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$DOC_SA" \
    --role="roles/storage.objectViewer"

# Backend Service Account roles
BACKEND_SA="backend-service-$ENVIRONMENT@$PROJECT_ID.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$BACKEND_SA" \
    --role="roles/aiplatform.predictor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$BACKEND_SA" \
    --role="roles/bigquery.dataViewer"

# Setup KMS encryption
echo "Setting up KMS encryption..."
gcloud kms keyrings create ml-encryption-keys \
    --location=$REGION \
    --project=$PROJECT_ID

gcloud kms keys create model-encryption-key \
    --location=$REGION \
    --keyring=ml-encryption-keys \
    --purpose=encryption \
    --project=$PROJECT_ID

# Grant KMS access to ML service account
gcloud kms keys add-iam-policy-binding model-encryption-key \
    --location=$REGION \
    --keyring=ml-encryption-keys \
    --member="serviceAccount:$ML_SA" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" \
    --project=$PROJECT_ID

echo "GCP authentication setup complete for $ENVIRONMENT environment!"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "ML Service Account: $ML_SA"
echo "Document AI Service Account: $DOC_SA"
echo "Backend Service Account: $BACKEND_SA"
```

### 2. CI/CD Authentication

```yaml
# .github/workflows/deploy.yml
name: Deploy ML Services

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
        service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
    
    - name: Verify authentication
      run: |
        gcloud auth list
        gcloud config list project
    
    - name: Deploy ML services
      run: |
        # Deploy Vertex AI models
        python scripts/deploy_models.py --environment=production
        
        # Deploy document processing services
        python scripts/deploy_document_ai.py --environment=production
    
    - name: Run integration tests
      run: |
        python -m pytest tests/integration/test_gcp_services.py
```

## Best Practices Summary

### 1. Security
- Use workload identity in production environments
- Implement least privilege access principles
- Rotate service account keys regularly
- Monitor authentication events and anomalies
- Use Secret Manager for sensitive data

### 2. Environment Management
- Separate service accounts per environment
- Use different projects for dev/staging/prod
- Implement proper resource isolation
- Use infrastructure as code (Terraform)

### 3. Monitoring
- Log all authentication events
- Monitor permission usage
- Set up alerts for failed authentication
- Regular security audits

### 4. Development Workflow
- Use service account keys only for local development
- Implement automated testing for authentication
- Document all required permissions
- Use environment-specific configurations

## Next Steps

1. **Service Account Creation**: Create and configure service accounts
2. **Permission Testing**: Verify all required permissions
3. **Workload Identity Setup**: Configure for production environment
4. **Monitoring Implementation**: Set up authentication monitoring
5. **Security Audit**: Conduct comprehensive security review

## Related Documentation

- [Vertex AI Integration](../vertex-ai/README.md)
- [Document AI Integration](../document-ai/README.md)
- [Cost Optimization](../cost-optimization/README.md)
- [AI Platform Setup](../ai-platform/README.md)