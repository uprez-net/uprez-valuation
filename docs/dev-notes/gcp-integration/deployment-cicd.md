# CI/CD Pipeline Implementation for GCP AI/ML Services

## Overview

This guide provides comprehensive CI/CD pipeline implementation for deploying and managing Google Cloud AI/ML services in the IPO valuation platform. It covers automated testing, deployment strategies, and continuous monitoring.

## Pipeline Architecture

### Core Components
- **Source Control**: GitHub/GitLab integration
- **Build Pipeline**: Cloud Build for containerization
- **Testing Pipeline**: Automated ML model testing
- **Deployment Pipeline**: Multi-environment deployment
- **Monitoring Pipeline**: Post-deployment validation
- **Rollback Strategy**: Automated rollback capabilities

### CI/CD Workflow

```yaml
# .github/workflows/ai-ml-cicd.yml
name: AI/ML Services CI/CD

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/backend/ml_services/**'
      - 'src/backend/gcp_integration/**'
      - 'config/vertex-ai/**'
      - 'config/document-ai/**'
  
  pull_request:
    branches: [main]
    paths:
      - 'src/backend/ml_services/**'
      - 'src/backend/gcp_integration/**'

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  GCP_REGION: us-central1
  ARTIFACT_REGISTRY: us-central1-docker.pkg.dev
  SERVICE_ACCOUNT: ${{ secrets.GCP_SERVICE_ACCOUNT }}

jobs:
  # Data validation and model testing
  test-ml-models:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
    
    - name: Install dependencies
      run: |
        pip install -r src/backend/requirements.txt
        pip install -r tests/requirements-test.txt
    
    - name: Run data validation tests
      run: |
        python -m pytest tests/ml/test_data_validation.py -v
        python -m pytest tests/ml/test_feature_engineering.py -v
    
    - name: Run model unit tests
      run: |
        python -m pytest tests/ml/test_model_training.py -v
        python -m pytest tests/ml/test_model_serving.py -v
    
    - name: Run integration tests
      run: |
        python -m pytest tests/integration/test_vertex_ai.py -v
        python -m pytest tests/integration/test_document_ai.py -v
        python -m pytest tests/integration/test_bigquery_ml.py -v
    
    - name: Generate test coverage report
      run: |
        coverage run -m pytest tests/ml/
        coverage report --include="src/backend/ml_services/*" --fail-under=80

  # Build and test Docker images
  build-ml-services:
    needs: test-ml-models
    runs-on: ubuntu-latest
    
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}
    
    - name: Configure Docker to use gcloud as credential helper
      run: |
        gcloud auth configure-docker ${{ env.ARTIFACT_REGISTRY }}
    
    - name: Build ML service image
      id: build
      run: |
        IMAGE_TAG="${{ env.ARTIFACT_REGISTRY }}/${{ env.PROJECT_ID }}/ml-services/ipo-valuation:${{ github.sha }}"
        
        docker build -f src/backend/Dockerfile \
          --tag $IMAGE_TAG \
          --build-arg BUILD_ENV=production \
          src/backend/
        
        docker push $IMAGE_TAG
        
        echo "digest=$(docker inspect --format='{{index .RepoDigests 0}}' $IMAGE_TAG)" >> $GITHUB_OUTPUT
    
    - name: Security scan
      run: |
        gcloud container images scan ${{ steps.build.outputs.image-tag }} \
          --quiet \
          --format="value(response.scan.analysisKind,response.scan.analysisCompleted)"

  # Deploy to staging environment
  deploy-staging:
    needs: [test-ml-models, build-ml-services]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}
    
    - name: Deploy ML models to staging
      run: |
        python scripts/deploy_ml_models.py \
          --environment=staging \
          --image-digest=${{ needs.build-ml-services.outputs.image-digest }} \
          --project-id=${{ env.PROJECT_ID }}
    
    - name: Run staging validation tests
      run: |
        python -m pytest tests/staging/test_ml_endpoints.py -v
        python -m pytest tests/staging/test_model_performance.py -v
    
    - name: Performance benchmarking
      run: |
        python scripts/benchmark_ml_services.py --environment=staging

  # Deploy to production
  deploy-production:
    needs: [deploy-staging]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
        service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}
    
    - name: Deploy ML models to production
      run: |
        python scripts/deploy_ml_models.py \
          --environment=production \
          --image-digest=${{ needs.build-ml-services.outputs.image-digest }} \
          --project-id=${{ env.PROJECT_ID }} \
          --canary-percentage=10
    
    - name: Run production smoke tests
      run: |
        python -m pytest tests/production/test_ml_smoke.py -v
    
    - name: Full production validation
      run: |
        python scripts/validate_production_deployment.py
    
    - name: Update traffic to 100% after validation
      run: |
        python scripts/update_traffic_allocation.py --percentage=100

  # Post-deployment monitoring
  post-deployment:
    needs: [deploy-production]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Setup monitoring alerts
      run: |
        python scripts/setup_deployment_monitoring.py \
          --deployment-id=${{ github.sha }} \
          --environment=production
    
    - name: Notify teams
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#ml-deployments'
        message: |
          ML services deployed to production
          Commit: ${{ github.sha }}
          Models: Vertex AI, Document AI, BigQuery ML
```

## Cloud Build Configuration

### 1. ML Model Training Pipeline

```yaml
# cloudbuild/ml-training.yaml
steps:
  # Data validation
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'validate-data'
    script: |
      #!/bin/bash
      echo "Validating training data..."
      
      # Check data freshness
      python scripts/validate_training_data.py \
        --dataset=$_DATASET_ID \
        --max-age-days=7
      
      # Validate data schema
      python scripts/validate_data_schema.py \
        --dataset=$_DATASET_ID \
        --schema-file=config/data-schemas/training-schema.json

  # Feature engineering
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'feature-engineering'
    script: |
      #!/bin/bash
      echo "Running feature engineering..."
      
      # Run feature engineering pipeline
      python scripts/run_feature_engineering.py \
        --input-dataset=$_DATASET_ID \
        --output-dataset=$_FEATURE_DATASET_ID \
        --config-file=config/feature-engineering.yaml

  # Model training
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'train-models'
    script: |
      #!/bin/bash
      echo "Training ML models..."
      
      # Train multiple models in parallel
      python scripts/train_automl_models.py \
        --config-file=config/model-training.yaml \
        --dataset=$_FEATURE_DATASET_ID \
        --output-bucket=$_MODEL_BUCKET
      
      # Train custom models
      python scripts/train_custom_models.py \
        --config-file=config/custom-models.yaml \
        --dataset=$_FEATURE_DATASET_ID \
        --output-bucket=$_MODEL_BUCKET

  # Model evaluation
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'evaluate-models'
    script: |
      #!/bin/bash
      echo "Evaluating model performance..."
      
      # Run model evaluation
      python scripts/evaluate_models.py \
        --model-bucket=$_MODEL_BUCKET \
        --test-dataset=$_TEST_DATASET_ID \
        --evaluation-output=$_EVALUATION_BUCKET
      
      # Check if models meet quality criteria
      python scripts/check_model_quality.py \
        --evaluation-results=$_EVALUATION_BUCKET \
        --quality-thresholds=config/quality-thresholds.yaml

  # Model deployment
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'deploy-models'
    script: |
      #!/bin/bash
      echo "Deploying models to Vertex AI..."
      
      # Deploy approved models
      python scripts/deploy_approved_models.py \
        --model-bucket=$_MODEL_BUCKET \
        --environment=$_ENVIRONMENT \
        --endpoint-config=config/endpoints.yaml

substitutions:
  _DATASET_ID: 'ipo_valuation.training_data'
  _FEATURE_DATASET_ID: 'ipo_valuation.engineered_features'
  _TEST_DATASET_ID: 'ipo_valuation.test_data'
  _MODEL_BUCKET: 'gs://ipo-valuation-models'
  _EVALUATION_BUCKET: 'gs://ipo-valuation-evaluations'
  _ENVIRONMENT: 'staging'

options:
  machineType: 'E2_HIGHCPU_8'
  logging: CLOUD_LOGGING_ONLY
  
timeout: '3600s'
```

### 2. Document AI Pipeline

```yaml
# cloudbuild/document-ai.yaml
steps:
  # Test document processors
  - name: 'python:3.9'
    id: 'test-processors'
    script: |
      #!/bin/bash
      pip install -r src/backend/requirements.txt
      
      # Test Document AI processors
      python -m pytest tests/document_ai/ -v
      
      # Validate processor configurations
      python scripts/validate_processor_configs.py \
        --config-dir=config/document-ai/

  # Build document processing service
  - name: 'gcr.io/cloud-builders/docker'
    id: 'build-doc-service'
    args:
      - 'build'
      - '-f'
      - 'src/backend/docker/Dockerfile.document-ai'
      - '-t'
      - '$_ARTIFACT_REGISTRY/$PROJECT_ID/document-ai-service:$BUILD_ID'
      - 'src/backend/'

  # Push image
  - name: 'gcr.io/cloud-builders/docker'
    id: 'push-doc-service'
    args:
      - 'push'
      - '$_ARTIFACT_REGISTRY/$PROJECT_ID/document-ai-service:$BUILD_ID'

  # Deploy to Cloud Run
  - name: 'gcr.io/cloud-builders/gcloud'
    id: 'deploy-doc-service'
    script: |
      #!/bin/bash
      gcloud run deploy document-ai-service \
        --image=$_ARTIFACT_REGISTRY/$PROJECT_ID/document-ai-service:$BUILD_ID \
        --platform=managed \
        --region=$_REGION \
        --service-account=document-processor@$PROJECT_ID.iam.gserviceaccount.com \
        --set-env-vars="PROJECT_ID=$PROJECT_ID,ENVIRONMENT=$_ENVIRONMENT" \
        --max-instances=10 \
        --cpu=2 \
        --memory=4Gi \
        --timeout=3600 \
        --no-allow-unauthenticated

  # Test deployment
  - name: 'python:3.9'
    id: 'test-deployment'
    script: |
      #!/bin/bash
      pip install requests pytest
      
      # Get service URL
      SERVICE_URL=$(gcloud run services describe document-ai-service \
        --platform=managed \
        --region=$_REGION \
        --format="value(status.url)")
      
      # Run integration tests
      python tests/integration/test_document_ai_service.py \
        --service-url=$SERVICE_URL

substitutions:
  _ARTIFACT_REGISTRY: 'us-central1-docker.pkg.dev'
  _REGION: 'us-central1'
  _ENVIRONMENT: 'staging'
```

## Deployment Scripts

### 1. ML Model Deployment Script

```python
# scripts/deploy_ml_models.py
import argparse
import asyncio
import json
from typing import Dict, Any, List
from google.cloud import aiplatform
import yaml

class MLModelDeployer:
    """Deploy ML models to Vertex AI with proper versioning"""
    
    def __init__(self, project_id: str, region: str, environment: str):
        self.project_id = project_id
        self.region = region
        self.environment = environment
        aiplatform.init(project=project_id, location=region)
    
    async def deploy_models_from_config(
        self,
        config_file: str,
        canary_percentage: int = 10
    ) -> Dict[str, Any]:
        """
        Deploy models based on configuration file
        
        Args:
            config_file: Path to deployment configuration
            canary_percentage: Percentage of traffic for canary deployment
            
        Returns:
            Deployment results
        """
        
        # Load deployment configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        deployment_results = {}
        
        # Deploy each model
        for model_config in config.get('models', []):
            model_name = model_config['name']
            
            try:
                result = await self._deploy_single_model(model_config, canary_percentage)
                deployment_results[model_name] = result
                
            except Exception as e:
                deployment_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return deployment_results
    
    async def _deploy_single_model(
        self,
        model_config: Dict[str, Any],
        canary_percentage: int
    ) -> Dict[str, Any]:
        """Deploy a single model with canary strategy"""
        
        model_name = model_config['name']
        model_artifact_uri = model_config['artifact_uri']
        endpoint_name = model_config['endpoint_name']
        
        # Upload model to Vertex AI
        model = aiplatform.Model.upload(
            display_name=f"{model_name}-{self.environment}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            artifact_uri=model_artifact_uri,
            serving_container_image_uri=model_config.get(
                'serving_image',
                "gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-11:latest"
            ),
            description=f"Production model for {model_name} in {self.environment}",
            labels={
                'environment': self.environment,
                'model_type': model_config.get('model_type', 'custom'),
                'version': model_config.get('version', '1.0.0')
            }
        )
        
        # Get or create endpoint
        endpoint = await self._get_or_create_endpoint(endpoint_name)
        
        # Deploy with canary strategy
        if canary_percentage < 100:
            deployment_result = await self._canary_deployment(
                endpoint, model, model_config, canary_percentage
            )
        else:
            deployment_result = await self._full_deployment(
                endpoint, model, model_config
            )
        
        return deployment_result
    
    async def _get_or_create_endpoint(self, endpoint_name: str) -> aiplatform.Endpoint:
        """Get existing endpoint or create new one"""
        
        try:
            # Try to find existing endpoint
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"'
            )
            
            if endpoints:
                return endpoints[0]
            
        except Exception:
            pass
        
        # Create new endpoint
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            description=f"Endpoint for {endpoint_name} in {self.environment}",
            labels={'environment': self.environment},
            network=f"projects/{self.project_id}/global/networks/ml-network" if self.environment == 'production' else None
        )
        
        return endpoint
    
    async def _canary_deployment(
        self,
        endpoint: aiplatform.Endpoint,
        model: aiplatform.Model,
        model_config: Dict[str, Any],
        canary_percentage: int
    ) -> Dict[str, Any]:
        """Deploy model using canary strategy"""
        
        # Get current deployed models
        current_models = endpoint.list_models()
        
        # Deploy new model with canary traffic
        new_deployment = endpoint.deploy(
            model=model,
            deployed_model_display_name=f"{model.display_name}-canary",
            machine_type=model_config.get('machine_type', 'n1-standard-4'),
            min_replica_count=1,
            max_replica_count=model_config.get('max_replicas', 3),
            traffic_percentage=canary_percentage,
            sync=True
        )
        
        # Update existing model traffic
        if current_models:
            for deployed_model in current_models:
                endpoint.update(
                    deployed_models=[
                        {
                            'id': deployed_model.id,
                            'traffic_percentage': 100 - canary_percentage
                        },
                        {
                            'id': new_deployment.id,
                            'traffic_percentage': canary_percentage
                        }
                    ]
                )
        
        return {
            'status': 'canary_deployed',
            'canary_percentage': canary_percentage,
            'deployed_model_id': new_deployment.id,
            'endpoint_id': endpoint.resource_name
        }
    
    async def _full_deployment(
        self,
        endpoint: aiplatform.Endpoint,
        model: aiplatform.Model,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy model with full traffic"""
        
        deployment = endpoint.deploy(
            model=model,
            deployed_model_display_name=f"{model.display_name}-full",
            machine_type=model_config.get('machine_type', 'n1-standard-4'),
            min_replica_count=model_config.get('min_replicas', 1),
            max_replica_count=model_config.get('max_replicas', 5),
            traffic_percentage=100,
            sync=True
        )
        
        return {
            'status': 'fully_deployed',
            'deployed_model_id': deployment.id,
            'endpoint_id': endpoint.resource_name
        }

# Main deployment script
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', required=True, choices=['staging', 'production'])
    parser.add_argument('--config-file', default='config/deployment.yaml')
    parser.add_argument('--canary-percentage', type=int, default=10)
    parser.add_argument('--project-id', required=True)
    parser.add_argument('--region', default='us-central1')
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = MLModelDeployer(args.project_id, args.region, args.environment)
    
    # Deploy models
    results = await deployer.deploy_models_from_config(
        args.config_file,
        args.canary_percentage
    )
    
    # Print results
    print(f"Deployment Results for {args.environment}:")
    for model_name, result in results.items():
        print(f"  {model_name}: {result['status']}")
        if result['status'] == 'failed':
            print(f"    Error: {result['error']}")

if __name__ == '__main__':
    asyncio.run(main())
```

### 2. Automated Testing Framework

```python
# scripts/automated_ml_testing.py
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from google.cloud import aiplatform
import pytest

class MLModelTester:
    """Automated testing framework for ML models"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
    
    async def run_model_validation_suite(
        self,
        model_endpoints: List[str],
        test_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run comprehensive model validation"""
        
        validation_results = {}
        
        for endpoint_name in model_endpoints:
            endpoint = aiplatform.Endpoint(endpoint_name)
            
            # Run different types of tests
            tests = {
                'performance_test': await self._test_model_performance(endpoint, test_config),
                'latency_test': await self._test_prediction_latency(endpoint, test_config),
                'load_test': await self._test_model_load_handling(endpoint, test_config),
                'bias_test': await self._test_model_bias(endpoint, test_config),
                'robustness_test': await self._test_model_robustness(endpoint, test_config)
            }
            
            validation_results[endpoint_name] = tests
        
        return validation_results
    
    async def _test_model_performance(
        self,
        endpoint: aiplatform.Endpoint,
        test_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test model performance against validation data"""
        
        # Load test data
        test_data_path = test_config.get('test_data_path')
        df = pd.read_csv(test_data_path)
        
        # Prepare test instances
        feature_columns = test_config.get('feature_columns', [])
        target_column = test_config.get('target_column')
        
        instances = df[feature_columns].to_dict('records')
        actual_values = df[target_column].values
        
        # Make predictions
        predictions = endpoint.predict(instances=instances)
        predicted_values = np.array([pred['value'] for pred in predictions.predictions])
        
        # Calculate performance metrics
        performance_metrics = {
            'mae': float(np.mean(np.abs(actual_values - predicted_values))),
            'rmse': float(np.sqrt(np.mean((actual_values - predicted_values) ** 2))),
            'r_squared': float(1 - np.sum((actual_values - predicted_values) ** 2) / 
                             np.sum((actual_values - np.mean(actual_values)) ** 2)),
            'mape': float(np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100)
        }
        
        # Check against thresholds
        thresholds = test_config.get('performance_thresholds', {})
        performance_check = {
            'passes_threshold': all(
                performance_metrics.get(metric, 0) <= threshold
                for metric, threshold in thresholds.items()
            ),
            'metrics': performance_metrics,
            'thresholds': thresholds
        }
        
        return performance_check
    
    async def _test_prediction_latency(
        self,
        endpoint: aiplatform.Endpoint,
        test_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test prediction latency"""
        
        # Create sample instances
        sample_instance = test_config.get('sample_instance', {})
        if not sample_instance:
            # Create dummy instance if not provided
            feature_columns = test_config.get('feature_columns', [])
            sample_instance = {col: 1.0 for col in feature_columns}
        
        # Run latency test
        latencies = []
        test_iterations = test_config.get('latency_test_iterations', 100)
        
        for _ in range(test_iterations):
            start_time = asyncio.get_event_loop().time()
            
            try:
                prediction = endpoint.predict(instances=[sample_instance])
                end_time = asyncio.get_event_loop().time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                print(f"Prediction failed: {e}")
        
        if latencies:
            latency_stats = {
                'mean_latency_ms': np.mean(latencies),
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'max_latency_ms': np.max(latencies),
                'success_rate': len(latencies) / test_iterations
            }
            
            # Check latency thresholds
            latency_threshold = test_config.get('max_latency_ms', 1000)
            passes_latency_test = latency_stats['p95_latency_ms'] <= latency_threshold
            
            return {
                'passes_test': passes_latency_test,
                'latency_stats': latency_stats,
                'threshold': latency_threshold
            }
        
        return {'passes_test': False, 'error': 'No successful predictions'}
    
    async def _test_model_load_handling(
        self,
        endpoint: aiplatform.Endpoint,
        test_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test model under load"""
        
        concurrent_requests = test_config.get('concurrent_requests', 10)
        requests_per_client = test_config.get('requests_per_client', 20)
        sample_instance = test_config.get('sample_instance', {})
        
        async def make_predictions(client_id: int) -> List[float]:
            """Make predictions for a single client"""
            client_latencies = []
            
            for _ in range(requests_per_client):
                start_time = asyncio.get_event_loop().time()
                
                try:
                    prediction = endpoint.predict(instances=[sample_instance])
                    end_time = asyncio.get_event_loop().time()
                    
                    latency = (end_time - start_time) * 1000
                    client_latencies.append(latency)
                    
                except Exception as e:
                    print(f"Client {client_id} prediction failed: {e}")
            
            return client_latencies
        
        # Run concurrent load test
        tasks = [make_predictions(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        all_latencies = []
        total_requests = 0
        successful_requests = 0
        
        for client_result in results:
            if isinstance(client_result, list):
                all_latencies.extend(client_result)
                successful_requests += len(client_result)
            total_requests += requests_per_client
        
        load_test_results = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / total_requests,
            'concurrent_clients': concurrent_requests
        }
        
        if all_latencies:
            load_test_results.update({
                'mean_latency_ms': np.mean(all_latencies),
                'p95_latency_ms': np.percentile(all_latencies, 95),
                'degradation_factor': np.percentile(all_latencies, 95) / np.percentile(all_latencies, 50)
            })
        
        # Check if load test passes
        max_degradation = test_config.get('max_latency_degradation', 2.0)
        min_success_rate = test_config.get('min_success_rate', 0.95)
        
        passes_load_test = (
            load_test_results.get('degradation_factor', float('inf')) <= max_degradation and
            load_test_results['success_rate'] >= min_success_rate
        )
        
        return {
            'passes_test': passes_load_test,
            'load_test_results': load_test_results,
            'thresholds': {
                'max_degradation': max_degradation,
                'min_success_rate': min_success_rate
            }
        }
```

### 3. Production Validation Script

```python
# scripts/validate_production_deployment.py
import asyncio
import json
from typing import Dict, Any
from google.cloud import aiplatform, monitoring_v3
import time

class ProductionValidator:
    """Validate production ML deployments"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        aiplatform.init(project=project_id, location=region)
    
    async def validate_production_deployment(
        self,
        deployment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run complete production validation"""
        
        validation_results = {
            'deployment_id': deployment_config.get('deployment_id'),
            'validation_timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'pending'
        }
        
        # 1. Health checks
        health_check = await self._run_health_checks(deployment_config)
        validation_results['health_check'] = health_check
        
        # 2. Performance validation
        performance_check = await self._validate_performance(deployment_config)
        validation_results['performance_check'] = performance_check
        
        # 3. Security validation
        security_check = await self._validate_security(deployment_config)
        validation_results['security_check'] = security_check
        
        # 4. Cost validation
        cost_check = await self._validate_costs(deployment_config)
        validation_results['cost_check'] = cost_check
        
        # 5. Business logic validation
        business_check = await self._validate_business_logic(deployment_config)
        validation_results['business_check'] = business_check
        
        # Determine overall status
        all_checks = [health_check, performance_check, security_check, cost_check, business_check]
        all_passed = all(check.get('status') == 'passed' for check in all_checks)
        
        validation_results['overall_status'] = 'passed' if all_passed else 'failed'
        validation_results['summary'] = self._generate_validation_summary(all_checks)
        
        return validation_results
    
    async def _run_health_checks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run basic health checks"""
        
        endpoints = config.get('endpoints', [])
        health_results = []
        
        for endpoint_name in endpoints:
            try:
                endpoint = aiplatform.Endpoint(endpoint_name)
                
                # Check endpoint status
                endpoint_info = {
                    'endpoint_name': endpoint_name,
                    'create_time': endpoint.create_time,
                    'update_time': endpoint.update_time,
                    'deployed_models': len(endpoint.list_models()),
                    'status': 'healthy'
                }
                
                # Test prediction capability
                sample_instance = config.get('sample_prediction_instance', {})
                if sample_instance:
                    prediction = endpoint.predict(instances=[sample_instance])
                    endpoint_info['prediction_test'] = 'passed'
                else:
                    endpoint_info['prediction_test'] = 'skipped'
                
                health_results.append(endpoint_info)
                
            except Exception as e:
                health_results.append({
                    'endpoint_name': endpoint_name,
                    'status': 'unhealthy',
                    'error': str(e)
                })
        
        overall_health = all(result['status'] == 'healthy' for result in health_results)
        
        return {
            'status': 'passed' if overall_health else 'failed',
            'endpoint_results': health_results
        }
    
    async def _validate_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance meets requirements"""
        
        performance_requirements = config.get('performance_requirements', {})
        
        # Run performance tests
        performance_results = {}
        
        # Latency test
        max_latency = performance_requirements.get('max_latency_ms', 1000)
        latency_test = await self._test_latency_requirement(config, max_latency)
        performance_results['latency'] = latency_test
        
        # Throughput test
        min_qps = performance_requirements.get('min_qps', 10)
        throughput_test = await self._test_throughput_requirement(config, min_qps)
        performance_results['throughput'] = throughput_test
        
        # Model accuracy test
        min_accuracy = performance_requirements.get('min_accuracy', 0.8)
        accuracy_test = await self._test_accuracy_requirement(config, min_accuracy)
        performance_results['accuracy'] = accuracy_test
        
        all_performance_passed = all(
            test.get('passed', False) for test in performance_results.values()
        )
        
        return {
            'status': 'passed' if all_performance_passed else 'failed',
            'performance_results': performance_results
        }
    
    async def _validate_costs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate deployment costs are within budget"""
        
        cost_requirements = config.get('cost_requirements', {})
        
        # Estimate hourly costs
        estimated_costs = {}
        
        endpoints = config.get('endpoints', [])
        for endpoint_name in endpoints:
            endpoint = aiplatform.Endpoint(endpoint_name)
            deployed_models = endpoint.list_models()
            
            endpoint_cost = 0
            for model in deployed_models:
                # Simplified cost calculation
                machine_type = getattr(model, 'machine_type', 'n1-standard-4')
                replica_count = getattr(model, 'min_replica_count', 1)
                
                # Machine type hourly costs (simplified)
                machine_costs = {
                    'n1-standard-4': 0.19,
                    'n1-standard-8': 0.38,
                    'n1-highmem-4': 0.32
                }
                
                hourly_cost = machine_costs.get(machine_type, 0.19) * replica_count
                endpoint_cost += hourly_cost
            
            estimated_costs[endpoint_name] = {
                'hourly_cost': endpoint_cost,
                'daily_cost': endpoint_cost * 24,
                'monthly_cost': endpoint_cost * 24 * 30
            }
        
        # Check against budget
        total_monthly_cost = sum(cost['monthly_cost'] for cost in estimated_costs.values())
        monthly_budget = cost_requirements.get('monthly_budget', 1000)
        
        within_budget = total_monthly_cost <= monthly_budget
        
        return {
            'status': 'passed' if within_budget else 'failed',
            'estimated_costs': estimated_costs,
            'total_monthly_cost': total_monthly_cost,
            'monthly_budget': monthly_budget,
            'budget_utilization': (total_monthly_cost / monthly_budget) * 100
        }
```

## Rollback and Recovery

### Automated Rollback Strategy

```python
# deployment/rollback_manager.py
from google.cloud import aiplatform
from typing import Dict, Any, List
import asyncio

class RollbackManager:
    """Manage automated rollbacks for ML deployments"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
    
    async def setup_automated_rollback(
        self,
        endpoint_name: str,
        rollback_config: Dict[str, Any]
    ) -> str:
        """Setup automated rollback triggers"""
        
        # Create monitoring for rollback triggers
        rollback_triggers = {
            'error_rate_threshold': rollback_config.get('max_error_rate', 0.05),
            'latency_threshold': rollback_config.get('max_latency_ms', 2000),
            'performance_degradation': rollback_config.get('max_performance_drop', 0.1)
        }
        
        # Setup monitoring job
        monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name=f"{endpoint_name}-rollback-monitoring",
            endpoint=endpoint_name,
            logging_sampling_strategy=aiplatform.SamplingStrategy(
                random_sample_config=aiplatform.RandomSampleConfig(sample_rate=1.0)
            ),
            schedule_config=aiplatform.ScheduleConfig(cron="*/5 * * * *"),  # Every 5 minutes
            model_monitoring_alert_config={
                "email_alert_config": {
                    "user_emails": rollback_config.get('alert_emails', [])
                }
            }
        )
        
        return monitoring_job.resource_name
    
    async def execute_rollback(
        self,
        endpoint_name: str,
        target_model_version: str = "previous"
    ) -> Dict[str, Any]:
        """Execute model rollback"""
        
        endpoint = aiplatform.Endpoint(endpoint_name)
        deployed_models = endpoint.list_models()
        
        if len(deployed_models) < 2:
            return {
                'status': 'failed',
                'error': 'No previous model version available for rollback'
            }
        
        # Sort models by deployment time
        models_by_time = sorted(
            deployed_models,
            key=lambda x: x.create_time,
            reverse=True
        )
        
        current_model = models_by_time[0]
        previous_model = models_by_time[1]
        
        # Execute rollback
        try:
            # Update traffic allocation
            endpoint.update(
                deployed_models=[
                    {
                        'id': previous_model.id,
                        'traffic_percentage': 100
                    },
                    {
                        'id': current_model.id,
                        'traffic_percentage': 0
                    }
                ]
            )
            
            # Log rollback event
            rollback_info = {
                'status': 'success',
                'endpoint': endpoint_name,
                'rolled_back_from': current_model.display_name,
                'rolled_back_to': previous_model.display_name,
                'rollback_timestamp': datetime.utcnow().isoformat()
            }
            
            print(f"Rollback completed: {rollback_info}")
            
            return rollback_info
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': f"Rollback failed: {str(e)}"
            }
```

## Monitoring and Observability

### Deployment Monitoring Dashboard

```python
# monitoring/deployment_dashboard.py
import streamlit as st
import plotly.graph_objects as go
from google.cloud import aiplatform, monitoring_v3
import pandas as pd

class DeploymentMonitoringDashboard:
    """Real-time monitoring dashboard for ML deployments"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        aiplatform.init(project=project_id, location=region)
    
    def render_dashboard(self):
        """Render the deployment monitoring dashboard"""
        
        st.set_page_config(page_title="ML Deployment Monitor", layout="wide")
        st.title("ML Deployment Monitoring Dashboard")
        
        # Sidebar
        st.sidebar.header("Configuration")
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["1 hour", "6 hours", "24 hours", "7 days"]
        )
        
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
        
        # Main dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_endpoint_status()
        
        with col2:
            self._render_performance_metrics(time_range)
        
        # Full-width sections
        st.subheader("Model Performance Trends")
        self._render_performance_trends(time_range)
        
        st.subheader("Cost Analysis")
        self._render_cost_analysis(time_range)
        
        st.subheader("Recent Deployments")
        self._render_recent_deployments()
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
    
    def _render_endpoint_status(self):
        """Render endpoint status overview"""
        
        st.subheader("Endpoint Status")
        
        # Get all endpoints
        endpoints = aiplatform.Endpoint.list()
        
        status_data = []
        for endpoint in endpoints:
            deployed_models = endpoint.list_models()
            
            status_data.append({
                'Endpoint': endpoint.display_name,
                'Models': len(deployed_models),
                'Status': 'Active' if deployed_models else 'Inactive',
                'Created': endpoint.create_time.strftime('%Y-%m-%d %H:%M')
            })
        
        if status_data:
            df = pd.DataFrame(status_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No endpoints found")
    
    def _render_performance_metrics(self, time_range: str):
        """Render real-time performance metrics"""
        
        st.subheader("Performance Metrics")
        
        # Convert time range to seconds
        time_mapping = {
            "1 hour": 3600,
            "6 hours": 21600,
            "24 hours": 86400,
            "7 days": 604800
        }
        
        seconds = time_mapping.get(time_range, 3600)
        
        # Query metrics
        project_name = f"projects/{self.project_id}"
        
        # Prediction count metric
        filter_str = 'metric.type="aiplatform.googleapis.com/prediction/online/prediction_count"'
        
        request = monitoring_v3.ListTimeSeriesRequest(
            name=project_name,
            filter=filter_str,
            interval=monitoring_v3.TimeInterval({
                "end_time": {"seconds": int(time.time())},
                "start_time": {"seconds": int(time.time() - seconds)}
            })
        )
        
        try:
            time_series = self.monitoring_client.list_time_series(request=request)
            
            # Process metrics
            total_predictions = 0
            for series in time_series:
                for point in series.points:
                    total_predictions += point.value.int64_value
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", total_predictions)
            
            with col2:
                avg_qps = total_predictions / seconds if seconds > 0 else 0
                st.metric("Average QPS", f"{avg_qps:.2f}")
            
            with col3:
                # Estimated cost
                cost_per_prediction = 0.001
                estimated_cost = total_predictions * cost_per_prediction
                st.metric("Estimated Cost", f"${estimated_cost:.2f}")
        
        except Exception as e:
            st.error(f"Error fetching metrics: {e}")
```

## Best Practices Summary

### 1. CI/CD Pipeline Design
- Implement comprehensive testing at each stage
- Use canary deployments for risk mitigation
- Automate rollback based on performance metrics
- Include security scanning in the pipeline

### 2. Model Deployment
- Use semantic versioning for models
- Implement blue-green deployments for zero downtime
- Monitor model performance post-deployment
- Validate business logic after deployment

### 3. Testing Strategy
- Unit tests for individual components
- Integration tests for service interactions
- Load tests for performance validation
- Bias and fairness testing for ML models

### 4. Monitoring and Alerting
- Real-time performance monitoring
- Cost monitoring and alerting
- Business metric tracking
- Automated incident response

## Next Steps

1. **Pipeline Implementation**: Set up complete CI/CD pipelines
2. **Testing Automation**: Implement comprehensive test suites
3. **Monitoring Setup**: Deploy monitoring and alerting
4. **Rollback Testing**: Test automated rollback procedures
5. **Documentation**: Create runbooks for incident response

## Related Documentation

- [Vertex AI Integration](./vertex-ai/README.md)
- [Authentication Setup](./authentication/README.md)
- [Cost Optimization](./cost-optimization/README.md)
- [AI Platform Setup](./ai-platform/README.md)