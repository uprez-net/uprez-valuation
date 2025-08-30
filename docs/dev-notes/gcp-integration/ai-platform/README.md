# AI Platform and MLOps Integration Guide

## Overview

This guide covers the complete MLOps implementation using Google Cloud AI Platform services for the IPO valuation platform. It includes model lifecycle management, continuous deployment, monitoring, and optimization strategies.

## Architecture Overview

### Core MLOps Components
- **Vertex AI Pipelines**: Automated ML workflows
- **Model Registry**: Centralized model versioning and metadata
- **Feature Store**: Managed feature engineering and serving
- **Model Monitoring**: Drift detection and performance tracking
- **Experiments Tracking**: A/B testing and experiment management
- **Continuous Training**: Automated retraining pipelines

### MLOps Workflow

```python
# High-level MLOps workflow
from google.cloud import aiplatform
from kfp.v2 import dsl
from typing import Dict, List, Any

class IPOValuationMLOps:
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
    
    async def setup_mlops_pipeline(self) -> str:
        """Setup complete MLOps pipeline"""
        pass
    
    async def deploy_model_with_monitoring(self, model_id: str) -> str:
        """Deploy model with monitoring setup"""
        pass
    
    async def trigger_retraining(self, trigger_conditions: Dict[str, Any]) -> str:
        """Trigger model retraining based on conditions"""
        pass
```

## Vertex AI Pipelines Implementation

### 1. Complete Training Pipeline

```python
# pipelines/ipo_valuation_pipeline.py
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from google.cloud import aiplatform
import json

@component(
    base_image="gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest",
    packages_to_install=["pandas", "scikit-learn", "google-cloud-bigquery", "google-cloud-storage"]
)
def data_extraction_component(
    project_id: str,
    query: str,
    output_dataset: Output[Dataset]
) -> None:
    """Extract training data from BigQuery"""
    
    from google.cloud import bigquery
    import pandas as pd
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Execute query
    df = client.query(query).to_dataframe()
    
    # Save dataset
    df.to_csv(output_dataset.path, index=False)
    
    print(f"Extracted {len(df)} rows to {output_dataset.path}")

@component(
    base_image="gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest",
    packages_to_install=["pandas", "scikit-learn", "numpy"]
)
def feature_engineering_component(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    feature_config: dict
) -> None:
    """Engineer features for model training"""
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Load data
    df = pd.read_csv(input_dataset.path)
    
    # Financial feature engineering
    df = engineer_financial_features(df, feature_config)
    
    # Save engineered dataset
    df.to_csv(output_dataset.path, index=False)
    
    print(f"Engineered features for {len(df)} samples")

def engineer_financial_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply financial feature engineering"""
    
    # Revenue growth features
    if 'revenue' in df.columns:
        df = df.sort_values(['company_id', 'fiscal_quarter'])
        df['revenue_growth_yoy'] = df.groupby('company_id')['revenue'].pct_change(periods=4)
        df['revenue_growth_qoq'] = df.groupby('company_id')['revenue'].pct_change(periods=1)
        
        # Rolling averages
        df['revenue_ma_4q'] = df.groupby('company_id')['revenue'].rolling(4).mean().reset_index(0, drop=True)
        df['revenue_volatility'] = df.groupby('company_id')['revenue'].rolling(8).std().reset_index(0, drop=True)
    
    # Profitability features
    if 'net_income' in df.columns and 'revenue' in df.columns:
        df['net_margin'] = df['net_income'] / df['revenue']
        df['margin_stability'] = df.groupby('company_id')['net_margin'].rolling(4).std().reset_index(0, drop=True)
    
    # Balance sheet features
    if 'total_debt' in df.columns and 'total_equity' in df.columns:
        df['debt_to_equity'] = df['total_debt'] / df['total_equity']
        df['leverage_trend'] = df.groupby('company_id')['debt_to_equity'].diff(periods=1)
    
    # Market features
    if 'market_cap' in df.columns:
        df['market_cap_log'] = np.log1p(df['market_cap'])
        df['market_cap_category'] = pd.cut(
            df['market_cap'],
            bins=[0, 2e9, 10e9, float('inf')],
            labels=['small_cap', 'mid_cap', 'large_cap']
        )
    
    # Sector encoding
    if 'sector' in df.columns:
        le = LabelEncoder()
        df['sector_encoded'] = le.fit_transform(df['sector'])
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df.groupby('company_id')[numeric_columns].fillna(method='ffill')
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    return df

@component(
    base_image="gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest",
    packages_to_install=["tensorflow", "scikit-learn", "pandas"]
)
def model_training_component(
    training_dataset: Input[Dataset],
    validation_dataset: Input[Dataset],
    hyperparameters: dict,
    model_output: Output[Model],
    metrics_output: Output[Metrics]
) -> None:
    """Train valuation model"""
    
    import tensorflow as tf
    import pandas as pd
    import json
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Load datasets
    train_df = pd.read_csv(training_dataset.path)
    val_df = pd.read_csv(validation_dataset.path)
    
    # Prepare features and targets
    feature_columns = hyperparameters.get('feature_columns', [])
    target_column = hyperparameters.get('target_column', 'enterprise_value')
    
    X_train = train_df[feature_columns].values
    y_train = train_df[target_column].values
    X_val = val_df[feature_columns].values
    y_val = val_df[target_column].values
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            hyperparameters.get('hidden_units_1', 128),
            activation='relu',
            input_shape=(len(feature_columns),)
        ),
        tf.keras.layers.Dropout(hyperparameters.get('dropout_1', 0.2)),
        tf.keras.layers.Dense(
            hyperparameters.get('hidden_units_2', 64),
            activation='relu'
        ),
        tf.keras.layers.Dropout(hyperparameters.get('dropout_2', 0.2)),
        tf.keras.layers.Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters.get('learning_rate', 0.001)),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=hyperparameters.get('epochs', 100),
        batch_size=hyperparameters.get('batch_size', 32),
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate model
    predictions = model.predict(X_val)
    
    # Calculate metrics
    metrics = {
        'mse': float(mean_squared_error(y_val, predictions)),
        'mae': float(mean_absolute_error(y_val, predictions)),
        'rmse': float(np.sqrt(mean_squared_error(y_val, predictions))),
        'r2': float(r2_score(y_val, predictions)),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    }
    
    # Save model
    model.save(model_output.path)
    
    # Save metrics
    with open(metrics_output.path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Model training completed. R² score: {metrics['r2']:.4f}")

@component(
    base_image="gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest",
    packages_to_install=["google-cloud-aiplatform"]
)
def model_evaluation_component(
    model: Input[Model],
    test_dataset: Input[Dataset],
    metrics: Input[Metrics],
    evaluation_output: Output[Metrics]
) -> str:
    """Evaluate model performance"""
    
    import tensorflow as tf
    import pandas as pd
    import json
    import numpy as np
    
    # Load model and data
    model = tf.keras.models.load_model(model.path)
    test_df = pd.read_csv(test_dataset.path)
    
    with open(metrics.path, 'r') as f:
        training_metrics = json.load(f)
    
    # Prepare test data
    feature_columns = test_df.columns[:-1].tolist()  # All but last column
    target_column = test_df.columns[-1]  # Last column as target
    
    X_test = test_df[feature_columns].values
    y_test = test_df[target_column].values
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    evaluation_metrics = {
        'test_mse': float(np.mean((y_test - predictions.flatten()) ** 2)),
        'test_mae': float(np.mean(np.abs(y_test - predictions.flatten()))),
        'test_r2': float(1 - np.sum((y_test - predictions.flatten()) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)),
        
        # Financial-specific metrics
        'mape': float(np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100),
        'directional_accuracy': float(np.mean(np.sign(y_test) == np.sign(predictions.flatten()))),
        
        # Model generalization
        'train_val_gap': abs(training_metrics['final_train_loss'] - training_metrics['final_val_loss']),
        'overfitting_indicator': training_metrics['final_train_loss'] / training_metrics['final_val_loss']
    }
    
    # Determine if model is ready for deployment
    deployment_criteria = {
        'r2_threshold': 0.7,
        'mape_threshold': 15.0,
        'overfitting_threshold': 2.0
    }
    
    ready_for_deployment = (
        evaluation_metrics['test_r2'] >= deployment_criteria['r2_threshold'] and
        evaluation_metrics['mape'] <= deployment_criteria['mape_threshold'] and
        evaluation_metrics['overfitting_indicator'] <= deployment_criteria['overfitting_threshold']
    )
    
    evaluation_metrics['ready_for_deployment'] = ready_for_deployment
    evaluation_metrics['deployment_criteria'] = deployment_criteria
    
    # Save evaluation results
    with open(evaluation_output.path, 'w') as f:
        json.dump(evaluation_metrics, f)
    
    return "approved" if ready_for_deployment else "rejected"

@component(
    base_image="gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest",
    packages_to_install=["google-cloud-aiplatform"]
)
def model_deployment_component(
    model: Input[Model],
    evaluation_metrics: Input[Metrics],
    endpoint_name: str,
    deployment_config: dict
) -> str:
    """Deploy model to Vertex AI endpoint"""
    
    import json
    from google.cloud import aiplatform
    
    # Load evaluation results
    with open(evaluation_metrics.path, 'r') as f:
        metrics = json.load(f)
    
    # Check if model passed evaluation
    if not metrics.get('ready_for_deployment', False):
        raise ValueError("Model did not meet deployment criteria")
    
    # Upload model
    uploaded_model = aiplatform.Model.upload(
        display_name=f"ipo-valuation-model-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        artifact_uri=model.path,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-11:latest",
        description=f"IPO valuation model with R²={metrics['test_r2']:.4f}"
    )
    
    # Get or create endpoint
    try:
        endpoint = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')[0]
    except IndexError:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            description="IPO valuation prediction endpoint"
        )
    
    # Deploy model
    deployed_model = endpoint.deploy(
        model=uploaded_model,
        deployed_model_display_name=f"{uploaded_model.display_name}-deployment",
        machine_type=deployment_config.get('machine_type', 'n1-standard-4'),
        min_replica_count=deployment_config.get('min_replicas', 1),
        max_replica_count=deployment_config.get('max_replicas', 5),
        traffic_percentage=deployment_config.get('traffic_percentage', 100),
        sync=True
    )
    
    return endpoint.resource_name

@pipeline(
    name="ipo-valuation-training-pipeline",
    description="Complete pipeline for IPO valuation model training and deployment"
)
def ipo_valuation_training_pipeline(
    project_id: str = "your-project-id",
    training_query: str = "",
    validation_query: str = "",
    test_query: str = "",
    hyperparameters: dict = {},
    deployment_config: dict = {},
    endpoint_name: str = "ipo-valuation-endpoint"
):
    """Complete training and deployment pipeline"""
    
    # Data extraction
    training_data_task = data_extraction_component(
        project_id=project_id,
        query=training_query
    )
    
    validation_data_task = data_extraction_component(
        project_id=project_id,
        query=validation_query
    )
    
    test_data_task = data_extraction_component(
        project_id=project_id,
        query=test_query
    )
    
    # Feature engineering
    feature_eng_task = feature_engineering_component(
        input_dataset=training_data_task.outputs["output_dataset"],
        feature_config=hyperparameters.get('feature_config', {})
    )
    
    # Model training
    training_task = model_training_component(
        training_dataset=feature_eng_task.outputs["output_dataset"],
        validation_dataset=validation_data_task.outputs["output_dataset"],
        hyperparameters=hyperparameters
    )
    
    # Model evaluation
    evaluation_task = model_evaluation_component(
        model=training_task.outputs["model_output"],
        test_dataset=test_data_task.outputs["output_dataset"],
        metrics=training_task.outputs["metrics_output"]
    )
    
    # Conditional deployment
    with dsl.Condition(
        evaluation_task.outputs["Output"] == "approved",
        name="deploy-if-approved"
    ):
        deployment_task = model_deployment_component(
            model=training_task.outputs["model_output"],
            evaluation_metrics=evaluation_task.outputs["evaluation_output"],
            endpoint_name=endpoint_name,
            deployment_config=deployment_config
        )

# Compile pipeline
def compile_pipeline():
    compiler.Compiler().compile(
        pipeline_func=ipo_valuation_training_pipeline,
        package_path="ipo_valuation_pipeline.json"
    )
```

### 2. Continuous Training Pipeline

```python
# pipelines/continuous_training.py
from google.cloud import aiplatform
from google.cloud import scheduler_v1
import json

class ContinuousTrainingManager:
    """Manage continuous training workflows"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        self.scheduler_client = scheduler_v1.CloudSchedulerClient()
        
        aiplatform.init(project=project_id, location=region)
    
    async def setup_scheduled_training(
        self,
        pipeline_template: str,
        schedule: str,
        job_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Setup scheduled training pipeline"""
        
        # Create Cloud Scheduler job
        parent = f"projects/{self.project_id}/locations/{self.region}"
        
        job = {
            "name": f"{parent}/jobs/{job_name}",
            "schedule": schedule,  # e.g., "0 2 * * 1" for Monday 2 AM
            "time_zone": "UTC",
            "http_target": {
                "uri": f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/pipelineJobs",
                "http_method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "displayName": f"scheduled-training-{job_name}",
                    "templateUri": pipeline_template,
                    "runtimeConfig": {
                        "parameters": parameters
                    }
                }).encode()
            }
        }
        
        request = scheduler_v1.CreateJobRequest(
            parent=parent,
            job=job
        )
        
        response = self.scheduler_client.create_job(request=request)
        return response.name
    
    async def setup_trigger_based_training(
        self,
        trigger_conditions: Dict[str, Any]
    ) -> str:
        """Setup trigger-based retraining"""
        
        # Trigger conditions examples:
        # - Model performance degradation
        # - Data drift detection
        # - New data availability
        # - Time-based triggers
        
        trigger_config = {
            'performance_threshold': trigger_conditions.get('min_r2_score', 0.65),
            'drift_threshold': trigger_conditions.get('max_drift_score', 0.1),
            'data_freshness_days': trigger_conditions.get('max_data_age_days', 30),
            'evaluation_frequency_hours': trigger_conditions.get('check_frequency_hours', 24)
        }
        
        # Create monitoring pipeline for triggers
        monitoring_pipeline = self._create_monitoring_pipeline(trigger_config)
        
        return monitoring_pipeline
    
    def _create_monitoring_pipeline(self, trigger_config: Dict[str, Any]) -> str:
        """Create pipeline to monitor for retraining triggers"""
        
        @component(
            base_image="gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest",
            packages_to_install=["google-cloud-aiplatform", "google-cloud-bigquery"]
        )
        def model_performance_monitor(
            project_id: str,
            endpoint_name: str,
            performance_threshold: float
        ) -> str:
            """Monitor model performance and trigger retraining if needed"""
            
            # Get recent predictions and actuals
            from google.cloud import bigquery
            
            client = bigquery.Client(project=project_id)
            query = f"""
            SELECT
              predicted_value,
              actual_value,
              prediction_timestamp
            FROM `{project_id}.ipo_valuation.prediction_history`
            WHERE prediction_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
              AND actual_value IS NOT NULL
            """
            
            df = client.query(query).to_dataframe()
            
            if len(df) > 10:  # Minimum samples for evaluation
                # Calculate current R² score
                from sklearn.metrics import r2_score
                current_r2 = r2_score(df['actual_value'], df['predicted_value'])
                
                if current_r2 < performance_threshold:
                    return "trigger_retraining"
                else:
                    return "performance_ok"
            
            return "insufficient_data"
        
        @component(
            base_image="gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest",
            packages_to_install=["google-cloud-aiplatform"]
        )
        def data_drift_monitor(
            project_id: str,
            dataset_name: str,
            drift_threshold: float
        ) -> str:
            """Monitor for data drift"""
            
            # Simplified drift detection
            # In practice, would use Vertex AI Model Monitoring
            
            from google.cloud import bigquery
            import numpy as np
            
            client = bigquery.Client(project=project_id)
            
            # Get recent data statistics
            recent_query = f"""
            SELECT * FROM `{dataset_name}`
            WHERE fiscal_quarter >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH)
            """
            
            historical_query = f"""
            SELECT * FROM `{dataset_name}`
            WHERE fiscal_quarter BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
                                    AND DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
            """
            
            recent_df = client.query(recent_query).to_dataframe()
            historical_df = client.query(historical_query).to_dataframe()
            
            if len(recent_df) > 0 and len(historical_df) > 0:
                # Calculate distribution differences for key features
                numeric_columns = recent_df.select_dtypes(include=[np.number]).columns
                
                drift_scores = []
                for col in numeric_columns:
                    if col in historical_df.columns:
                        # Simple distribution comparison using KL divergence approximation
                        recent_mean = recent_df[col].mean()
                        recent_std = recent_df[col].std()
                        hist_mean = historical_df[col].mean()
                        hist_std = historical_df[col].std()
                        
                        # Normalized difference
                        mean_drift = abs(recent_mean - hist_mean) / (hist_std + 1e-8)
                        drift_scores.append(mean_drift)
                
                avg_drift = np.mean(drift_scores) if drift_scores else 0.0
                
                if avg_drift > drift_threshold:
                    return "drift_detected"
                else:
                    return "no_drift"
            
            return "insufficient_data"
        
        # Create monitoring pipeline
        @pipeline(
            name="model-monitoring-pipeline",
            description="Monitor model performance and data drift"
        )
        def monitoring_pipeline(
            project_id: str,
            endpoint_name: str,
            dataset_name: str,
            performance_threshold: float,
            drift_threshold: float
        ):
            # Monitor performance
            perf_task = model_performance_monitor(
                project_id=project_id,
                endpoint_name=endpoint_name,
                performance_threshold=performance_threshold
            )
            
            # Monitor data drift
            drift_task = data_drift_monitor(
                project_id=project_id,
                dataset_name=dataset_name,
                drift_threshold=drift_threshold
            )
            
            # Trigger retraining if needed
            with dsl.Condition(
                (perf_task.outputs["Output"] == "trigger_retraining") |
                (drift_task.outputs["Output"] == "drift_detected"),
                name="trigger-retraining"
            ):
                # Would trigger the main training pipeline here
                pass
        
        return "monitoring_pipeline_created"
```

## Model Registry and Versioning

### Model Lifecycle Management

```python
# registry/model_lifecycle_manager.py
from google.cloud import aiplatform
from typing import Dict, List, Any, Optional
import semver

class ModelLifecycleManager:
    """Manage model versions and lifecycle"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
    
    async def register_model(
        self,
        model_artifact_uri: str,
        model_name: str,
        model_type: str,
        performance_metrics: Dict[str, float],
        training_metadata: Dict[str, Any]
    ) -> str:
        """Register new model version"""
        
        # Create model version
        version = self._generate_version_number(model_name)
        
        # Upload model to registry
        model = aiplatform.Model.upload(
            display_name=f"{model_name}-v{version}",
            artifact_uri=model_artifact_uri,
            serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-11:latest",
            description=f"{model_type} model for IPO valuation",
            labels={
                'model_type': model_type,
                'version': version,
                'environment': 'production'
            },
            explanation_metadata=aiplatform.explain.ExplanationMetadata(
                inputs={
                    'features': aiplatform.explain.ExplanationMetadata.InputMetadata(
                        input_tensor_name='features'
                    )
                },
                outputs={
                    'prediction': aiplatform.explain.ExplanationMetadata.OutputMetadata(
                        output_tensor_name='prediction'
                    )
                }
            )
        )
        
        # Store model metadata
        await self._store_model_metadata(
            model.resource_name,
            version,
            performance_metrics,
            training_metadata
        )
        
        return model.resource_name
    
    async def promote_model(
        self,
        model_name: str,
        from_environment: str,
        to_environment: str,
        approval_criteria: Dict[str, Any]
    ) -> bool:
        """Promote model between environments"""
        
        # Get model performance metrics
        metrics = await self._get_model_metrics(model_name, from_environment)
        
        # Check approval criteria
        approval_passed = self._check_approval_criteria(metrics, approval_criteria)
        
        if approval_passed:
            # Update model labels for promotion
            model = aiplatform.Model(model_name)
            model.update(
                labels={
                    **model.labels,
                    'environment': to_environment,
                    'promoted_at': datetime.utcnow().isoformat()
                }
            )
            
            # Create deployment in target environment
            await self._deploy_to_environment(model_name, to_environment)
            
            return True
        
        return False
    
    async def rollback_model(
        self,
        endpoint_name: str,
        target_version: str
    ) -> str:
        """Rollback to previous model version"""
        
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        # Get deployed models
        deployed_models = endpoint.list_models()
        
        # Find target version
        target_model = None
        for model in deployed_models:
            if target_version in model.display_name:
                target_model = model
                break
        
        if not target_model:
            raise ValueError(f"Model version {target_version} not found")
        
        # Update traffic allocation
        endpoint.update(
            deployed_models=[
                {
                    'id': target_model.id,
                    'traffic_percentage': 100
                }
            ]
        )
        
        return f"Rolled back to {target_version}"
    
    def _generate_version_number(self, model_name: str) -> str:
        """Generate semantic version number"""
        
        # Get existing models
        models = aiplatform.Model.list(filter=f'display_name:"{model_name}"')
        
        if not models:
            return "1.0.0"
        
        # Extract versions
        versions = []
        for model in models:
            version_label = model.labels.get('version')
            if version_label and semver.VersionInfo.isvalid(version_label):
                versions.append(version_label)
        
        if not versions:
            return "1.0.0"
        
        # Get latest version and increment
        latest_version = max(versions, key=lambda v: semver.VersionInfo.parse(v))
        next_version = semver.bump_minor(latest_version)
        
        return next_version
```

## Model Monitoring and Alerting

### 1. Performance Monitoring Setup

```python
# monitoring/model_performance_monitor.py
from google.cloud import aiplatform, monitoring_v3
from typing import Dict, Any
import asyncio

class ModelPerformanceMonitor:
    """Monitor model performance in production"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        
        aiplatform.init(project=project_id, location=region)
    
    async def setup_comprehensive_monitoring(
        self,
        endpoint_name: str,
        monitoring_config: Dict[str, Any]
    ) -> str:
        """Setup comprehensive model monitoring"""
        
        # Setup prediction drift monitoring
        drift_job = await self._setup_prediction_drift_monitoring(
            endpoint_name, monitoring_config.get('drift_config', {})
        )
        
        # Setup feature skew monitoring
        skew_job = await self._setup_feature_skew_monitoring(
            endpoint_name, monitoring_config.get('skew_config', {})
        )
        
        # Setup performance monitoring
        perf_job = await self._setup_performance_monitoring(
            endpoint_name, monitoring_config.get('performance_config', {})
        )
        
        # Setup custom business metrics monitoring
        business_job = await self._setup_business_metrics_monitoring(
            endpoint_name, monitoring_config.get('business_config', {})
        )
        
        return f"Monitoring setup complete: {drift_job}, {skew_job}, {perf_job}, {business_job}"
    
    async def _setup_prediction_drift_monitoring(
        self,
        endpoint_name: str,
        drift_config: Dict[str, Any]
    ) -> str:
        """Setup prediction drift monitoring"""
        
        # Create monitoring job
        job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name=f"{endpoint_name}-drift-monitoring",
            endpoint=endpoint_name,
            logging_sampling_strategy=aiplatform.SamplingStrategy(
                random_sample_config=aiplatform.RandomSampleConfig(
                    sample_rate=drift_config.get('sample_rate', 0.1)
                )
            ),
            model_monitoring_config={
                "objective_config": {
                    "prediction_drift_detection_config": {
                        "drift_thresholds": {
                            "valuation_prediction": {
                                "value": drift_config.get('drift_threshold', 0.1)
                            }
                        }
                    }
                }
            },
            model_monitoring_alert_config={
                "email_alert_config": {
                    "user_emails": drift_config.get('alert_emails', [])
                },
                "notification_channels": drift_config.get('notification_channels', [])
            }
        )
        
        return job.resource_name
    
    async def _setup_business_metrics_monitoring(
        self,
        endpoint_name: str,
        business_config: Dict[str, Any]
    ) -> str:
        """Setup business-specific metrics monitoring"""
        
        # Custom metrics for financial models
        financial_metrics = [
            'prediction_accuracy_by_sector',
            'valuation_error_distribution',
            'model_confidence_trends',
            'prediction_volume_by_market_condition'
        ]
        
        # Create custom metric descriptors
        for metric_name in financial_metrics:
            self._create_custom_metric_descriptor(
                metric_name, 
                business_config.get(f'{metric_name}_config', {})
            )
        
        return "business_metrics_monitoring_setup"
    
    def _create_custom_metric_descriptor(
        self,
        metric_name: str,
        metric_config: Dict[str, Any]
    ):
        """Create custom metric descriptor"""
        
        project_name = f"projects/{self.project_id}"
        
        descriptor = monitoring_v3.MetricDescriptor(
            type=f"custom.googleapis.com/ipo_valuation/{metric_name}",
            metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            value_type=monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            description=metric_config.get('description', f'Custom metric: {metric_name}'),
            display_name=metric_config.get('display_name', metric_name.replace('_', ' ').title())
        )
        
        try:
            self.monitoring_client.create_metric_descriptor(
                name=project_name,
                metric_descriptor=descriptor
            )
        except Exception as e:
            # Metric might already exist
            if "already exists" not in str(e):
                raise
```

## A/B Testing Framework

### Experiment Management

```python
# experiments/ab_testing_manager.py
from google.cloud import aiplatform
from typing import Dict, List, Any
import random

class ModelABTestingManager:
    """Manage A/B testing for model deployments"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
    
    async def setup_ab_test(
        self,
        endpoint_name: str,
        model_a: str,
        model_b: str,
        traffic_split: Dict[str, int],
        test_duration_days: int,
        success_metrics: List[str]
    ) -> str:
        """
        Setup A/B test between two models
        
        Args:
            endpoint_name: Endpoint for testing
            model_a: Control model
            model_b: Treatment model
            traffic_split: Traffic allocation (e.g., {'model_a': 70, 'model_b': 30})
            test_duration_days: Test duration
            success_metrics: Metrics to track for success
            
        Returns:
            Experiment ID
        """
        
        # Get endpoint
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        # Deploy both models to the endpoint
        model_a_deployment = endpoint.deploy(
            model=aiplatform.Model(model_a),
            deployed_model_display_name=f"model-a-{datetime.utcnow().strftime('%Y%m%d')}",
            traffic_percentage=traffic_split['model_a'],
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3
        )
        
        model_b_deployment = endpoint.deploy(
            model=aiplatform.Model(model_b),
            deployed_model_display_name=f"model-b-{datetime.utcnow().strftime('%Y%m%d')}",
            traffic_percentage=traffic_split['model_b'],
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3
        )
        
        # Create experiment tracking
        experiment_id = f"ab-test-{endpoint_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        experiment_config = {
            'experiment_id': experiment_id,
            'endpoint_name': endpoint_name,
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'start_date': datetime.utcnow().isoformat(),
            'end_date': (datetime.utcnow() + timedelta(days=test_duration_days)).isoformat(),
            'success_metrics': success_metrics,
            'status': 'running'
        }
        
        # Store experiment configuration
        await self._store_experiment_config(experiment_config)
        
        # Setup monitoring for the experiment
        await self._setup_experiment_monitoring(experiment_id, success_metrics)
        
        return experiment_id
    
    async def analyze_ab_test_results(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """Analyze A/B test results"""
        
        # Get experiment configuration
        experiment_config = await self._get_experiment_config(experiment_id)
        
        # Query prediction results from BigQuery
        from google.cloud import bigquery
        
        client = bigquery.Client(project=self.project_id)
        
        query = f"""
        WITH experiment_predictions AS (
          SELECT
            model_version,
            predicted_value,
            actual_value,
            prediction_timestamp,
            request_id
          FROM `{self.project_id}.ipo_valuation.prediction_history`
          WHERE prediction_timestamp BETWEEN '{experiment_config['start_date']}'
                                        AND '{experiment_config['end_date']}'
            AND endpoint_name = '{experiment_config['endpoint_name']}'
        )
        
        SELECT
          model_version,
          COUNT(*) as prediction_count,
          AVG(ABS(predicted_value - actual_value)) as mae,
          SQRT(AVG(POW(predicted_value - actual_value, 2))) as rmse,
          CORR(predicted_value, actual_value) as correlation,
          AVG(ABS((predicted_value - actual_value) / actual_value)) * 100 as mape
          
        FROM experiment_predictions
        WHERE actual_value IS NOT NULL
        GROUP BY model_version
        """
        
        results_df = client.query(query).to_dataframe()
        
        # Statistical significance testing
        significance_results = await self._calculate_statistical_significance(
            experiment_id, results_df
        )
        
        # Business impact analysis
        business_impact = await self._analyze_business_impact(
            experiment_id, results_df
        )
        
        return {
            'experiment_id': experiment_id,
            'model_performance': results_df.to_dict('records'),
            'statistical_significance': significance_results,
            'business_impact': business_impact,
            'recommendation': self._generate_recommendation(
                results_df, significance_results, business_impact
            )
        }
```

## Cost Optimization Strategies

### Resource Management

```python
# optimization/resource_optimizer.py
from google.cloud import aiplatform
from typing import Dict, Any

class ResourceOptimizer:
    """Optimize resource usage and costs for ML workloads"""
    
    def __init__(self, project_id: str, region: str):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
    
    async def optimize_endpoint_resources(
        self,
        endpoint_name: str,
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize endpoint resource allocation"""
        
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        # Analyze current usage patterns
        usage_analysis = await self._analyze_endpoint_usage(endpoint_name)
        
        # Calculate optimal configuration
        optimal_config = self._calculate_optimal_resources(
            usage_analysis, optimization_config
        )
        
        # Apply optimizations
        optimization_results = await self._apply_optimizations(
            endpoint, optimal_config
        )
        
        return optimization_results
    
    async def _analyze_endpoint_usage(self, endpoint_name: str) -> Dict[str, Any]:
        """Analyze endpoint usage patterns"""
        
        # Query prediction metrics from monitoring
        from google.cloud import monitoring_v3
        
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{self.project_id}"
        
        # Get QPS metrics
        qps_query = monitoring_v3.ListTimeSeriesRequest(
            name=project_name,
            filter=f'metric.type="aiplatform.googleapis.com/prediction/online/prediction_count" '
                   f'AND resource.labels.endpoint_id="{endpoint_name}"',
            interval=monitoring_v3.TimeInterval({
                "end_time": {"seconds": int(time.time())},
                "start_time": {"seconds": int(time.time() - 7 * 24 * 3600)}  # Last 7 days
            })
        )
        
        qps_series = client.list_time_series(request=qps_query)
        
        # Analyze patterns
        usage_patterns = {
            'avg_qps': 0,
            'peak_qps': 0,
            'min_qps': 0,
            'usage_trend': 'stable'
        }
        
        # Process time series data
        qps_values = []
        for series in qps_series:
            for point in series.points:
                qps_values.append(point.value.double_value)
        
        if qps_values:
            usage_patterns['avg_qps'] = np.mean(qps_values)
            usage_patterns['peak_qps'] = np.max(qps_values)
            usage_patterns['min_qps'] = np.min(qps_values)
            
            # Simple trend analysis
            if len(qps_values) > 10:
                recent_avg = np.mean(qps_values[-24:])  # Last 24 hours
                historical_avg = np.mean(qps_values[:-24])
                
                if recent_avg > historical_avg * 1.2:
                    usage_patterns['usage_trend'] = 'increasing'
                elif recent_avg < historical_avg * 0.8:
                    usage_patterns['usage_trend'] = 'decreasing'
        
        return usage_patterns
```

## Best Practices

### 1. Pipeline Design
- Use component-based architecture for reusability
- Implement proper error handling and retries
- Use parallel processing where possible
- Implement comprehensive logging

### 2. Model Management
- Use semantic versioning for models
- Implement automated testing for model quality
- Use feature stores for consistent feature engineering
- Implement model approval workflows

### 3. Monitoring Strategy
- Monitor both technical and business metrics
- Set up automated alerting for anomalies
- Implement drift detection for data and predictions
- Track model performance over time

### 4. Cost Management
- Use appropriate machine types for workloads
- Implement auto-scaling for variable demand
- Monitor and optimize resource usage
- Use batch processing for non-urgent workloads

## Next Steps

1. **Pipeline Implementation**: Deploy complete MLOps pipelines
2. **Monitoring Setup**: Implement comprehensive monitoring
3. **A/B Testing**: Set up experimentation framework
4. **Cost Optimization**: Implement resource optimization
5. **Integration Testing**: Test end-to-end workflows

## Related Documentation

- [Vertex AI Integration](../vertex-ai/README.md)
- [Authentication Setup](../authentication/README.md)
- [Cost Optimization](../cost-optimization/README.md)
- [BigQuery ML Integration](../bigquery-ml/README.md)