"""
Vertex AI Service for IPO Valuation Platform
Handles model training, deployment, and prediction operations
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip
from google.cloud import bigquery
from google.cloud import storage
import joblib

logger = logging.getLogger(__name__)

class VertexAIService:
    """Service for managing Vertex AI operations for IPO valuation"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        
        # Initialize AI Platform
        aiplatform.init(project=project_id, location=location)
        
        # Initialize clients
        self.bq_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        
        # Model configurations
        self.model_configs = {
            "dcf_prediction": {
                "display_name": "DCF Valuation Model",
                "container_image_uri": f"gcr.io/{project_id}/dcf-predictor:latest",
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1
            },
            "risk_assessment": {
                "display_name": "Risk Assessment Model", 
                "container_image_uri": f"gcr.io/{project_id}/risk-assessor:latest",
                "machine_type": "n1-highmem-2"
            },
            "market_multiple": {
                "display_name": "Market Multiple Model",
                "container_image_uri": f"gcr.io/{project_id}/market-multiple:latest", 
                "machine_type": "n1-standard-2"
            }
        }
    
    def upload_model(self, model_name: str, model_artifacts: Dict[str, str]) -> aiplatform.Model:
        """Upload trained model to Vertex AI Model Registry"""
        try:
            config = self.model_configs.get(model_name)
            if not config:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Upload model artifacts to GCS
            model_uri = self._upload_model_artifacts(model_name, model_artifacts)
            
            # Create model in Vertex AI
            model = aiplatform.Model.upload(
                display_name=config["display_name"],
                artifact_uri=model_uri,
                serving_container_image_uri=config["container_image_uri"],
                serving_container_predict_route="/predict",
                serving_container_health_route="/health",
                description=f"IPO valuation model: {model_name}",
                labels={"model_type": model_name, "version": "v1"}
            )
            
            logger.info(f"Model {model_name} uploaded successfully: {model.resource_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error uploading model {model_name}: {str(e)}")
            raise
    
    def deploy_model(self, model_name: str, endpoint_name: Optional[str] = None) -> aiplatform.Endpoint:
        """Deploy model to Vertex AI endpoint"""
        try:
            config = self.model_configs.get(model_name)
            if not config:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Get the latest model version
            models = aiplatform.Model.list(filter=f'display_name="{config["display_name"]}"')
            if not models:
                raise ValueError(f"No models found for {model_name}")
            
            model = models[0]  # Latest model
            
            # Create or get endpoint
            if endpoint_name:
                endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
                endpoint = endpoints[0] if endpoints else None
            else:
                endpoint = None
                
            if not endpoint:
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_name or f"{model_name}-endpoint",
                    labels={"model_type": model_name}
                )
            
            # Deploy model to endpoint
            endpoint.deploy(
                model=model,
                deployed_model_display_name=f"{model_name}-deployment",
                machine_type=config["machine_type"],
                accelerator_type=config.get("accelerator_type"),
                accelerator_count=config.get("accelerator_count", 0),
                min_replica_count=1,
                max_replica_count=10,
                traffic_percentage=100
            )
            
            logger.info(f"Model {model_name} deployed to endpoint: {endpoint.resource_name}")
            return endpoint
            
        except Exception as e:
            logger.error(f"Error deploying model {model_name}: {str(e)}")
            raise
    
    def predict(self, endpoint_name: str, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions using deployed model"""
        try:
            # Get endpoint
            endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
            if not endpoints:
                raise ValueError(f"Endpoint {endpoint_name} not found")
            
            endpoint = endpoints[0]
            
            # Make prediction
            prediction = endpoint.predict(instances=instances)
            
            return prediction.predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def batch_predict(self, 
                     model_name: str,
                     input_uri: str, 
                     output_uri: str,
                     machine_type: str = "n1-standard-4") -> aiplatform.BatchPredictionJob:
        """Run batch prediction job"""
        try:
            config = self.model_configs.get(model_name)
            if not config:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Get model
            models = aiplatform.Model.list(filter=f'display_name="{config["display_name"]}"')
            if not models:
                raise ValueError(f"No models found for {model_name}")
            
            model = models[0]
            
            # Create batch prediction job
            job = aiplatform.BatchPredictionJob.create(
                job_display_name=f"{model_name}-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                model_name=model.resource_name,
                gcs_source=input_uri,
                gcs_destination_prefix=output_uri,
                machine_type=machine_type,
                starting_replica_count=1,
                max_replica_count=5
            )
            
            logger.info(f"Batch prediction job created: {job.resource_name}")
            return job
            
        except Exception as e:
            logger.error(f"Error creating batch prediction job: {str(e)}")
            raise
    
    def create_training_job(self, 
                          model_name: str,
                          training_data_uri: str,
                          validation_data_uri: str,
                          hyperparameters: Dict[str, Any]) -> aiplatform.CustomTrainingJob:
        """Create custom training job"""
        try:
            config = self.model_configs.get(model_name) 
            if not config:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Create training job
            job = aiplatform.CustomTrainingJob(
                display_name=f"{model_name}-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                script_path="training_script.py",  # Path to training script
                container_uri=f"gcr.io/{self.project_id}/training-{model_name}:latest",
                requirements=["pandas", "scikit-learn", "tensorflow", "xgboost"],
                model_serving_container_image_uri=config["container_image_uri"]
            )
            
            # Run training
            model = job.run(
                dataset=None,  # Data loaded from GCS
                replica_count=1,
                machine_type="n1-standard-4",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
                args=[
                    f"--training-data={training_data_uri}",
                    f"--validation-data={validation_data_uri}",
                    f"--model-name={model_name}"
                ] + [f"--{k}={v}" for k, v in hyperparameters.items()],
                environment_variables={
                    "PROJECT_ID": self.project_id,
                    "LOCATION": self.location
                }
            )
            
            logger.info(f"Training job created: {job.resource_name}")
            return job
            
        except Exception as e:
            logger.error(f"Error creating training job: {str(e)}")
            raise
    
    def create_hyperparameter_tuning_job(self,
                                       model_name: str, 
                                       training_data_uri: str,
                                       parameter_spec: Dict[str, Any]) -> aiplatform.HyperparameterTuningJob:
        """Create hyperparameter tuning job"""
        try:
            config = self.model_configs.get(model_name)
            if not config:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Define worker pool spec
            worker_pool_specs = [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-4",
                        "accelerator_type": "NVIDIA_TESLA_T4",
                        "accelerator_count": 1,
                    },
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": f"gcr.io/{self.project_id}/training-{model_name}:latest",
                        "args": [
                            f"--training-data={training_data_uri}",
                            f"--model-name={model_name}"
                        ]
                    },
                }
            ]
            
            # Create hyperparameter tuning job
            job = aiplatform.HyperparameterTuningJob(
                display_name=f"{model_name}-hpt-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                custom_job={
                    "job_spec": {
                        "worker_pool_specs": worker_pool_specs,
                    }
                },
                metric_spec={
                    "accuracy": "maximize",
                    "loss": "minimize"
                },
                parameter_spec=parameter_spec,
                max_trial_count=20,
                parallel_trial_count=5
            )
            
            job.run()
            
            logger.info(f"Hyperparameter tuning job created: {job.resource_name}")
            return job
            
        except Exception as e:
            logger.error(f"Error creating hyperparameter tuning job: {str(e)}")
            raise
    
    def get_feature_attributions(self, 
                                endpoint_name: str,
                                instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get feature attributions for model predictions"""
        try:
            # Get endpoint
            endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
            if not endpoints:
                raise ValueError(f"Endpoint {endpoint_name} not found")
            
            endpoint = endpoints[0]
            
            # Make prediction with explanations
            explanation = endpoint.explain(instances=instances)
            
            return explanation.explanations
            
        except Exception as e:
            logger.error(f"Error getting feature attributions: {str(e)}")
            raise
    
    def monitor_model_drift(self, model_name: str, monitoring_config: Dict[str, Any]) -> aiplatform.ModelDeploymentMonitoringJob:
        """Set up model monitoring for drift detection"""
        try:
            config = self.model_configs.get(model_name)
            if not config:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Get endpoint
            endpoints = aiplatform.Endpoint.list(filter=f'display_name="{model_name}-endpoint"')
            if not endpoints:
                raise ValueError(f"Endpoint for {model_name} not found")
            
            endpoint = endpoints[0]
            
            # Create monitoring job
            monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
                display_name=f"{model_name}-monitoring",
                endpoint=endpoint,
                logging_sampling_strategy={
                    "random_sample_config": {"sample_rate": monitoring_config.get("sample_rate", 0.1)}
                },
                schedule_config={
                    "cron": monitoring_config.get("cron_schedule", "0 */12 * * *")  # Every 12 hours
                },
                model_deployment_monitoring_objective_configs=[
                    {
                        "deployed_model_id": endpoint.list_models()[0].id,
                        "objective_config": {
                            "prediction_drift_detection_config": {
                                "drift_thresholds": {
                                    "categorical_threshold": monitoring_config.get("categorical_threshold", 0.3),
                                    "numeric_threshold": monitoring_config.get("numeric_threshold", 0.3)
                                }
                            },
                            "training_prediction_skew_detection_config": {
                                "skew_thresholds": {
                                    "categorical_threshold": monitoring_config.get("skew_categorical_threshold", 0.3),
                                    "numeric_threshold": monitoring_config.get("skew_numeric_threshold", 0.3)
                                }
                            }
                        }
                    }
                ],
                model_monitoring_alert_config={
                    "email_alert_config": {
                        "user_emails": monitoring_config.get("alert_emails", [])
                    },
                    "enable_logging": True
                }
            )
            
            logger.info(f"Model monitoring job created: {monitoring_job.resource_name}")
            return monitoring_job
            
        except Exception as e:
            logger.error(f"Error creating model monitoring job: {str(e)}")
            raise
    
    def _upload_model_artifacts(self, model_name: str, artifacts: Dict[str, str]) -> str:
        """Upload model artifacts to GCS"""
        bucket_name = f"{self.project_id}-ml-models"
        model_path = f"models/{model_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        bucket = self.storage_client.bucket(bucket_name)
        
        for artifact_name, local_path in artifacts.items():
            blob_path = f"{model_path}/{artifact_name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            
        return f"gs://{bucket_name}/{model_path}"
    
    def get_model_performance_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        try:
            config = self.model_configs.get(model_name)
            if not config:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Get model evaluation metrics from BigQuery
            query = f"""
            SELECT 
                evaluation_date,
                accuracy,
                precision,
                recall,
                f1_score,
                auc_roc,
                mean_squared_error,
                mean_absolute_error,
                r2_score
            FROM `{self.project_id}.ml_models.model_evaluations`
            WHERE model_name = '{model_name}'
            ORDER BY evaluation_date DESC
            LIMIT 1
            """
            
            query_job = self.bq_client.query(query)
            results = query_job.result()
            
            metrics = {}
            for row in results:
                metrics = dict(row)
                break
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            raise
    
    def retrain_model_if_needed(self, model_name: str, performance_threshold: float) -> bool:
        """Check if model needs retraining based on performance"""
        try:
            metrics = self.get_model_performance_metrics(model_name)
            
            # Determine if retraining is needed based on metrics
            primary_metric = metrics.get("accuracy") or metrics.get("r2_score", 0)
            
            if primary_metric < performance_threshold:
                logger.info(f"Model {model_name} performance below threshold. Triggering retraining.")
                
                # Trigger retraining job
                training_job = self.create_training_job(
                    model_name=model_name,
                    training_data_uri=f"gs://{self.project_id}-training-data/{model_name}/latest/",
                    validation_data_uri=f"gs://{self.project_id}-validation-data/{model_name}/latest/",
                    hyperparameters={}  # Use default hyperparameters
                )
                
                return True
            else:
                logger.info(f"Model {model_name} performance above threshold. No retraining needed.")
                return False
                
        except Exception as e:
            logger.error(f"Error checking retraining need: {str(e)}")
            raise