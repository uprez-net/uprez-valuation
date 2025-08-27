"""
Vertex AI Integration Client
Google Cloud Vertex AI integration for ML model serving and training
"""
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic
from google.cloud.aiplatform_v1 import (
    EndpointServiceClient,
    ModelServiceClient,
    PipelineServiceClient
)
import numpy as np

from ...config import settings
from ...utils.metrics import track_ml_inference, track_time, ML_INFERENCE_DURATION
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelPrediction:
    """Model prediction result"""
    predictions: List[Any]
    explanations: Optional[List[Any]] = None
    confidence_scores: Optional[List[float]] = None
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


class VertexAIClient:
    """Client for Google Cloud Vertex AI operations"""
    
    def __init__(self):
        self.project_id = settings.gcp.project_id
        self.region = settings.gcp.region
        self.endpoint_client = None
        self.model_client = None
        self.pipeline_client = None
        
        # Initialize clients if GCP is configured
        if self.project_id:
            self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Vertex AI clients"""
        try:
            # Initialize AI Platform
            aiplatform.init(project=self.project_id, location=self.region)
            
            # Initialize service clients
            self.endpoint_client = EndpointServiceClient(
                client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
            )
            self.model_client = ModelServiceClient(
                client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
            )
            self.pipeline_client = PipelineServiceClient(
                client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
            )
            
            logger.info("Vertex AI clients initialized", project=self.project_id, region=self.region)
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI clients: {str(e)}")
            raise
    
    @track_time(ML_INFERENCE_DURATION, {"model_name": "vertex_ai"})
    async def predict(
        self,
        endpoint_name: str,
        instances: List[Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ModelPrediction:
        """
        Make predictions using a deployed model endpoint
        
        Args:
            endpoint_name: Name or ID of the endpoint
            instances: List of instances to predict
            parameters: Optional prediction parameters
        
        Returns:
            Prediction results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            track_ml_inference("vertex_ai_predict", start_time, True)
            
            # Build endpoint resource name if needed
            if not endpoint_name.startswith("projects/"):
                endpoint_name = f"projects/{self.project_id}/locations/{self.region}/endpoints/{endpoint_name}"
            
            # Prepare request
            request = gapic.PredictRequest(
                endpoint=endpoint_name,
                instances=instances,
                parameters=parameters or {}
            )
            
            # Make prediction
            response = self.endpoint_client.predict(request=request)
            
            # Calculate latency
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Extract predictions and metadata
            predictions = [dict(prediction) for prediction in response.predictions]
            explanations = None
            if hasattr(response, 'explanations') and response.explanations:
                explanations = [dict(explanation) for explanation in response.explanations]
            
            # Extract confidence scores if available
            confidence_scores = None
            if predictions and 'confidence' in predictions[0]:
                confidence_scores = [pred.get('confidence') for pred in predictions]
            
            result = ModelPrediction(
                predictions=predictions,
                explanations=explanations,
                confidence_scores=confidence_scores,
                model_version=getattr(response, 'model_version_id', None),
                latency_ms=latency_ms
            )
            
            logger.info(
                "Vertex AI prediction completed",
                endpoint=endpoint_name,
                instances_count=len(instances),
                latency_ms=latency_ms
            )
            
            return result
            
        except Exception as e:
            track_ml_inference("vertex_ai_predict", start_time, False)
            logger.error(f"Vertex AI prediction failed: {str(e)}", endpoint=endpoint_name)
            raise
    
    async def batch_predict(
        self,
        model_name: str,
        input_config: Dict[str, Any],
        output_config: Dict[str, Any],
        machine_type: str = "n1-standard-4",
        max_replica_count: int = 10
    ) -> str:
        """
        Submit batch prediction job
        
        Args:
            model_name: Name of the model
            input_config: Input data configuration
            output_config: Output data configuration
            machine_type: Machine type for batch job
            max_replica_count: Maximum number of replicas
        
        Returns:
            Batch job ID
        """
        try:
            # Create batch prediction job
            job = aiplatform.BatchPredictionJob.create(
                job_display_name=f"batch-prediction-{model_name}",
                model_name=model_name,
                gcs_source=input_config.get('gcs_source'),
                gcs_destination_prefix=output_config.get('gcs_destination'),
                machine_type=machine_type,
                max_replica_count=max_replica_count
            )
            
            logger.info(
                "Batch prediction job created",
                job_id=job.name,
                model=model_name
            )
            
            return job.name
            
        except Exception as e:
            logger.error(f"Batch prediction job creation failed: {str(e)}")
            raise
    
    async def deploy_model(
        self,
        model_name: str,
        endpoint_name: str,
        machine_type: str = "n1-standard-4",
        min_replica_count: int = 1,
        max_replica_count: int = 3
    ) -> str:
        """
        Deploy model to an endpoint
        
        Args:
            model_name: Name of the model to deploy
            endpoint_name: Name of the endpoint
            machine_type: Machine type for serving
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
        
        Returns:
            Deployment ID
        """
        try:
            # Get model
            model = aiplatform.Model(model_name)
            
            # Get or create endpoint
            try:
                endpoint = aiplatform.Endpoint(endpoint_name)
            except:
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_name,
                    project=self.project_id,
                    location=self.region
                )
            
            # Deploy model
            deployment = model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=f"{model_name}-deployment",
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                traffic_percentage=100
            )
            
            logger.info(
                "Model deployed successfully",
                model=model_name,
                endpoint=endpoint_name,
                deployment_id=deployment.id
            )
            
            return deployment.id
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            raise
    
    async def train_custom_model(
        self,
        display_name: str,
        training_script_path: str,
        container_uri: str,
        training_data_uri: str,
        validation_data_uri: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        machine_type: str = "n1-standard-4"
    ) -> str:
        """
        Train custom model using Vertex AI Training
        
        Args:
            display_name: Display name for the training job
            training_script_path: Path to training script
            container_uri: Container image URI
            training_data_uri: URI of training data
            validation_data_uri: URI of validation data
            hyperparameters: Training hyperparameters
            machine_type: Machine type for training
        
        Returns:
            Training job ID
        """
        try:
            # Create custom training job
            job = aiplatform.CustomTrainingJob(
                display_name=display_name,
                script_path=training_script_path,
                container_uri=container_uri,
                requirements=["scikit-learn", "pandas", "numpy"],
                model_serving_container_image_uri=container_uri
            )
            
            # Submit training job
            model = job.run(
                dataset=training_data_uri,
                validation_dataset=validation_data_uri,
                args=hyperparameters or {},
                machine_type=machine_type,
                replica_count=1
            )
            
            logger.info(
                "Custom training job submitted",
                job_name=job.display_name,
                model_name=model.display_name if model else None
            )
            
            return job.resource_name
            
        except Exception as e:
            logger.error(f"Custom training job failed: {str(e)}")
            raise
    
    async def create_automl_model(
        self,
        display_name: str,
        dataset_id: str,
        prediction_type: str,
        objective: str,
        budget_milli_node_hours: int = 1000
    ) -> str:
        """
        Create AutoML model
        
        Args:
            display_name: Display name for the model
            dataset_id: ID of the training dataset
            prediction_type: Type of prediction (classification/regression)
            objective: Training objective
            budget_milli_node_hours: Training budget in milli node hours
        
        Returns:
            Model ID
        """
        try:
            if prediction_type.lower() == "classification":
                # Create AutoML Tabular Classification model
                model = aiplatform.AutoMLTabularTrainingJob(
                    display_name=display_name,
                    optimization_prediction_type="classification",
                    optimization_objective=objective,
                    column_transformations=[],
                    budget_milli_node_hours=budget_milli_node_hours
                )
            else:
                # Create AutoML Tabular Regression model
                model = aiplatform.AutoMLTabularTrainingJob(
                    display_name=display_name,
                    optimization_prediction_type="regression",
                    optimization_objective=objective,
                    column_transformations=[],
                    budget_milli_node_hours=budget_milli_node_hours
                )
            
            # Get dataset
            dataset = aiplatform.TabularDataset(dataset_id)
            
            # Train model
            trained_model = model.run(dataset=dataset, sync=False)
            
            logger.info(
                "AutoML training job created",
                display_name=display_name,
                dataset_id=dataset_id,
                prediction_type=prediction_type
            )
            
            return trained_model.resource_name if trained_model else model.resource_name
            
        except Exception as e:
            logger.error(f"AutoML model creation failed: {str(e)}")
            raise
    
    async def get_model_evaluation(self, model_name: str) -> ModelMetrics:
        """
        Get model evaluation metrics
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model evaluation metrics
        """
        try:
            model = aiplatform.Model(model_name)
            
            # Get model evaluations
            evaluations = model.list_model_evaluations()
            
            if not evaluations:
                raise ValueError(f"No evaluations found for model {model_name}")
            
            # Extract metrics from the latest evaluation
            latest_eval = evaluations[0]
            metrics_dict = latest_eval.metrics
            
            # Map to standardized metrics
            metrics = ModelMetrics(
                accuracy=metrics_dict.get('accuracy', 0.0),
                precision=metrics_dict.get('precision', 0.0),
                recall=metrics_dict.get('recall', 0.0),
                f1_score=metrics_dict.get('f1Score', 0.0),
                auc=metrics_dict.get('auc', None),
                custom_metrics={k: v for k, v in metrics_dict.items() 
                              if k not in ['accuracy', 'precision', 'recall', 'f1Score', 'auc']}
            )
            
            logger.info(
                "Model evaluation retrieved",
                model=model_name,
                accuracy=metrics.accuracy
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get model evaluation: {str(e)}")
            raise
    
    async def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all endpoints in the project"""
        try:
            parent = f"projects/{self.project_id}/locations/{self.region}"
            request = gapic.ListEndpointsRequest(parent=parent)
            
            page_result = self.endpoint_client.list_endpoints(request=request)
            
            endpoints = []
            for endpoint in page_result:
                endpoints.append({
                    'name': endpoint.name,
                    'display_name': endpoint.display_name,
                    'create_time': endpoint.create_time,
                    'update_time': endpoint.update_time,
                    'deployed_models': [
                        {
                            'id': dm.id,
                            'model': dm.model,
                            'display_name': dm.display_name
                        }
                        for dm in endpoint.deployed_models
                    ]
                })
            
            return endpoints
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {str(e)}")
            raise
    
    async def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete an endpoint"""
        try:
            if not endpoint_name.startswith("projects/"):
                endpoint_name = f"projects/{self.project_id}/locations/{self.region}/endpoints/{endpoint_name}"
            
            request = gapic.DeleteEndpointRequest(name=endpoint_name)
            operation = self.endpoint_client.delete_endpoint(request=request)
            
            # Wait for operation to complete
            operation.result()
            
            logger.info("Endpoint deleted", endpoint=endpoint_name)
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {str(e)}")
            return False


# Factory function
def create_vertex_ai_client() -> VertexAIClient:
    """Create configured Vertex AI client"""
    return VertexAIClient()