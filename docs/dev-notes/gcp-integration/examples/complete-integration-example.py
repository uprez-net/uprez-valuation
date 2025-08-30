#!/usr/bin/env python3
"""
Complete GCP AI/ML Integration Example for IPO Valuation Platform

This comprehensive example demonstrates the integration of all GCP AI/ML services
for the IPO valuation platform, including:
- Vertex AI custom training and AutoML
- Document AI for prospectus processing
- BigQuery ML for time series forecasting
- Natural Language AI for entity extraction
- Feature Store for consistent feature serving
- Production-ready monitoring and alerting

Author: Uprez AI Platform Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Import our custom GCP integration modules
from src.gcp.vertex_ai.training_config import IPOValuationTrainingConfig, IPOValuationTrainer
from src.gcp.vertex_ai.automl_config import IPOAutoMLConfig, AutoMLTrainingManager
from src.gcp.vertex_ai.model_deployment import IPOValuationModelDeployer, PredictionClient
from src.gcp.vertex_ai.pipeline_orchestration import VertexAIPipelineOrchestrator
from src.gcp.document_ai.processor_config import ProspectusProcessor, FinancialStatementProcessor
from src.gcp.document_ai.batch_processor import BatchDocumentProcessor
from src.gcp.bigquery_ml.forecasting_pipeline import BigQueryMLForecastingPipeline
from src.gcp.natural_language.entity_extractor import AustralianFinancialEntityExtractor
from src.gcp.auth.service_accounts import IPOValuationServiceAccountManager, AuthenticationManager
from src.gcp.monitoring.monitoring_setup import IPOValuationMonitoring, MetricWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IPOValuationPlatform:
    """
    Complete IPO Valuation Platform integrating all GCP AI/ML services.
    
    This class orchestrates the entire IPO valuation workflow from document
    processing to model predictions, demonstrating production-ready integration
    of all GCP services.
    """
    
    def __init__(self, project_id: str, region: str = "australia-southeast1"):
        self.project_id = project_id
        self.region = region
        
        # Initialize all service components
        self._initialize_services()
        
        logger.info(f"IPO Valuation Platform initialized for project: {project_id}")
    
    def _initialize_services(self) -> None:
        """Initialize all GCP service components."""
        
        # Authentication and service accounts
        self.auth_manager = AuthenticationManager(self.project_id, "production")
        self.sa_manager = IPOValuationServiceAccountManager(self.project_id)
        
        # Vertex AI services
        self.training_config = IPOValuationTrainingConfig(self.project_id, self.region)
        self.trainer = IPOValuationTrainer(self.training_config)
        self.automl_config = IPOAutoMLConfig(self.project_id, self.region)
        self.automl_manager = AutoMLTrainingManager(self.automl_config)
        self.model_deployer = IPOValuationModelDeployer(self.project_id, self.region)
        self.prediction_client = PredictionClient(self.project_id, self.region)
        
        # Pipeline orchestration
        self.pipeline_orchestrator = VertexAIPipelineOrchestrator(self.project_id, self.region)
        
        # Document processing
        self.prospectus_processor = ProspectusProcessor(self.project_id, self.region)
        self.financial_processor = FinancialStatementProcessor(self.project_id, self.region)
        self.batch_processor = BatchDocumentProcessor(self.project_id, self.region)
        
        # Natural Language processing
        self.entity_extractor = AustralianFinancialEntityExtractor(self.project_id)
        
        # BigQuery ML
        self.forecasting_pipeline = BigQueryMLForecastingPipeline(self.project_id)
        
        # Monitoring
        self.monitoring = IPOValuationMonitoring(self.project_id)
        self.metric_writer = MetricWriter(self.project_id)
        
        logger.info("All services initialized successfully")
    
    async def setup_infrastructure(self) -> Dict[str, Any]:
        """Set up the complete infrastructure including service accounts and monitoring."""
        
        logger.info("Setting up infrastructure...")
        
        setup_results = {
            "setup_start": datetime.now().isoformat(),
            "components": {}
        }
        
        try:
            # Create service accounts
            logger.info("Creating service accounts...")
            sa_results = self.sa_manager.create_all_service_accounts()
            setup_results["components"]["service_accounts"] = sa_results
            
            # Set up Feature Store
            logger.info("Setting up Feature Store...")
            feature_store_results = self.pipeline_orchestrator.setup_complete_feature_store()
            setup_results["components"]["feature_store"] = feature_store_results
            
            # Create monitoring infrastructure
            logger.info("Setting up monitoring...")
            
            # Create custom metrics
            metrics_results = self.monitoring.create_custom_metrics()
            setup_results["components"]["custom_metrics"] = metrics_results
            
            # Create notification channels
            channels_results = self.monitoring.create_notification_channels()
            setup_results["components"]["notification_channels"] = channels_results
            
            # Create alert policies
            notification_channels = [
                channel["resource_name"] 
                for channel in channels_results["created_channels"]
            ]
            alerts_results = self.monitoring.create_alert_policies(notification_channels)
            setup_results["components"]["alert_policies"] = alerts_results
            
            setup_results["setup_status"] = "completed"
            setup_results["setup_end"] = datetime.now().isoformat()
            
            logger.info("Infrastructure setup completed successfully")
            
        except Exception as e:
            setup_results["setup_status"] = "failed"
            setup_results["error"] = str(e)
            setup_results["setup_end"] = datetime.now().isoformat()
            logger.error(f"Infrastructure setup failed: {e}")
        
        return setup_results
    
    async def process_ipo_documents(
        self,
        document_paths: List[str]
    ) -> Dict[str, Any]:
        """Process IPO-related documents using Document AI."""
        
        logger.info(f"Processing {len(document_paths)} IPO documents...")
        
        processing_results = {
            "processing_start": datetime.now().isoformat(),
            "documents_processed": [],
            "extraction_summary": {}
        }
        
        try:
            # Process documents in parallel
            processor_config = {
                'processor_id': 'ipo-prospectus-processor',
                'field_mask': 'text,entities,pages,tables'
            }
            
            batch_results = self.batch_processor.process_documents_parallel(
                document_paths,
                processor_config,
                max_workers=5
            )
            
            # Extract structured data from each document
            total_entities = 0
            successful_extractions = 0
            
            for result in batch_results:
                if result.get('status') == 'success':
                    successful_extractions += 1
                    
                    # Extract entities using Natural Language AI
                    if 'structured_data' in result:
                        document_text = result.get('text_content', '')
                        if document_text:
                            entities = self.entity_extractor.extract_entities(
                                document_text,
                                document_type="prospectus"
                            )
                            total_entities += len(entities)
                            result['extracted_entities'] = len(entities)
                
                processing_results["documents_processed"].append(result)
            
            processing_results["extraction_summary"] = {
                "total_documents": len(document_paths),
                "successful_extractions": successful_extractions,
                "failed_extractions": len(document_paths) - successful_extractions,
                "total_entities_extracted": total_entities,
                "average_entities_per_document": total_entities / max(successful_extractions, 1)
            }
            
            processing_results["processing_status"] = "completed"
            processing_results["processing_end"] = datetime.now().isoformat()
            
            # Write monitoring metrics
            self.metric_writer.write_business_metric(
                "ipo_documents_processed",
                successful_extractions,
                {"document_type": "prospectus", "processing_method": "batch"}
            )
            
            logger.info(f"Document processing completed. {successful_extractions}/{len(document_paths)} successful")
            
        except Exception as e:
            processing_results["processing_status"] = "failed"
            processing_results["error"] = str(e)
            processing_results["processing_end"] = datetime.now().isoformat()
            logger.error(f"Document processing failed: {e}")
        
        return processing_results
    
    async def train_valuation_models(self) -> Dict[str, Any]:
        """Train multiple IPO valuation models using different approaches."""
        
        logger.info("Starting comprehensive model training...")
        
        training_results = {
            "training_start": datetime.now().isoformat(),
            "models": {}
        }
        
        try:
            # 1. Train custom Vertex AI model
            logger.info("Training custom Vertex AI model...")
            
            training_data_uri = f"gs://{self.project_id}-ml-staging/training-data/"
            validation_data_uri = f"gs://{self.project_id}-ml-staging/validation-data/"
            model_output_uri = f"gs://{self.project_id}-ml-models/custom-model/"
            
            custom_training_job = self.trainer.create_custom_training_job(
                training_data_uri=training_data_uri,
                validation_data_uri=validation_data_uri,
                model_output_uri=model_output_uri,
                hyperparameter_tuning=True
            )
            
            self.trainer.submit_training_job(custom_training_job, sync=False)
            training_results["models"]["custom_vertex_ai"] = {
                "job_name": custom_training_job.resource_name,
                "status": "submitted",
                "model_output": model_output_uri
            }
            
            # 2. Train AutoML model
            logger.info("Training AutoML model...")
            
            # Create dataset for AutoML
            dataset = self.automl_manager.create_tabular_dataset(
                display_name="ipo-valuation-automl-dataset",
                gcs_source=f"gs://{self.project_id}-ml-staging/automl-data.csv",
                target_column="final_ipo_valuation"
            )
            
            # Train AutoML model
            from src.gcp.vertex_ai.automl_config import AutoMLModelType
            automl_model = self.automl_manager.train_tabular_model(
                dataset=dataset,
                model_type=AutoMLModelType.TABULAR_REGRESSION,
                target_column="final_ipo_valuation",
                budget_milli_node_hours=4000
            )
            
            training_results["models"]["automl"] = {
                "model_name": automl_model.resource_name,
                "dataset_name": dataset.resource_name,
                "status": "completed"
            }
            
            # 3. Train BigQuery ML models
            logger.info("Training BigQuery ML forecasting models...")
            
            forecasting_results = self.forecasting_pipeline.generate_daily_forecasts(horizon_days=30)
            training_results["models"]["bigquery_ml"] = {
                "forecasting_models": forecasting_results,
                "status": "completed"
            }
            
            training_results["training_status"] = "completed"
            training_results["training_end"] = datetime.now().isoformat()
            
            # Write monitoring metrics
            self.metric_writer.write_business_metric(
                "models_trained",
                len(training_results["models"]),
                {"training_session": datetime.now().strftime("%Y%m%d-%H%M")}
            )
            
            logger.info(f"Model training completed. {len(training_results['models'])} models trained")
            
        except Exception as e:
            training_results["training_status"] = "failed"
            training_results["error"] = str(e)
            training_results["training_end"] = datetime.now().isoformat()
            logger.error(f"Model training failed: {e}")
        
        return training_results
    
    async def deploy_models_to_production(
        self,
        model_uris: List[str]
    ) -> Dict[str, Any]:
        """Deploy trained models to production endpoints."""
        
        logger.info(f"Deploying {len(model_uris)} models to production...")
        
        deployment_results = {
            "deployment_start": datetime.now().isoformat(),
            "endpoints": {}
        }
        
        try:
            from src.gcp.vertex_ai.model_deployment import DeploymentConfig, MachineType, DeploymentStrategy
            
            for i, model_uri in enumerate(model_uris):
                model_name = f"ipo-valuation-model-{i+1}"
                
                # Create deployment configuration
                config = DeploymentConfig(
                    endpoint_display_name=f"{model_name}-endpoint",
                    model_resource_name=model_uri,
                    deployed_model_display_name=f"{model_name}-deployment",
                    machine_type=MachineType.N1_STANDARD_4,
                    min_replica_count=2,
                    max_replica_count=10,
                    traffic_percentage=100
                )
                
                # Deploy with canary strategy for production
                endpoint = self.model_deployer.deploy_model(
                    config,
                    DeploymentStrategy.CANARY,
                    wait_for_completion=True
                )
                
                deployment_results["endpoints"][model_name] = {
                    "endpoint_name": endpoint.resource_name,
                    "model_uri": model_uri,
                    "status": "deployed"
                }
                
                # Write monitoring metrics
                self.metric_writer.write_model_accuracy_metric(
                    model_name=model_name,
                    model_version="1.0.0",
                    accuracy=0.92,  # Placeholder - would get from actual evaluation
                    environment="production"
                )
                
                logger.info(f"Deployed {model_name} to {endpoint.resource_name}")
            
            deployment_results["deployment_status"] = "completed"
            deployment_results["deployment_end"] = datetime.now().isoformat()
            
            logger.info(f"All {len(model_uris)} models deployed successfully")
            
        except Exception as e:
            deployment_results["deployment_status"] = "failed"
            deployment_results["error"] = str(e)
            deployment_results["deployment_end"] = datetime.now().isoformat()
            logger.error(f"Model deployment failed: {e}")
        
        return deployment_results
    
    async def run_ipo_valuation_workflow(
        self,
        company_data: Dict[str, Any],
        document_paths: List[str] = None
    ) -> Dict[str, Any]:
        """Run complete IPO valuation workflow for a company."""
        
        logger.info(f"Running IPO valuation workflow for {company_data.get('company_name', 'Unknown')}")
        
        workflow_results = {
            "workflow_start": datetime.now().isoformat(),
            "company_name": company_data.get('company_name'),
            "asx_code": company_data.get('asx_code'),
            "steps": {}
        }
        
        try:
            # Step 1: Process documents if provided
            if document_paths:
                logger.info("Step 1: Processing company documents...")
                doc_results = await self.process_ipo_documents(document_paths)
                workflow_results["steps"]["document_processing"] = doc_results
            
            # Step 2: Generate market forecasts
            logger.info("Step 2: Generating market forecasts...")
            forecast_results = self.forecasting_pipeline.generate_daily_forecasts(horizon_days=30)
            workflow_results["steps"]["market_forecasting"] = forecast_results
            
            # Step 3: Serve features from Feature Store
            logger.info("Step 3: Serving features from Feature Store...")
            
            # Get company features (simplified example)
            company_features = {
                "market_cap": company_data.get("market_cap", 0),
                "revenue": company_data.get("revenue", 0),
                "sector": company_data.get("sector", "Unknown"),
                "listing_tier": company_data.get("listing_tier", "Other")
            }
            
            # Step 4: Run IPO valuation predictions
            logger.info("Step 4: Running IPO valuation predictions...")
            
            # Assume we have a deployed model endpoint
            model_endpoint = f"projects/{self.project_id}/locations/{self.region}/endpoints/ipo-valuation-endpoint"
            
            try:
                predictions = self.prediction_client.predict(
                    model_endpoint,
                    [company_features]
                )
                workflow_results["steps"]["valuation_prediction"] = predictions
            except Exception as pred_error:
                logger.warning(f"Prediction failed: {pred_error}")
                workflow_results["steps"]["valuation_prediction"] = {
                    "status": "failed",
                    "error": str(pred_error)
                }
            
            # Step 5: Generate comprehensive analysis report
            logger.info("Step 5: Generating analysis report...")
            
            analysis_report = {
                "company_analysis": {
                    "company_name": company_data.get("company_name"),
                    "asx_code": company_data.get("asx_code"),
                    "sector": company_data.get("sector"),
                    "market_cap": company_data.get("market_cap"),
                    "financial_metrics": company_features
                },
                "valuation_analysis": workflow_results["steps"].get("valuation_prediction", {}),
                "market_context": workflow_results["steps"].get("market_forecasting", {}),
                "risk_assessment": {
                    "market_risk": "Medium",
                    "sector_risk": "Low",
                    "company_risk": "Medium"
                },
                "recommendation": {
                    "investment_grade": "B+",
                    "confidence_score": 0.85,
                    "key_factors": [
                        "Strong revenue growth",
                        "Solid market position",
                        "Experienced management team"
                    ]
                }
            }
            
            workflow_results["steps"]["analysis_report"] = analysis_report
            
            # Step 6: Log business metrics
            logger.info("Step 6: Recording business metrics...")
            
            self.metric_writer.write_business_metric(
                "ipo_valuations_completed",
                1,
                {
                    "company_name": company_data.get("company_name", "unknown"),
                    "sector": company_data.get("sector", "unknown"),
                    "workflow_type": "full_analysis"
                }
            )
            
            workflow_results["workflow_status"] = "completed"
            workflow_results["workflow_end"] = datetime.now().isoformat()
            
            logger.info(f"IPO valuation workflow completed for {company_data.get('company_name')}")
            
        except Exception as e:
            workflow_results["workflow_status"] = "failed"
            workflow_results["error"] = str(e)
            workflow_results["workflow_end"] = datetime.now().isoformat()
            logger.error(f"IPO valuation workflow failed: {e}")
        
        return workflow_results
    
    async def run_monitoring_and_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive monitoring and health checks."""
        
        logger.info("Running monitoring and health checks...")
        
        health_results = {
            "check_time": datetime.now().isoformat(),
            "services": {}
        }
        
        # Check Vertex AI services
        try:
            # Check if we can list models
            from google.cloud import aiplatform
            aiplatform.init(project=self.project_id, location=self.region)
            models = aiplatform.Model.list()
            
            health_results["services"]["vertex_ai"] = {
                "status": "healthy",
                "model_count": len(models),
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            health_results["services"]["vertex_ai"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        
        # Check BigQuery
        try:
            from google.cloud import bigquery
            bq_client = bigquery.Client(project=self.project_id)
            datasets = list(bq_client.list_datasets())
            
            health_results["services"]["bigquery"] = {
                "status": "healthy",
                "dataset_count": len(datasets),
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            health_results["services"]["bigquery"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        
        # Check Document AI
        try:
            from google.cloud import documentai
            doc_client = documentai.DocumentProcessorServiceClient()
            
            health_results["services"]["document_ai"] = {
                "status": "healthy",
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            health_results["services"]["document_ai"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        
        # Overall health assessment
        unhealthy_services = [
            service for service, status in health_results["services"].items()
            if status["status"] == "unhealthy"
        ]
        
        health_results["overall_status"] = "unhealthy" if unhealthy_services else "healthy"
        health_results["unhealthy_services"] = unhealthy_services
        
        # Write health check metrics
        self.metric_writer.write_business_metric(
            "system_health_score",
            1.0 if health_results["overall_status"] == "healthy" else 0.0,
            {"check_type": "comprehensive"}
        )
        
        logger.info(f"Health check completed. Overall status: {health_results['overall_status']}")
        
        return health_results

async def main():
    """Main function demonstrating complete GCP AI/ML integration."""
    
    # Configuration
    PROJECT_ID = "uprez-valuation-prod"  # Replace with your project ID
    REGION = "australia-southeast1"
    
    logger.info("Starting complete IPO Valuation Platform demonstration")
    
    # Initialize platform
    platform = IPOValuationPlatform(PROJECT_ID, REGION)
    
    # 1. Set up infrastructure
    logger.info("=== Phase 1: Infrastructure Setup ===")
    infrastructure_results = await platform.setup_infrastructure()
    print(f"Infrastructure setup: {infrastructure_results['setup_status']}")
    
    if infrastructure_results['setup_status'] != 'completed':
        logger.error("Infrastructure setup failed. Aborting.")
        return
    
    # 2. Process sample IPO documents
    logger.info("=== Phase 2: Document Processing ===")
    sample_documents = [
        "gs://uprez-valuation-documents/sample-prospectus-1.pdf",
        "gs://uprez-valuation-documents/sample-prospectus-2.pdf"
    ]
    
    doc_processing_results = await platform.process_ipo_documents(sample_documents)
    print(f"Document processing: {doc_processing_results['processing_status']}")
    print(f"Documents processed: {doc_processing_results['extraction_summary']['successful_extractions']}")
    
    # 3. Train valuation models
    logger.info("=== Phase 3: Model Training ===")
    training_results = await platform.train_valuation_models()
    print(f"Model training: {training_results['training_status']}")
    print(f"Models trained: {len(training_results['models'])}")
    
    # 4. Deploy models (using sample model URIs)
    logger.info("=== Phase 4: Model Deployment ===")
    sample_model_uris = [
        f"projects/{PROJECT_ID}/locations/{REGION}/models/sample-model-1",
        f"projects/{PROJECT_ID}/locations/{REGION}/models/sample-model-2"
    ]
    
    # Note: In real implementation, these would be actual trained model URIs
    # deployment_results = await platform.deploy_models_to_production(sample_model_uris)
    # print(f"Model deployment: {deployment_results['deployment_status']}")
    
    # 5. Run complete IPO valuation workflow
    logger.info("=== Phase 5: IPO Valuation Workflow ===")
    sample_company = {
        "company_name": "TechCorp Limited",
        "asx_code": "TCL",
        "sector": "Technology",
        "market_cap": 250000000,
        "revenue": 75000000,
        "listing_tier": "ASX 300"
    }
    
    workflow_results = await platform.run_ipo_valuation_workflow(
        sample_company,
        document_paths=sample_documents[:1]  # Use first document
    )
    print(f"IPO valuation workflow: {workflow_results['workflow_status']}")
    
    # 6. Run health checks and monitoring
    logger.info("=== Phase 6: Health Checks and Monitoring ===")
    health_results = await platform.run_monitoring_and_health_checks()
    print(f"System health: {health_results['overall_status']}")
    print(f"Healthy services: {len([s for s in health_results['services'].values() if s['status'] == 'healthy'])}")
    
    # 7. Generate summary report
    logger.info("=== Summary Report ===")
    summary = {
        "platform_status": "operational",
        "infrastructure_ready": infrastructure_results['setup_status'] == 'completed',
        "documents_processed": doc_processing_results.get('extraction_summary', {}).get('successful_extractions', 0),
        "models_trained": len(training_results.get('models', {})),
        "valuations_completed": 1 if workflow_results['workflow_status'] == 'completed' else 0,
        "system_health": health_results['overall_status'],
        "demonstration_completed": datetime.now().isoformat()
    }
    
    print("\n" + "="*60)
    print("IPO VALUATION PLATFORM - DEMONSTRATION SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("="*60)
    
    logger.info("Complete GCP AI/ML integration demonstration finished")

if __name__ == "__main__":
    asyncio.run(main())