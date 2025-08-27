"""
Model Serving Pipeline for ML-powered IPO Valuation
Real-time and batch prediction capabilities with monitoring and caching
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import pickle
import joblib
from pathlib import Path

# Caching and storage
import redis
from sqlalchemy import create_engine, text
import sqlite3

# API and web serving
from fastapi import HTTPException
from pydantic import BaseModel, Field

# ML model libraries
from sklearn.base import BaseEstimator
import torch
import onnx
import onnxruntime as ort

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class PredictionRequest:
    """Request for model prediction"""
    request_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest())
    model_name: str = ""
    model_version: str = "latest"
    inputs: Dict[str, Any] = field(default_factory=dict)
    prediction_type: str = "real_time"  # real_time, batch, streaming
    
    # Request metadata
    client_id: Optional[str] = None
    priority: int = 1  # 1=low, 5=high
    timeout_seconds: int = 30
    cache_ttl: int = 3600  # Cache TTL in seconds
    enable_caching: bool = True
    
    # Feature flags
    enable_monitoring: bool = True
    enable_explanation: bool = False
    enable_confidence_intervals: bool = True
    
    # Validation options
    validate_inputs: bool = True
    auto_retry: bool = True
    max_retries: int = 3

@dataclass
class PredictionResponse:
    """Response from model prediction"""
    request_id: str
    prediction: Dict[str, Any]
    model_info: Dict[str, str]
    
    # Performance metrics
    prediction_time_ms: float
    model_load_time_ms: float = 0
    preprocessing_time_ms: float = 0
    inference_time_ms: float = 0
    postprocessing_time_ms: float = 0
    
    # Prediction metadata
    confidence_score: float = 0.0
    prediction_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Quality indicators
    data_quality_score: float = 1.0
    model_drift_score: float = 0.0
    prediction_uncertainty: float = 0.0
    
    # Caching and versioning
    cached: bool = False
    cache_hit_rate: float = 0.0
    model_version: str = "latest"
    
    # Explanation (if requested)
    explanation: Optional[Dict[str, Any]] = None
    
    # Error handling
    warnings: List[str] = field(default_factory=list)
    status: str = "success"  # success, partial, error

@dataclass
class BatchPredictionJob:
    """Batch prediction job configuration"""
    job_id: str
    model_name: str
    input_data: Union[str, pd.DataFrame]  # File path or DataFrame
    output_path: str
    
    # Job configuration
    batch_size: int = 1000
    parallel_workers: int = 4
    priority: int = 1
    
    # Processing options
    chunk_processing: bool = True
    enable_progress_tracking: bool = True
    save_intermediate_results: bool = True
    
    # Quality controls
    data_validation: bool = True
    outlier_detection: bool = True
    confidence_threshold: float = 0.5
    
    # Job metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    progress_percentage: float = 0.0
    
    # Results summary
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    average_confidence: float = 0.0

class ModelCache:
    """High-performance model caching with Redis backend"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()  # Test connection
            self.cache_enabled = True
        except Exception as e:
            logger.warning(f"Redis connection failed, disabling cache: {e}")
            self.cache_enabled = False
            self.local_cache = {}
    
    async def get_prediction(self, cache_key: str) -> Optional[Dict]:
        """Get cached prediction result"""
        if not self.cache_enabled:
            return None
        
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def set_prediction(self, cache_key: str, prediction: Dict, ttl: int = 3600):
        """Cache prediction result"""
        if not self.cache_enabled:
            return
        
        try:
            self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(prediction, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key from request"""
        key_data = {
            'model_name': request.model_name,
            'model_version': request.model_version,
            'inputs_hash': hashlib.md5(str(sorted(request.inputs.items())).encode()).hexdigest()
        }
        return f"prediction:{hashlib.md5(str(key_data).encode()).hexdigest()}"

class ModelRegistry:
    """Model registry for managing multiple valuation models"""
    
    def __init__(self, models_directory: str = "models/"):
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.model_metadata = {}
        
        # Initialize model type mappings
        self.model_types = {
            'dcf': 'AdvancedDCFModel',
            'cca': 'EnhancedCCAModel',
            'risk': 'MultiFactorRiskModel',
            'sentiment': 'MarketSentimentAnalyzer',
            'ensemble': 'EnsembleValuationModel',
            'financial_analytics': 'AdvancedFinancialAnalytics'
        }
    
    async def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load model from registry"""
        model_key = f"{model_name}:{version}"
        
        # Return cached model if already loaded
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        # Load model from storage
        model_path = self.models_directory / model_name / version
        
        try:
            # Try different loading methods
            if (model_path / "model.pkl").exists():
                model = joblib.load(model_path / "model.pkl")
            elif (model_path / "model.onnx").exists():
                model = ort.InferenceSession(str(model_path / "model.onnx"))
            else:
                # Create new model instance
                model = await self._create_model_instance(model_name)
            
            self.loaded_models[model_key] = model
            
            # Load metadata
            metadata_path = model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata[model_key] = json.load(f)
            
            logger.info(f"Loaded model: {model_key}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
    
    async def _create_model_instance(self, model_name: str):
        """Create new model instance"""
        if model_name == 'dcf':
            from .advanced_dcf import create_advanced_dcf_model
            return create_advanced_dcf_model()
        elif model_name == 'cca':
            from .comparable_company_analysis import create_enhanced_cca_model
            return create_enhanced_cca_model()
        elif model_name == 'risk':
            from .risk_assessment import create_multi_factor_risk_model
            return create_multi_factor_risk_model()
        elif model_name == 'sentiment':
            from .market_sentiment import create_market_sentiment_analyzer
            return create_market_sentiment_analyzer()
        elif model_name == 'ensemble':
            from .ensemble_framework import create_ensemble_valuation_model
            return create_ensemble_valuation_model()
        elif model_name == 'financial_analytics':
            from .financial_analytics import create_advanced_financial_analytics
            return create_advanced_financial_analytics()
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    async def register_model(self, model_name: str, model: Any, version: str = "latest", metadata: Dict = None):
        """Register a new model version"""
        model_key = f"{model_name}:{version}"
        
        # Store model in memory
        self.loaded_models[model_key] = model
        
        # Store metadata
        if metadata:
            self.model_metadata[model_key] = metadata
        
        # Persist to disk
        model_dir = self.models_directory / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model
            joblib.dump(model, model_dir / "model.pkl")
            
            # Save metadata
            if metadata:
                with open(model_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Registered model: {model_key}")
            
        except Exception as e:
            logger.warning(f"Failed to persist model {model_key}: {e}")
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all available models and versions"""
        models = {}
        
        # From loaded models
        for model_key in self.loaded_models.keys():
            name, version = model_key.split(':', 1)
            if name not in models:
                models[name] = []
            if version not in models[name]:
                models[name].append(version)
        
        # From disk
        if self.models_directory.exists():
            for model_dir in self.models_directory.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    if model_name not in models:
                        models[model_name] = []
                    
                    for version_dir in model_dir.iterdir():
                        if version_dir.is_dir():
                            version = version_dir.name
                            if version not in models[model_name]:
                                models[model_name].append(version)
        
        return models
    
    def get_model_metadata(self, model_name: str, version: str = "latest") -> Dict:
        """Get model metadata"""
        model_key = f"{model_name}:{version}"
        return self.model_metadata.get(model_key, {})

class ModelServingPipeline:
    """
    Comprehensive Model Serving Pipeline
    
    Features:
    - Real-time prediction serving
    - Batch processing capabilities
    - Model caching and registry
    - Performance monitoring
    - Load balancing and scaling
    - Input validation and preprocessing
    - Error handling and retries
    """
    
    def __init__(
        self,
        models_directory: str = "models/",
        cache_enabled: bool = True,
        monitoring_enabled: bool = True,
        max_workers: int = 4
    ):
        self.model_registry = ModelRegistry(models_directory)
        self.cache = ModelCache() if cache_enabled else None
        self.monitoring_enabled = monitoring_enabled
        self.max_workers = max_workers
        
        # Thread pools for different types of work
        self.prediction_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_executor = ProcessPoolExecutor(max_workers=max_workers//2)
        
        # Batch job tracking
        self.batch_jobs = {}
        
        # Performance metrics (Prometheus)
        if monitoring_enabled:
            self.prediction_counter = Counter('model_predictions_total', 'Total predictions', ['model', 'status'])
            self.prediction_duration = Histogram('model_prediction_duration_seconds', 'Prediction duration', ['model'])
            self.model_load_duration = Histogram('model_load_duration_seconds', 'Model load duration', ['model'])
            self.cache_hit_rate = Gauge('prediction_cache_hit_rate', 'Cache hit rate', ['model'])
        
        # Initialize built-in models
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self):
        """Initialize commonly used models"""
        try:
            # Pre-load frequently used models
            await self.model_registry.load_model('ensemble')
            logger.info("Initialized ensemble model")
        except Exception as e:
            logger.warning(f"Failed to initialize models: {e}")
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make real-time prediction
        """
        start_time = time.time()
        
        try:
            # Input validation
            if request.validate_inputs:
                await self._validate_request(request)
            
            # Check cache
            cache_key = None
            cached_result = None
            if self.cache and request.enable_caching:
                cache_key = self.cache.generate_cache_key(request)
                cached_result = await self.cache.get_prediction(cache_key)
                
                if cached_result:
                    logger.info(f"Cache hit for request {request.request_id}")
                    return PredictionResponse(
                        request_id=request.request_id,
                        prediction=cached_result['prediction'],
                        model_info=cached_result['model_info'],
                        prediction_time_ms=(time.time() - start_time) * 1000,
                        cached=True,
                        status="success"
                    )
            
            # Load model
            model_load_start = time.time()
            model = await self.model_registry.load_model(request.model_name, request.model_version)
            model_load_time = (time.time() - model_load_start) * 1000
            
            # Preprocessing
            preprocessing_start = time.time()
            processed_inputs = await self._preprocess_inputs(request.inputs, request.model_name)
            preprocessing_time = (time.time() - preprocessing_start) * 1000
            
            # Inference
            inference_start = time.time()
            prediction_result = await self._run_inference(model, processed_inputs, request.model_name)
            inference_time = (time.time() - inference_start) * 1000
            
            # Postprocessing
            postprocessing_start = time.time()
            final_prediction = await self._postprocess_outputs(prediction_result, request)
            postprocessing_time = (time.time() - postprocessing_start) * 1000
            
            # Create response
            total_time = (time.time() - start_time) * 1000
            
            response = PredictionResponse(
                request_id=request.request_id,
                prediction=final_prediction,
                model_info={
                    'name': request.model_name,
                    'version': request.model_version,
                    'type': type(model).__name__
                },
                prediction_time_ms=total_time,
                model_load_time_ms=model_load_time,
                preprocessing_time_ms=preprocessing_time,
                inference_time_ms=inference_time,
                postprocessing_time_ms=postprocessing_time,
                model_version=request.model_version,
                status="success"
            )
            
            # Cache result
            if self.cache and request.enable_caching and cache_key:
                await self.cache.set_prediction(
                    cache_key,
                    {
                        'prediction': final_prediction,
                        'model_info': response.model_info
                    },
                    request.cache_ttl
                )
            
            # Update metrics
            if self.monitoring_enabled:
                self.prediction_counter.labels(model=request.model_name, status='success').inc()
                self.prediction_duration.labels(model=request.model_name).observe(total_time / 1000)
            
            logger.info(f"Prediction completed for {request.request_id} in {total_time:.2f}ms")
            return response
            
        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            logger.error(f"Prediction failed for {request.request_id}: {e}")
            
            # Update error metrics
            if self.monitoring_enabled:
                self.prediction_counter.labels(model=request.model_name, status='error').inc()
            
            # Retry logic
            if request.auto_retry and hasattr(request, '_retry_count'):
                request._retry_count = getattr(request, '_retry_count', 0) + 1
                if request._retry_count < request.max_retries:
                    logger.info(f"Retrying prediction for {request.request_id} (attempt {request._retry_count + 1})")
                    await asyncio.sleep(0.5 * request._retry_count)  # Exponential backoff
                    return await self.predict(request)
            
            return PredictionResponse(
                request_id=request.request_id,
                prediction={},
                model_info={'name': request.model_name, 'error': str(e)},
                prediction_time_ms=error_time,
                status="error",
                warnings=[str(e)]
            )
    
    async def start_batch_job(self, job: BatchPredictionJob) -> str:
        """
        Start batch prediction job
        """
        try:
            # Validate job
            await self._validate_batch_job(job)
            
            # Store job
            self.batch_jobs[job.job_id] = job
            job.status = "queued"
            job.started_at = datetime.now()
            
            # Submit to executor
            future = self.batch_executor.submit(self._run_batch_job, job)
            
            logger.info(f"Started batch job {job.job_id}")
            return job.job_id
            
        except Exception as e:
            job.status = "failed"
            logger.error(f"Failed to start batch job {job.job_id}: {e}")
            raise
    
    def _run_batch_job(self, job: BatchPredictionJob):
        """
        Run batch prediction job (executed in process pool)
        """
        try:
            job.status = "running"
            
            # Load data
            if isinstance(job.input_data, str):
                if job.input_data.endswith('.csv'):
                    data = pd.read_csv(job.input_data)
                elif job.input_data.endswith('.parquet'):
                    data = pd.read_parquet(job.input_data)
                else:
                    raise ValueError(f"Unsupported file format: {job.input_data}")
            else:
                data = job.input_data
            
            job.total_records = len(data)
            
            # Process in chunks
            results = []
            processed_count = 0
            
            for i in range(0, len(data), job.batch_size):
                chunk = data.iloc[i:i + job.batch_size]
                
                # Process chunk
                chunk_results = self._process_batch_chunk(chunk, job)
                results.extend(chunk_results)
                
                processed_count += len(chunk)
                job.processed_records = processed_count
                job.progress_percentage = (processed_count / job.total_records) * 100
                
                logger.info(f"Batch job {job.job_id} progress: {job.progress_percentage:.1f}%")
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(job.output_path, index=False)
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now()
            job.average_confidence = results_df['confidence_score'].mean() if 'confidence_score' in results_df.columns else 0
            
            logger.info(f"Completed batch job {job.job_id}")
            
        except Exception as e:
            job.status = "failed"
            logger.error(f"Batch job {job.job_id} failed: {e}")
            raise
    
    def _process_batch_chunk(self, chunk: pd.DataFrame, job: BatchPredictionJob) -> List[Dict]:
        """Process a batch chunk"""
        results = []
        
        # This would need to be adapted based on the specific model
        # For now, return placeholder results
        
        for _, row in chunk.iterrows():
            result = {
                'id': row.get('id', ''),
                'prediction': 100.0,  # Placeholder
                'confidence_score': 0.8,
                'model_name': job.model_name
            }
            results.append(result)
        
        return results
    
    async def get_batch_job_status(self, job_id: str) -> Optional[BatchPredictionJob]:
        """Get batch job status"""
        return self.batch_jobs.get(job_id)
    
    async def cancel_batch_job(self, job_id: str) -> bool:
        """Cancel batch job"""
        if job_id in self.batch_jobs:
            job = self.batch_jobs[job_id]
            if job.status in ["queued", "running"]:
                job.status = "cancelled"
                return True
        return False
    
    async def _validate_request(self, request: PredictionRequest):
        """Validate prediction request"""
        if not request.model_name:
            raise ValueError("Model name is required")
        
        if not request.inputs:
            raise ValueError("Inputs are required")
        
        # Model-specific validation would go here
        
    async def _validate_batch_job(self, job: BatchPredictionJob):
        """Validate batch job configuration"""
        if not job.model_name:
            raise ValueError("Model name is required")
        
        if isinstance(job.input_data, str):
            if not Path(job.input_data).exists():
                raise ValueError(f"Input file does not exist: {job.input_data}")
        elif not isinstance(job.input_data, pd.DataFrame):
            raise ValueError("Input data must be file path or DataFrame")
        
        # Validate output path
        output_dir = Path(job.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    
    async def _preprocess_inputs(self, inputs: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Preprocess inputs based on model requirements"""
        
        # Model-specific preprocessing
        if model_name == 'dcf':
            return await self._preprocess_dcf_inputs(inputs)
        elif model_name == 'cca':
            return await self._preprocess_cca_inputs(inputs)
        elif model_name == 'risk':
            return await self._preprocess_risk_inputs(inputs)
        elif model_name == 'sentiment':
            return await self._preprocess_sentiment_inputs(inputs)
        else:
            return inputs
    
    async def _preprocess_dcf_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess DCF model inputs"""
        # Convert to DCF input format
        # This would create AdvancedDCFInputs object
        return inputs
    
    async def _preprocess_cca_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess CCA model inputs"""
        return inputs
    
    async def _preprocess_risk_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess risk model inputs"""
        return inputs
    
    async def _preprocess_sentiment_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess sentiment model inputs"""
        return inputs
    
    async def _run_inference(self, model: Any, inputs: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Run model inference"""
        
        if model_name == 'dcf':
            # Run DCF model
            result = await model.calculate_advanced_valuation(inputs)
            return {
                'enterprise_value': result.enterprise_value,
                'equity_value': result.equity_value,
                'value_per_share': result.value_per_share
            }
        elif model_name == 'cca':
            # Run CCA model
            result = await model.analyze_comparable_companies(inputs.get('target_company'), inputs.get('universe'), inputs.get('criteria'))
            return {
                'implied_valuations': result.implied_valuations,
                'peer_analysis': result.peer_cluster_analysis
            }
        elif model_name == 'risk':
            # Run risk model
            result = await model.assess_comprehensive_risk(inputs)
            return {
                'composite_risk_score': result.composite_risk_score,
                'risk_grade': result.risk_grade,
                'category_scores': result.category_scores
            }
        elif model_name == 'sentiment':
            # Run sentiment model
            result = await model.analyze_market_sentiment(inputs)
            return {
                'composite_sentiment_score': result.composite_sentiment_score,
                'sentiment_confidence': result.sentiment_confidence,
                'sentiment_trend': result.sentiment_trend
            }
        elif model_name == 'ensemble':
            # Run ensemble model
            result = await model.predict_ensemble_valuation(inputs)
            return {
                'ensemble_valuation': result.ensemble_valuation,
                'confidence_score': result.confidence_score,
                'individual_predictions': [
                    {
                        'model': pred.model_name,
                        'value': pred.predicted_value,
                        'confidence': pred.confidence_score
                    }
                    for pred in result.individual_predictions
                ]
            }
        else:
            # Generic model interface
            if hasattr(model, 'predict'):
                return {'prediction': model.predict(inputs)}
            else:
                raise ValueError(f"Unknown model interface for {model_name}")
    
    async def _postprocess_outputs(self, prediction_result: Dict[str, Any], request: PredictionRequest) -> Dict[str, Any]:
        """Postprocess model outputs"""
        
        # Add confidence intervals if requested
        if request.enable_confidence_intervals:
            # Add confidence intervals to prediction result
            pass
        
        # Add explanations if requested
        if request.enable_explanation:
            # Add model explanations
            pass
        
        return prediction_result
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the serving pipeline"""
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': len(self.model_registry.loaded_models),
            'cache_enabled': self.cache.cache_enabled if self.cache else False,
            'batch_jobs_running': sum(1 for job in self.batch_jobs.values() if job.status == 'running')
        }
        
        # Check model registry
        try:
            models = self.model_registry.list_models()
            health_status['available_models'] = models
        except Exception as e:
            health_status['status'] = 'degraded'
            health_status['model_registry_error'] = str(e)
        
        # Check cache
        if self.cache:
            try:
                if self.cache.cache_enabled:
                    self.cache.redis_client.ping()
                    health_status['cache_status'] = 'connected'
                else:
                    health_status['cache_status'] = 'disabled'
            except Exception as e:
                health_status['status'] = 'degraded'
                health_status['cache_error'] = str(e)
        
        return health_status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get serving pipeline metrics"""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'models': {
                'loaded_count': len(self.model_registry.loaded_models),
                'available_models': self.model_registry.list_models()
            },
            'batch_jobs': {
                'total': len(self.batch_jobs),
                'by_status': {}
            }
        }
        
        # Batch job metrics
        for job in self.batch_jobs.values():
            status = job.status
            if status not in metrics['batch_jobs']['by_status']:
                metrics['batch_jobs']['by_status'][status] = 0
            metrics['batch_jobs']['by_status'][status] += 1
        
        # Cache metrics
        if self.cache and self.cache.cache_enabled:
            try:
                cache_info = self.cache.redis_client.info()
                metrics['cache'] = {
                    'connected_clients': cache_info.get('connected_clients', 0),
                    'used_memory': cache_info.get('used_memory', 0),
                    'keyspace_hits': cache_info.get('keyspace_hits', 0),
                    'keyspace_misses': cache_info.get('keyspace_misses', 0)
                }
                
                # Calculate hit rate
                hits = metrics['cache']['keyspace_hits']
                misses = metrics['cache']['keyspace_misses']
                if hits + misses > 0:
                    metrics['cache']['hit_rate'] = hits / (hits + misses)
                else:
                    metrics['cache']['hit_rate'] = 0.0
                    
            except Exception:
                metrics['cache'] = {'status': 'error'}
        
        return metrics
    
    async def shutdown(self):
        """Gracefully shutdown the serving pipeline"""
        
        logger.info("Shutting down model serving pipeline")
        
        # Cancel running batch jobs
        for job in self.batch_jobs.values():
            if job.status == 'running':
                job.status = 'cancelled'
        
        # Shutdown executors
        self.prediction_executor.shutdown(wait=True)
        self.batch_executor.shutdown(wait=True)
        
        # Close cache connection
        if self.cache and self.cache.cache_enabled:
            self.cache.redis_client.close()
        
        logger.info("Model serving pipeline shutdown complete")

# Factory function
def create_model_serving_pipeline(**kwargs) -> ModelServingPipeline:
    """Factory function for creating model serving pipeline"""
    return ModelServingPipeline(**kwargs)

# Utility functions
def estimate_prediction_cost(request: PredictionRequest, model_metadata: Dict) -> float:
    """Estimate computational cost of prediction"""
    
    # Base cost factors
    base_cost = 0.01  # $0.01 base cost
    
    # Model complexity factor
    model_complexity = model_metadata.get('complexity_factor', 1.0)
    
    # Input size factor
    input_size_factor = len(str(request.inputs)) / 1000  # Per KB of input
    
    # Feature flags cost
    feature_cost = 0
    if request.enable_explanation:
        feature_cost += 0.005
    if request.enable_confidence_intervals:
        feature_cost += 0.002
    
    total_cost = base_cost * model_complexity * (1 + input_size_factor) + feature_cost
    
    return round(total_cost, 4)

def validate_model_inputs(inputs: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate model inputs against schema"""
    
    errors = []
    
    # Check required fields
    for field, config in schema.items():
        if config.get('required', False) and field not in inputs:
            errors.append(f"Missing required field: {field}")
        
        if field in inputs:
            value = inputs[field]
            expected_type = config.get('type')
            
            # Type validation
            if expected_type == 'float' and not isinstance(value, (int, float)):
                errors.append(f"Field {field} must be numeric")
            elif expected_type == 'string' and not isinstance(value, str):
                errors.append(f"Field {field} must be string")
            elif expected_type == 'list' and not isinstance(value, list):
                errors.append(f"Field {field} must be list")
            
            # Range validation
            if 'min_value' in config and isinstance(value, (int, float)) and value < config['min_value']:
                errors.append(f"Field {field} must be >= {config['min_value']}")
            if 'max_value' in config and isinstance(value, (int, float)) and value > config['max_value']:
                errors.append(f"Field {field} must be <= {config['max_value']}")
    
    return errors