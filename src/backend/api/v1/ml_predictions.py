"""
ML Predictions API Endpoints
Real-time and batch prediction capabilities for IPO valuation models
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import asyncio
import json
import pandas as pd
import io
from datetime import datetime, timedelta
import logging
import uuid
from pathlib import Path

# Import model classes
from ...ml_services.models.model_serving import (
    ModelServingPipeline, PredictionRequest, PredictionResponse,
    BatchPredictionJob, create_model_serving_pipeline
)
from ...ml_services.models.model_monitoring import (
    ModelPerformanceMonitor, create_model_performance_monitor
)

# Import model-specific input/output classes
from ...ml_services.models.advanced_dcf import AdvancedDCFInputs
from ...ml_services.models.comparable_company_analysis import CompanyFinancialData, PeerSelectionCriteria
from ...ml_services.models.risk_assessment import RiskAssessmentInputs
from ...ml_services.models.market_sentiment import SentimentAnalysisInputs
from ...ml_services.models.ensemble_framework import EnsembleInputs

logger = logging.getLogger(__name__)

# Initialize global instances
serving_pipeline = create_model_serving_pipeline()
performance_monitor = create_model_performance_monitor()

router = APIRouter(prefix="/ml", tags=["Machine Learning Predictions"])

# Pydantic models for API
class DCFPredictionRequest(BaseModel):
    """DCF prediction request model"""
    company_name: str = Field(..., description="Company name")
    sector: str = Field(..., description="Industry sector")
    industry: str = Field(..., description="Industry classification")
    
    # Historical financial data
    historical_revenues: List[float] = Field(..., description="Historical revenue data", min_items=3)
    historical_ebitda: List[float] = Field(..., description="Historical EBITDA data", min_items=3)
    historical_fcf: List[float] = Field(..., description="Historical free cash flow data", min_items=3)
    
    # Growth projections
    revenue_growth_rates: List[float] = Field(..., description="Projected revenue growth rates")
    ebitda_margin_targets: List[float] = Field(..., description="Target EBITDA margins")
    
    # Discount rate inputs
    risk_free_rate: float = Field(0.025, description="Risk-free rate", ge=0, le=0.1)
    market_risk_premium: float = Field(0.06, description="Market risk premium", ge=0, le=0.15)
    beta: float = Field(1.0, description="Company beta", ge=0, le=3.0)
    cost_of_debt: float = Field(0.05, description="Cost of debt", ge=0, le=0.2)
    
    # Other parameters
    terminal_growth_rate: float = Field(0.025, description="Terminal growth rate", ge=0, le=0.05)
    tax_rate: float = Field(0.25, description="Tax rate", ge=0, le=0.5)
    shares_outstanding: float = Field(..., description="Shares outstanding", gt=0)
    
    # Analysis options
    include_monte_carlo: bool = Field(True, description="Include Monte Carlo simulation")
    include_scenarios: bool = Field(True, description="Include scenario analysis")
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.8, le=0.99)

class CCACompanyData(BaseModel):
    """Company data for CCA analysis"""
    company_name: str
    sector: str
    industry: str
    market_cap: float = Field(gt=0)
    revenue: float = Field(gt=0)
    ebitda: float
    net_income: float
    shares_outstanding: float = Field(gt=0)
    
    # Optional financial metrics
    ev_revenue: Optional[float] = None
    ev_ebitda: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: Optional[float] = 1.0
    revenue_growth_3y: Optional[float] = None
    ebitda_margin: Optional[float] = None

class CCAPredictionRequest(BaseModel):
    """CCA prediction request model"""
    target_company: CCACompanyData = Field(..., description="Target company data")
    universe_companies: List[CCACompanyData] = Field(..., description="Universe of comparable companies", min_items=5)
    
    # Selection criteria
    same_sector_required: bool = Field(True, description="Require same sector")
    size_multiple_range: tuple = Field((0.5, 2.0), description="Size multiple range")
    max_peers: int = Field(15, description="Maximum number of peers", ge=3, le=50)
    min_peers: int = Field(5, description="Minimum number of peers", ge=3)
    
    # Analysis options
    include_regression_analysis: bool = Field(True, description="Include regression analysis")
    include_time_series: bool = Field(False, description="Include time series analysis")

class RiskPredictionRequest(BaseModel):
    """Risk assessment prediction request"""
    company_name: str = Field(..., description="Company name")
    sector: str = Field(..., description="Industry sector")
    
    # Market risk factors
    beta: float = Field(1.0, ge=0, le=3.0)
    volatility: float = Field(0.2, ge=0, le=1.0)
    market_cap: Optional[float] = Field(None, gt=0)
    
    # Financial risk factors
    current_ratio: float = Field(1.5, gt=0)
    debt_to_equity: float = Field(0.3, ge=0)
    interest_coverage: float = Field(5.0, ge=0)
    earnings_volatility: float = Field(0.15, ge=0, le=1.0)
    
    # Operational risk factors
    revenue_concentration: float = Field(0.2, ge=0, le=1.0, description="Revenue concentration risk")
    operating_leverage: float = Field(1.5, ge=0)
    
    # ESG scores
    environmental_score: float = Field(50.0, ge=0, le=100)
    social_score: float = Field(50.0, ge=0, le=100)
    governance_score: float = Field(50.0, ge=0, le=100)
    
    # Analysis options
    include_stress_testing: bool = Field(True, description="Include stress testing")
    include_peer_comparison: bool = Field(False, description="Include peer comparison")

class SentimentPredictionRequest(BaseModel):
    """Market sentiment prediction request"""
    company_name: str = Field(..., description="Company name")
    ticker_symbol: Optional[str] = Field(None, description="Stock ticker symbol")
    sector: str = Field(..., description="Industry sector")
    
    # Analysis configuration
    analysis_period_days: int = Field(90, ge=30, le=365, description="Analysis period in days")
    primary_keywords: List[str] = Field([], description="Primary keywords for analysis")
    
    # Analysis options
    include_finbert: bool = Field(True, description="Use FinBERT for news analysis")
    include_social_sentiment: bool = Field(True, description="Include social media sentiment")
    include_analyst_sentiment: bool = Field(True, description="Include analyst sentiment")
    include_market_timing: bool = Field(True, description="Include market timing analysis")

class EnsemblePredictionRequest(BaseModel):
    """Ensemble prediction request"""
    company_name: str = Field(..., description="Company name")
    
    # Input data for different models
    dcf_inputs: Optional[DCFPredictionRequest] = None
    cca_inputs: Optional[CCAPredictionRequest] = None
    risk_inputs: Optional[RiskPredictionRequest] = None
    sentiment_inputs: Optional[SentimentPredictionRequest] = None
    
    # Ensemble configuration
    models_to_include: List[str] = Field(["dcf", "cca", "risk", "sentiment"], description="Models to include in ensemble")
    weighting_method: str = Field("dynamic", description="Ensemble weighting method")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99)

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    job_name: str = Field(..., description="Job name")
    model_name: str = Field(..., description="Model name")
    
    # Processing options
    batch_size: int = Field(1000, ge=1, le=10000, description="Batch size")
    parallel_workers: int = Field(4, ge=1, le=16, description="Number of parallel workers")
    
    # Output options
    output_format: str = Field("csv", description="Output format", regex="^(csv|json|parquet)$")
    include_confidence_intervals: bool = Field(True, description="Include confidence intervals")
    include_explanations: bool = Field(False, description="Include model explanations")

class PredictionResponse(BaseModel):
    """Standard prediction response"""
    request_id: str
    model_name: str
    prediction: Dict[str, Any]
    confidence_score: float = Field(ge=0, le=1)
    prediction_time_ms: float
    
    # Optional fields
    confidence_intervals: Optional[Dict[str, tuple]] = None
    explanation: Optional[Dict[str, Any]] = None
    model_version: str = "latest"
    warnings: List[str] = []

class BatchJobStatus(BaseModel):
    """Batch job status response"""
    job_id: str
    job_name: str
    status: str
    progress_percentage: float = Field(ge=0, le=100)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Job statistics
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    
    # Results
    output_path: Optional[str] = None
    download_url: Optional[str] = None

# Real-time prediction endpoints

@router.post("/predict/dcf", response_model=PredictionResponse)
async def predict_dcf_valuation(
    request: DCFPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    DCF valuation prediction with Monte Carlo simulation
    """
    try:
        # Convert to internal format
        dcf_inputs = AdvancedDCFInputs(
            company_name=request.company_name,
            sector=request.sector,
            industry=request.industry,
            historical_revenues=request.historical_revenues,
            historical_ebitda=request.historical_ebitda,
            historical_fcf=request.historical_fcf,
            revenue_growth_rates=request.revenue_growth_rates,
            ebitda_margin_targets=request.ebitda_margin_targets,
            risk_free_rate=request.risk_free_rate,
            market_risk_premium=request.market_risk_premium,
            beta=request.beta,
            cost_of_debt=request.cost_of_debt,
            terminal_growth_rate=request.terminal_growth_rate,
            tax_rate=request.tax_rate,
            shares_outstanding=request.shares_outstanding
        )
        
        # Create prediction request
        prediction_request = PredictionRequest(
            model_name="dcf",
            inputs=dcf_inputs,
            enable_confidence_intervals=True,
            validate_inputs=True
        )
        
        # Make prediction
        result = await serving_pipeline.predict(prediction_request)
        
        # Record for monitoring
        background_tasks.add_task(
            performance_monitor.record_prediction,
            "dcf",
            result.prediction.get("value_per_share", 0),
            result.confidence_score,
            result.prediction_time_ms,
            {"company": request.company_name}
        )
        
        return PredictionResponse(
            request_id=result.request_id,
            model_name="dcf",
            prediction=result.prediction,
            confidence_score=result.confidence_score,
            prediction_time_ms=result.prediction_time_ms,
            confidence_intervals=result.prediction_intervals,
            model_version=result.model_version,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"DCF prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/cca", response_model=PredictionResponse)
async def predict_cca_valuation(
    request: CCAPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Comparable Company Analysis valuation prediction
    """
    try:
        # Convert to internal format
        target_company = CompanyFinancialData(
            company_name=request.target_company.company_name,
            sector=request.target_company.sector,
            industry=request.target_company.industry,
            market_cap=request.target_company.market_cap,
            revenue=request.target_company.revenue,
            ebitda=request.target_company.ebitda,
            net_income=request.target_company.net_income,
            shares_outstanding=request.target_company.shares_outstanding
        )
        
        universe_companies = [
            CompanyFinancialData(
                company_name=comp.company_name,
                sector=comp.sector,
                industry=comp.industry,
                market_cap=comp.market_cap,
                revenue=comp.revenue,
                ebitda=comp.ebitda,
                net_income=comp.net_income,
                shares_outstanding=comp.shares_outstanding,
                ev_revenue=comp.ev_revenue,
                ev_ebitda=comp.ev_ebitda,
                pe_ratio=comp.pe_ratio,
                beta=comp.beta
            )
            for comp in request.universe_companies
        ]
        
        selection_criteria = PeerSelectionCriteria(
            target_company=request.target_company.company_name,
            same_sector_required=request.same_sector_required,
            size_multiple_range=request.size_multiple_range,
            max_peers=request.max_peers,
            min_peers=request.min_peers
        )
        
        # Create prediction request
        cca_inputs = {
            "target_company": target_company,
            "universe": universe_companies,
            "criteria": selection_criteria
        }
        
        prediction_request = PredictionRequest(
            model_name="cca",
            inputs=cca_inputs,
            enable_confidence_intervals=True
        )
        
        # Make prediction
        result = await serving_pipeline.predict(prediction_request)
        
        # Record for monitoring
        background_tasks.add_task(
            performance_monitor.record_prediction,
            "cca",
            list(result.prediction.get("implied_valuations", {}).values())[0] if result.prediction.get("implied_valuations") else 0,
            result.confidence_score,
            result.prediction_time_ms,
            {"company": request.target_company.company_name}
        )
        
        return PredictionResponse(
            request_id=result.request_id,
            model_name="cca",
            prediction=result.prediction,
            confidence_score=result.confidence_score,
            prediction_time_ms=result.prediction_time_ms,
            model_version=result.model_version,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"CCA prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/risk", response_model=PredictionResponse)
async def predict_risk_assessment(
    request: RiskPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Risk assessment prediction with multi-factor analysis
    """
    try:
        # Convert to internal format
        from ...ml_services.models.risk_assessment import (
            MarketRiskFactors, FinancialRiskFactors, OperationalRiskFactors,
            ESGMetrics, RegulatoryRiskFactors
        )
        
        risk_inputs = RiskAssessmentInputs(
            company_name=request.company_name,
            sector=request.sector,
            market_risks=MarketRiskFactors(
                beta=request.beta,
                volatility=request.volatility,
                market_cap_risk=0.1
            ),
            financial_risks=FinancialRiskFactors(
                current_ratio=request.current_ratio,
                debt_to_equity=request.debt_to_equity,
                interest_coverage=request.interest_coverage,
                earnings_volatility=request.earnings_volatility
            ),
            operational_risks=OperationalRiskFactors(
                revenue_concentration=request.revenue_concentration,
                operating_leverage=request.operating_leverage
            ),
            esg_metrics=ESGMetrics(
                environmental_score=request.environmental_score,
                social_score=request.social_score,
                governance_score=request.governance_score
            ),
            regulatory_risks=RegulatoryRiskFactors()
        )
        
        # Create prediction request
        prediction_request = PredictionRequest(
            model_name="risk",
            inputs=risk_inputs,
            enable_confidence_intervals=True
        )
        
        # Make prediction
        result = await serving_pipeline.predict(prediction_request)
        
        # Record for monitoring
        background_tasks.add_task(
            performance_monitor.record_prediction,
            "risk",
            result.prediction.get("composite_risk_score", 50),
            result.confidence_score,
            result.prediction_time_ms,
            {"company": request.company_name}
        )
        
        return PredictionResponse(
            request_id=result.request_id,
            model_name="risk",
            prediction=result.prediction,
            confidence_score=result.confidence_score,
            prediction_time_ms=result.prediction_time_ms,
            model_version=result.model_version,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"Risk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/sentiment", response_model=PredictionResponse)
async def predict_market_sentiment(
    request: SentimentPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Market sentiment analysis prediction
    """
    try:
        # Convert to internal format
        sentiment_inputs = SentimentAnalysisInputs(
            company_name=request.company_name,
            ticker_symbol=request.ticker_symbol,
            sector=request.sector,
            analysis_period_days=request.analysis_period_days,
            primary_keywords=request.primary_keywords,
            include_finbert=request.include_finbert,
            include_social_sentiment=request.include_social_sentiment,
            include_analyst_sentiment=request.include_analyst_sentiment,
            include_market_timing=request.include_market_timing
        )
        
        # Create prediction request
        prediction_request = PredictionRequest(
            model_name="sentiment",
            inputs=sentiment_inputs,
            enable_confidence_intervals=True
        )
        
        # Make prediction
        result = await serving_pipeline.predict(prediction_request)
        
        # Record for monitoring
        background_tasks.add_task(
            performance_monitor.record_prediction,
            "sentiment",
            result.prediction.get("composite_sentiment_score", 0),
            result.confidence_score,
            result.prediction_time_ms,
            {"company": request.company_name}
        )
        
        return PredictionResponse(
            request_id=result.request_id,
            model_name="sentiment",
            prediction=result.prediction,
            confidence_score=result.confidence_score,
            prediction_time_ms=result.prediction_time_ms,
            model_version=result.model_version,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"Sentiment prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/ensemble", response_model=PredictionResponse)
async def predict_ensemble_valuation(
    request: EnsemblePredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Ensemble valuation prediction combining multiple models
    """
    try:
        # Convert to internal format
        ensemble_inputs = EnsembleInputs(
            company_name=request.company_name,
            models_to_include=request.models_to_include,
            weighting_method=request.weighting_method,
            confidence_level=request.confidence_level
        )
        
        # Add model-specific inputs
        if request.dcf_inputs:
            ensemble_inputs.dcf_inputs = AdvancedDCFInputs(
                company_name=request.dcf_inputs.company_name,
                historical_revenues=request.dcf_inputs.historical_revenues,
                historical_ebitda=request.dcf_inputs.historical_ebitda,
                revenue_growth_rates=request.dcf_inputs.revenue_growth_rates,
                ebitda_margin_targets=request.dcf_inputs.ebitda_margin_targets,
                shares_outstanding=request.dcf_inputs.shares_outstanding
            )
        
        # Create prediction request
        prediction_request = PredictionRequest(
            model_name="ensemble",
            inputs=ensemble_inputs,
            enable_confidence_intervals=True,
            enable_explanation=True
        )
        
        # Make prediction
        result = await serving_pipeline.predict(prediction_request)
        
        # Record for monitoring
        background_tasks.add_task(
            performance_monitor.record_prediction,
            "ensemble",
            result.prediction.get("ensemble_valuation", 0),
            result.confidence_score,
            result.prediction_time_ms,
            {"company": request.company_name}
        )
        
        return PredictionResponse(
            request_id=result.request_id,
            model_name="ensemble",
            prediction=result.prediction,
            confidence_score=result.confidence_score,
            prediction_time_ms=result.prediction_time_ms,
            explanation=result.explanation,
            model_version=result.model_version,
            warnings=result.warnings
        )
        
    except Exception as e:
        logger.error(f"Ensemble prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoints

@router.post("/batch/predict", response_model=Dict[str, str])
async def start_batch_prediction(
    request: BatchPredictionRequest,
    file: UploadFile = File(..., description="CSV file with input data"),
    background_tasks: BackgroundTasks
):
    """
    Start batch prediction job
    """
    try:
        # Validate file format
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read and validate input data
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="Input file is empty")
        
        # Generate job ID
        job_id = f"{request.job_name}_{uuid.uuid4().hex[:8]}"
        
        # Create output path
        output_dir = Path("batch_outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{job_id}_results.{request.output_format}"
        
        # Create batch job
        batch_job = BatchPredictionJob(
            job_id=job_id,
            model_name=request.model_name,
            input_data=df,
            output_path=str(output_path),
            batch_size=request.batch_size,
            parallel_workers=request.parallel_workers
        )
        
        # Start job
        job_id = await serving_pipeline.start_batch_job(batch_job)
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Batch prediction start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch/status/{job_id}", response_model=BatchJobStatus)
async def get_batch_job_status(job_id: str):
    """
    Get batch job status
    """
    try:
        job = await serving_pipeline.get_batch_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return BatchJobStatus(
            job_id=job.job_id,
            job_name=job_id.split('_')[0],  # Extract job name from ID
            status=job.status,
            progress_percentage=job.progress_percentage,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            total_records=job.total_records,
            processed_records=job.processed_records,
            failed_records=job.failed_records,
            output_path=job.output_path if job.status == "completed" else None,
            download_url=f"/ml/batch/download/{job_id}" if job.status == "completed" else None
        )
        
    except Exception as e:
        logger.error(f"Get batch status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch/download/{job_id}")
async def download_batch_results(job_id: str):
    """
    Download batch prediction results
    """
    try:
        job = await serving_pipeline.get_batch_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")
        
        output_path = Path(job.output_path)
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Results file not found")
        
        return FileResponse(
            path=str(output_path),
            filename=f"{job_id}_results.csv",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Download batch results failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/batch/{job_id}")
async def cancel_batch_job(job_id: str):
    """
    Cancel running batch job
    """
    try:
        success = await serving_pipeline.cancel_batch_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
        return {"message": "Job cancelled successfully"}
        
    except Exception as e:
        logger.error(f"Cancel batch job failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/batch/jobs", response_model=List[BatchJobStatus])
async def list_batch_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of jobs to return")
):
    """
    List batch jobs
    """
    try:
        # This would typically query a database
        # For now, return empty list as placeholder
        return []
        
    except Exception as e:
        logger.error(f"List batch jobs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints

@router.get("/models", response_model=Dict[str, List[str]])
async def list_available_models():
    """
    List available models and versions
    """
    try:
        models = serving_pipeline.model_registry.list_models()
        return models
        
    except Exception as e:
        logger.error(f"List models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}/metadata")
async def get_model_metadata(model_name: str, version: str = "latest"):
    """
    Get model metadata
    """
    try:
        metadata = serving_pipeline.model_registry.get_model_metadata(model_name, version)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Get model metadata failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health and monitoring endpoints

@router.get("/health")
async def health_check():
    """
    Health check for ML services
    """
    try:
        health = await serving_pipeline.health_check()
        return health
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics():
    """
    Get ML service metrics
    """
    try:
        metrics = await serving_pipeline.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/{model_name}")
async def get_model_monitoring_data(model_name: str):
    """
    Get monitoring data for specific model
    """
    try:
        monitoring_data = await performance_monitor.get_monitoring_dashboard_data(model_name)
        return monitoring_data
        
    except Exception as e:
        logger.error(f"Get monitoring data failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints

@router.post("/validate")
async def validate_prediction_inputs(
    model_name: str,
    inputs: Dict[str, Any]
):
    """
    Validate prediction inputs for a specific model
    """
    try:
        # This would validate inputs against model schema
        # For now, return basic validation
        
        if not inputs:
            raise HTTPException(status_code=400, detail="Inputs cannot be empty")
        
        return {"valid": True, "message": "Inputs are valid"}
        
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/schema/{model_name}")
async def get_model_input_schema(model_name: str):
    """
    Get input schema for a specific model
    """
    try:
        # Return schema based on model type
        if model_name == "dcf":
            return DCFPredictionRequest.schema()
        elif model_name == "cca":
            return CCAPredictionRequest.schema()
        elif model_name == "risk":
            return RiskPredictionRequest.schema()
        elif model_name == "sentiment":
            return SentimentPredictionRequest.schema()
        elif model_name == "ensemble":
            return EnsemblePredictionRequest.schema()
        else:
            raise HTTPException(status_code=404, detail="Model not found")
        
    except Exception as e:
        logger.error(f"Get model schema failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming predictions (for real-time updates)
@router.get("/stream/predictions/{model_name}")
async def stream_predictions(model_name: str):
    """
    Stream real-time prediction updates
    """
    async def generate_updates():
        while True:
            # This would stream real-time updates
            # For now, yield placeholder data
            yield f"data: {json.dumps({'timestamp': datetime.now().isoformat(), 'model': model_name})}\n\n"
            await asyncio.sleep(5)
    
    return StreamingResponse(generate_updates(), media_type="text/plain")

# Initialize monitoring on startup
@router.on_event("startup")
async def startup_event():
    """Initialize ML services on startup"""
    logger.info("Starting ML prediction services...")
    
    # Start performance monitoring
    await performance_monitor.start_monitoring()
    
    logger.info("ML prediction services started successfully")

@router.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ML prediction services...")
    
    # Stop monitoring
    await performance_monitor.stop_monitoring()
    
    # Shutdown serving pipeline
    await serving_pipeline.shutdown()
    
    logger.info("ML prediction services shut down successfully")