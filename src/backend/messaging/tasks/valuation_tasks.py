"""
Valuation Celery Tasks
Async tasks for valuation calculations
"""
import asyncio
from typing import Dict, Any, Optional
from celery import Task
import structlog

from ..celery_app import celery_app
from ...ml_services.valuation_engine.dcf_model import DCFValuationModel, DCFInputs
from ...ml_services.valuation_engine.cca_model import CCAValuationModel, CCAInputs
from ...database.base import get_db
from ...database.models import Valuation, Company
from ...utils.metrics import track_valuation_created
from ...utils.logging import audit_logger

logger = structlog.get_logger(__name__)


class BaseValuationTask(Task):
    """Base class for valuation tasks"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(
            "Valuation task failed",
            task_id=task_id,
            exception=str(exc),
            args=args,
            kwargs=kwargs
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info(
            "Valuation task completed",
            task_id=task_id,
            result_type=type(retval).__name__
        )


@celery_app.task(
    bind=True,
    base=BaseValuationTask,
    name="valuation.dcf_calculation",
    queue="valuation",
    max_retries=3,
    default_retry_delay=60
)
def calculate_dcf_valuation(self, inputs_dict: Dict[str, Any], valuation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate DCF valuation
    
    Args:
        inputs_dict: DCF input parameters as dictionary
        valuation_id: Optional valuation record ID to update
    
    Returns:
        DCF calculation results
    """
    try:
        logger.info("Starting DCF calculation", valuation_id=valuation_id)
        
        # Convert dict to DCFInputs
        inputs = DCFInputs(**inputs_dict)
        
        # Create DCF model and calculate
        dcf_model = DCFValuationModel()
        result = dcf_model.calculate_valuation(inputs)
        
        # Update database if valuation_id provided
        if valuation_id:
            with next(get_db()) as db:
                valuation = db.query(Valuation).filter(Valuation.id == valuation_id).first()
                if valuation:
                    # Update valuation record
                    valuation.base_case_value = result.equity_value
                    valuation.target_price = result.value_per_share
                    valuation.confidence_level = result.model_r_squared * 100
                    valuation.assumptions = {
                        'wacc': result.wacc,
                        'terminal_growth_rate': inputs.terminal_growth_rate,
                        'projection_years': len(result.revenue_projections)
                    }
                    
                    db.commit()
                    
                    # Log audit trail
                    audit_logger.log_data_access(
                        user_id=str(valuation.analyst_id),
                        resource_type="valuation",
                        resource_id=str(valuation.id),
                        action="dcf_calculation"
                    )
        
        # Track metrics
        track_valuation_created("dcf", True)
        
        # Convert result to serializable format
        result_dict = {
            'enterprise_value': float(result.enterprise_value),
            'equity_value': float(result.equity_value),
            'value_per_share': float(result.value_per_share),
            'terminal_value': float(result.terminal_value),
            'wacc': result.wacc,
            'confidence_interval_95': result.confidence_interval_95,
            'sensitivity_matrix': result.sensitivity_matrix,
            'revenue_projections': result.revenue_projections,
            'fcf_projections': result.fcf_projections,
            'model_r_squared': result.model_r_squared
        }
        
        logger.info(
            "DCF calculation completed",
            valuation_id=valuation_id,
            value_per_share=result_dict['value_per_share']
        )
        
        return result_dict
        
    except Exception as exc:
        track_valuation_created("dcf", False)
        logger.error(f"DCF calculation failed: {str(exc)}", valuation_id=valuation_id)
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        raise exc


@celery_app.task(
    bind=True,
    base=BaseValuationTask,
    name="valuation.cca_analysis",
    queue="valuation",
    max_retries=3,
    default_retry_delay=60
)
def perform_cca_analysis(self, inputs_dict: Dict[str, Any], valuation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform Comparable Company Analysis
    
    Args:
        inputs_dict: CCA input parameters as dictionary
        valuation_id: Optional valuation record ID to update
    
    Returns:
        CCA analysis results
    """
    try:
        logger.info("Starting CCA analysis", valuation_id=valuation_id)
        
        # Convert dict to CCAInputs
        inputs = CCAInputs(**inputs_dict)
        
        # Create CCA model and analyze
        cca_model = CCAValuationModel()
        result = cca_model.calculate_valuation(inputs)
        
        # Update database if valuation_id provided
        if valuation_id:
            with next(get_db()) as db:
                valuation = db.query(Valuation).filter(Valuation.id == valuation_id).first()
                if valuation:
                    # Update valuation record
                    valuation.base_case_value = result.weighted_valuation
                    valuation.bear_case_value = result.valuation_range[0]
                    valuation.bull_case_value = result.valuation_range[1]
                    valuation.confidence_level = result.selection_confidence * 100
                    valuation.comparable_metrics = {
                        'peer_count': len(result.selected_peers),
                        'data_completeness': result.data_completeness,
                        'selection_confidence': result.selection_confidence
                    }
                    
                    db.commit()
        
        # Track metrics
        track_valuation_created("cca", True)
        
        # Convert result to serializable format
        result_dict = {
            'weighted_valuation': float(result.weighted_valuation),
            'valuation_range': [float(result.valuation_range[0]), float(result.valuation_range[1])],
            'selected_peers_count': len(result.selected_peers),
            'selection_confidence': result.selection_confidence,
            'data_completeness': result.data_completeness,
            'multiple_statistics': result.multiple_statistics,
            'peer_statistics': result.peer_statistics,
            'valuation_volatility': result.valuation_volatility
        }
        
        logger.info(
            "CCA analysis completed",
            valuation_id=valuation_id,
            weighted_valuation=result_dict['weighted_valuation']
        )
        
        return result_dict
        
    except Exception as exc:
        track_valuation_created("cca", False)
        logger.error(f"CCA analysis failed: {str(exc)}", valuation_id=valuation_id)
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        raise exc


@celery_app.task(
    bind=True,
    base=BaseValuationTask,
    name="valuation.risk_assessment",
    queue="valuation",
    max_retries=2
)
def assess_investment_risk(self, company_id: str, valuation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive risk assessment
    
    Args:
        company_id: Company identifier
        valuation_data: Valuation data for risk analysis
    
    Returns:
        Risk assessment results
    """
    try:
        logger.info("Starting risk assessment", company_id=company_id)
        
        # Import here to avoid circular imports
        from ...ml_services.risk_models.risk_analyzer import RiskAnalyzer
        
        # Create risk analyzer
        risk_analyzer = RiskAnalyzer()
        
        # Get company data
        with next(get_db()) as db:
            company = db.query(Company).filter(Company.id == company_id).first()
            if not company:
                raise ValueError(f"Company not found: {company_id}")
        
        # Perform risk analysis
        risk_result = risk_analyzer.analyze_risk({
            'company': company.to_dict(),
            'valuation_data': valuation_data
        })
        
        # Convert to serializable format
        result_dict = {
            'overall_risk_score': risk_result.overall_risk_score,
            'risk_level': risk_result.risk_level,
            'risk_factors': risk_result.risk_factors,
            'risk_breakdown': risk_result.risk_breakdown,
            'mitigation_suggestions': risk_result.mitigation_suggestions
        }
        
        logger.info(
            "Risk assessment completed",
            company_id=company_id,
            risk_score=result_dict['overall_risk_score']
        )
        
        return result_dict
        
    except Exception as exc:
        logger.error(f"Risk assessment failed: {str(exc)}", company_id=company_id)
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30)
        
        raise exc


@celery_app.task(
    bind=True,
    name="valuation.comprehensive_analysis",
    queue="valuation"
)
def perform_comprehensive_valuation(
    self,
    company_id: str,
    dcf_inputs: Dict[str, Any],
    cca_inputs: Dict[str, Any],
    user_id: str
) -> Dict[str, Any]:
    """
    Perform comprehensive valuation using multiple methods
    
    Args:
        company_id: Company identifier
        dcf_inputs: DCF input parameters
        cca_inputs: CCA input parameters
        user_id: User performing the analysis
    
    Returns:
        Comprehensive valuation results
    """
    try:
        logger.info("Starting comprehensive valuation", company_id=company_id, user_id=user_id)
        
        # Create valuation record
        with next(get_db()) as db:
            company = db.query(Company).filter(Company.id == company_id).first()
            if not company:
                raise ValueError(f"Company not found: {company_id}")
            
            valuation = Valuation(
                company_id=company_id,
                analyst_id=user_id,
                name=f"Comprehensive Analysis - {company.name}",
                primary_method="comprehensive",
                methods_used=["dcf", "cca"],
                status="in_progress"
            )
            
            db.add(valuation)
            db.commit()
            db.refresh(valuation)
            valuation_id = str(valuation.id)
        
        # Execute DCF calculation
        dcf_task = calculate_dcf_valuation.delay(dcf_inputs, valuation_id)
        dcf_result = dcf_task.get(timeout=300)  # 5 minutes timeout
        
        # Execute CCA analysis
        cca_task = perform_cca_analysis.delay(cca_inputs, valuation_id)
        cca_result = cca_task.get(timeout=300)
        
        # Execute risk assessment
        valuation_data = {
            'dcf': dcf_result,
            'cca': cca_result
        }
        risk_task = assess_investment_risk.delay(company_id, valuation_data)
        risk_result = risk_task.get(timeout=180)  # 3 minutes timeout
        
        # Calculate weighted average valuation
        dcf_weight = 0.6
        cca_weight = 0.4
        
        weighted_valuation = (
            dcf_result['value_per_share'] * dcf_weight +
            cca_result['weighted_valuation'] * cca_weight
        )
        
        # Update final valuation
        with next(get_db()) as db:
            valuation = db.query(Valuation).filter(Valuation.id == valuation_id).first()
            if valuation:
                valuation.base_case_value = weighted_valuation
                valuation.bear_case_value = min(dcf_result['equity_value'], cca_result['valuation_range'][0])
                valuation.bull_case_value = max(dcf_result['equity_value'], cca_result['valuation_range'][1])
                valuation.overall_risk_level = risk_result['risk_level']
                valuation.risk_score = risk_result['overall_risk_score']
                valuation.status = "completed"
                
                db.commit()
        
        comprehensive_result = {
            'valuation_id': valuation_id,
            'weighted_valuation': weighted_valuation,
            'dcf_result': dcf_result,
            'cca_result': cca_result,
            'risk_result': risk_result,
            'methodology': {
                'dcf_weight': dcf_weight,
                'cca_weight': cca_weight
            }
        }
        
        logger.info(
            "Comprehensive valuation completed",
            company_id=company_id,
            valuation_id=valuation_id,
            weighted_valuation=weighted_valuation
        )
        
        return comprehensive_result
        
    except Exception as exc:
        logger.error(f"Comprehensive valuation failed: {str(exc)}", company_id=company_id)
        
        # Update valuation status to failed
        if 'valuation_id' in locals():
            with next(get_db()) as db:
                valuation = db.query(Valuation).filter(Valuation.id == valuation_id).first()
                if valuation:
                    valuation.status = "failed"
                    db.commit()
        
        raise exc