"""
Cloud Function for ML Model Inference
Serverless inference endpoint for IPO valuation models
"""

import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import monitoring_v3
from google.cloud import secretmanager
import functions_framework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model clients
vertex_client = None
bq_client = None
monitoring_client = None

def initialize_clients():
    """Initialize Google Cloud clients"""
    global vertex_client, bq_client, monitoring_client
    
    if vertex_client is None:
        aiplatform.init(project="PROJECT_ID", location="us-central1")
        vertex_client = aiplatform
        
    if bq_client is None:
        bq_client = bigquery.Client()
        
    if monitoring_client is None:
        monitoring_client = monitoring_v3.MetricServiceClient()

@functions_framework.http
def ml_inference(request):
    """HTTP Cloud Function for ML model inference"""
    
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    }
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return ('', 204, headers)
    
    try:
        # Initialize clients
        initialize_clients()
        
        # Parse request
        request_json = request.get_json(silent=True)
        if not request_json:
            return json.dumps({'error': 'Invalid JSON in request'}), 400, headers
        
        # Validate request format
        validation_result = validate_request(request_json)
        if not validation_result['valid']:
            return json.dumps({'error': validation_result['error']}), 400, headers
        
        # Extract parameters
        model_type = request_json.get('model_type')
        features = request_json.get('features')
        company_id = request_json.get('company_id')
        request_id = request_json.get('request_id', generate_request_id())
        
        # Log request (without sensitive data)
        log_request(request_id, model_type, company_id)
        
        # Route to appropriate model
        if model_type == 'dcf_prediction':
            result = predict_dcf_valuation(features, request_id)
        elif model_type == 'risk_assessment':
            result = assess_risk(features, request_id)
        elif model_type == 'market_multiple':
            result = predict_market_multiple(features, request_id)
        elif model_type == 'sentiment_analysis':
            result = analyze_sentiment(features, request_id)
        elif model_type == 'batch_valuation':
            result = batch_valuation(features, request_id)
        else:
            return json.dumps({'error': f'Unknown model type: {model_type}'}), 400, headers
        
        # Add metadata to response
        result['request_id'] = request_id
        result['timestamp'] = datetime.utcnow().isoformat()
        result['model_version'] = get_model_version(model_type)
        
        # Log successful prediction
        log_prediction(request_id, model_type, result)
        
        # Send custom metrics
        send_prediction_metric(model_type, success=True)
        
        return json.dumps(result), 200, headers
        
    except Exception as e:
        error_message = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"Error in ML inference: {error_message}")
        logger.error(f"Traceback: {error_trace}")
        
        # Log error
        log_error(request_id if 'request_id' in locals() else 'unknown', 
                 model_type if 'model_type' in locals() else 'unknown', 
                 error_message)
        
        # Send error metric
        send_prediction_metric(model_type if 'model_type' in locals() else 'unknown', 
                             success=False)
        
        return json.dumps({'error': 'Internal server error', 'request_id': request_id if 'request_id' in locals() else None}), 500, headers

def validate_request(request_json: Dict[str, Any]) -> Dict[str, Any]:
    """Validate incoming request"""
    try:
        # Check required fields
        required_fields = ['model_type', 'features']
        for field in required_fields:
            if field not in request_json:
                return {'valid': False, 'error': f'Missing required field: {field}'}
        
        model_type = request_json['model_type']
        features = request_json['features']
        
        # Validate model type
        valid_models = ['dcf_prediction', 'risk_assessment', 'market_multiple', 
                       'sentiment_analysis', 'batch_valuation']
        if model_type not in valid_models:
            return {'valid': False, 'error': f'Invalid model type: {model_type}'}
        
        # Validate features based on model type
        validation_result = validate_features_for_model(model_type, features)
        if not validation_result['valid']:
            return validation_result
        
        return {'valid': True}
        
    except Exception as e:
        return {'valid': False, 'error': f'Validation error: {str(e)}'}

def validate_features_for_model(model_type: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Validate features for specific model type"""
    
    if model_type == 'dcf_prediction':
        required_features = [
            'revenue_growth_rate', 'ebitda_margin', 'capex_percentage',
            'working_capital_change', 'tax_rate', 'wacc', 'terminal_growth_rate'
        ]
        
        for feature in required_features:
            if feature not in features:
                return {'valid': False, 'error': f'Missing DCF feature: {feature}'}
            
            # Validate numeric ranges
            value = features[feature]
            if not isinstance(value, (int, float)):
                return {'valid': False, 'error': f'Feature {feature} must be numeric'}
            
            # Business logic validation
            if feature == 'revenue_growth_rate' and (value < -1.0 or value > 5.0):
                return {'valid': False, 'error': 'Revenue growth rate out of reasonable range'}
            
            if feature == 'ebitda_margin' and (value < -1.0 or value > 1.0):
                return {'valid': False, 'error': 'EBITDA margin out of reasonable range'}
                
    elif model_type == 'risk_assessment':
        required_features = [
            'debt_to_equity', 'current_ratio', 'interest_coverage',
            'market_beta', 'sector_volatility'
        ]
        
        for feature in required_features:
            if feature not in features:
                return {'valid': False, 'error': f'Missing risk feature: {feature}'}
                
    elif model_type == 'sentiment_analysis':
        if 'text' not in features:
            return {'valid': False, 'error': 'Missing text field for sentiment analysis'}
        
        if not isinstance(features['text'], str):
            return {'valid': False, 'error': 'Text field must be a string'}
            
        if len(features['text']) > 10000:  # 10KB limit
            return {'valid': False, 'error': 'Text too long (max 10KB)'}
    
    return {'valid': True}

def predict_dcf_valuation(features: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Predict DCF valuation using Vertex AI model"""
    try:
        # Get DCF prediction endpoint
        endpoints = vertex_client.Endpoint.list(filter='display_name="dcf-prediction-endpoint"')
        if not endpoints:
            raise ValueError("DCF prediction endpoint not found")
        
        endpoint = endpoints[0]
        
        # Prepare instance for prediction
        instance = {
            'revenue_growth_rate': features['revenue_growth_rate'],
            'ebitda_margin': features['ebitda_margin'],
            'capex_percentage': features['capex_percentage'],
            'working_capital_change': features['working_capital_change'],
            'tax_rate': features['tax_rate'],
            'wacc': features['wacc'],
            'terminal_growth_rate': features['terminal_growth_rate'],
            'company_age': features.get('company_age', 10),
            'market_sector': features.get('market_sector', 'Technology')
        }
        
        # Make prediction
        prediction = endpoint.predict(instances=[instance])
        
        # Extract results
        dcf_value = float(prediction.predictions[0]['dcf_value'])
        confidence_interval = prediction.predictions[0].get('confidence_interval', {})
        
        # Get feature attributions if available
        try:
            explanations = endpoint.explain(instances=[instance])
            feature_attributions = explanations.explanations[0].attributions
        except:
            feature_attributions = []
        
        # Validate result reasonableness
        validation_result = validate_dcf_prediction(dcf_value, features)
        
        result = {
            'model_type': 'dcf_prediction',
            'dcf_value': dcf_value,
            'confidence_interval_lower': confidence_interval.get('lower', dcf_value * 0.8),
            'confidence_interval_upper': confidence_interval.get('upper', dcf_value * 1.2),
            'feature_attributions': [
                {'feature': attr.feature_name, 'attribution': attr.attribution}
                for attr in feature_attributions[:10]  # Top 10
            ],
            'validation_warnings': validation_result.get('warnings', []),
            'confidence_score': calculate_prediction_confidence(prediction, features)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in DCF prediction: {str(e)}")
        raise

def assess_risk(features: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Assess risk using ML models"""
    try:
        # Get risk assessment endpoint
        endpoints = vertex_client.Endpoint.list(filter='display_name="risk-assessment-endpoint"')
        if not endpoints:
            raise ValueError("Risk assessment endpoint not found")
        
        endpoint = endpoints[0]
        
        # Prepare instance
        instance = {
            'debt_to_equity': features['debt_to_equity'],
            'current_ratio': features['current_ratio'],
            'interest_coverage': features['interest_coverage'],
            'market_beta': features['market_beta'],
            'sector_volatility': features['sector_volatility'],
            'revenue_growth': features.get('revenue_growth', 0.1),
            'profit_margin': features.get('profit_margin', 0.1)
        }
        
        # Make prediction
        prediction = endpoint.predict(instances=[instance])
        
        # Extract risk classification
        risk_class = prediction.predictions[0]['risk_class']
        risk_probability = float(prediction.predictions[0]['risk_probability'])
        risk_factors = prediction.predictions[0].get('risk_factors', [])
        
        # Map risk class to level
        risk_level_mapping = {
            'LOW': 1,
            'MEDIUM': 2,
            'HIGH': 3,
            'CRITICAL': 4
        }
        
        result = {
            'model_type': 'risk_assessment',
            'risk_level': risk_class,
            'risk_score': risk_probability,
            'risk_level_numeric': risk_level_mapping.get(risk_class, 2),
            'primary_risk_factors': risk_factors[:5],  # Top 5 risk factors
            'recommendations': generate_risk_recommendations(risk_class, features),
            'confidence_score': risk_probability
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}")
        raise

def predict_market_multiple(features: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Predict market multiple using ML model"""
    try:
        # Use BigQuery ML model for market multiple prediction
        query = f"""
        SELECT 
            predicted_pe_multiple,
            predicted_pb_multiple,
            predicted_ev_ebitda_multiple
        FROM ML.PREDICT(
            MODEL `ipo_valuation.ml_models.market_multiple_model`,
            (
                SELECT 
                    {features.get('revenue_growth_rate', 0.1)} as revenue_growth_rate,
                    {features.get('profit_margin', 0.1)} as profit_margin,
                    {features.get('roe', 0.15)} as roe,
                    {features.get('debt_to_equity', 1.0)} as debt_to_equity,
                    '{features.get('market_sector', 'Technology')}' as market_sector
            )
        )
        """
        
        query_job = bq_client.query(query)
        results = list(query_job.result())
        
        if not results:
            raise ValueError("No prediction results from BigQuery ML")
        
        row = results[0]
        
        result = {
            'model_type': 'market_multiple',
            'pe_multiple': float(row.predicted_pe_multiple),
            'pb_multiple': float(row.predicted_pb_multiple),
            'ev_ebitda_multiple': float(row.predicted_ev_ebitda_multiple),
            'industry_averages': get_industry_averages(features.get('market_sector', 'Technology')),
            'relative_valuation': calculate_relative_valuation(row, features)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in market multiple prediction: {str(e)}")
        raise

def analyze_sentiment(features: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Analyze sentiment of text using NLP model"""
    try:
        from google.cloud import language_v1 as language
        
        # Initialize Language client
        language_client = language.LanguageServiceClient()
        
        text = features['text']
        document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
        
        # Analyze sentiment
        sentiment_response = language_client.analyze_sentiment(
            request={"document": document, "encoding_type": language.EncodingType.UTF8}
        )
        
        # Analyze entities
        entities_response = language_client.analyze_entities(
            request={"document": document, "encoding_type": language.EncodingType.UTF8}
        )
        
        # Extract key phrases (simplified)
        key_phrases = extract_financial_key_phrases(text)
        
        result = {
            'model_type': 'sentiment_analysis',
            'sentiment_score': sentiment_response.document_sentiment.score,
            'sentiment_magnitude': sentiment_response.document_sentiment.magnitude,
            'sentiment_label': classify_sentiment(sentiment_response.document_sentiment.score),
            'entities': [
                {
                    'name': entity.name,
                    'type': entity.type_.name,
                    'salience': entity.salience
                }
                for entity in entities_response.entities[:10]
            ],
            'key_financial_phrases': key_phrases,
            'confidence_score': abs(sentiment_response.document_sentiment.score)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise

def batch_valuation(features: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """Perform comprehensive batch valuation"""
    try:
        # Run multiple valuation methods
        dcf_result = predict_dcf_valuation(features, request_id)
        multiple_result = predict_market_multiple(features, request_id)
        risk_result = assess_risk(features, request_id)
        
        # Calculate weighted average valuation
        dcf_value = dcf_result['dcf_value']
        pe_multiple = multiple_result['pe_multiple']
        estimated_earnings = features.get('estimated_earnings', 0)
        multiple_value = pe_multiple * estimated_earnings if estimated_earnings > 0 else 0
        
        # Weight valuations based on confidence and risk
        dcf_weight = 0.6 * (1 - risk_result['risk_score'] * 0.3)
        multiple_weight = 0.4 * (1 - risk_result['risk_score'] * 0.2)
        
        if multiple_value > 0:
            weighted_valuation = (dcf_value * dcf_weight + multiple_value * multiple_weight) / (dcf_weight + multiple_weight)
        else:
            weighted_valuation = dcf_value
        
        result = {
            'model_type': 'batch_valuation',
            'dcf_valuation': dcf_result,
            'market_multiple_valuation': multiple_result,
            'risk_assessment': risk_result,
            'weighted_valuation': weighted_valuation,
            'valuation_range': {
                'low': min(dcf_value, multiple_value if multiple_value > 0 else dcf_value) * 0.8,
                'high': max(dcf_value, multiple_value if multiple_value > 0 else dcf_value) * 1.2
            },
            'recommendation': generate_investment_recommendation(weighted_valuation, risk_result)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in batch valuation: {str(e)}")
        raise

def validate_dcf_prediction(dcf_value: float, features: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DCF prediction for reasonableness"""
    warnings = []
    
    # Check if valuation seems reasonable given inputs
    revenue = features.get('estimated_revenue', 0)
    if revenue > 0:
        revenue_multiple = dcf_value / revenue
        if revenue_multiple > 50:
            warnings.append("DCF valuation appears high relative to revenue")
        elif revenue_multiple < 0.5:
            warnings.append("DCF valuation appears low relative to revenue")
    
    # Check for negative valuation
    if dcf_value < 0:
        warnings.append("Negative DCF valuation - review assumptions")
    
    return {'warnings': warnings}

def calculate_prediction_confidence(prediction, features: Dict[str, Any]) -> float:
    """Calculate confidence score for prediction"""
    # Simple confidence calculation based on feature completeness and model certainty
    feature_completeness = len([v for v in features.values() if v is not None]) / len(features)
    
    # Model-specific confidence (would come from model output in practice)
    model_confidence = 0.85  # Default confidence
    
    return min(feature_completeness * model_confidence, 1.0)

def get_industry_averages(sector: str) -> Dict[str, float]:
    """Get industry average multiples"""
    industry_averages = {
        'Technology': {'pe': 25.0, 'pb': 4.0, 'ev_ebitda': 18.0},
        'Healthcare': {'pe': 22.0, 'pb': 3.5, 'ev_ebitda': 16.0},
        'Financial': {'pe': 12.0, 'pb': 1.2, 'ev_ebitda': 10.0},
        'Energy': {'pe': 15.0, 'pb': 1.8, 'ev_ebitda': 8.0},
        'Consumer': {'pe': 18.0, 'pb': 2.5, 'ev_ebitda': 12.0}
    }
    
    return industry_averages.get(sector, industry_averages['Technology'])

def calculate_relative_valuation(prediction_row, features: Dict[str, Any]) -> str:
    """Calculate relative valuation assessment"""
    sector = features.get('market_sector', 'Technology')
    industry_avg = get_industry_averages(sector)
    
    pe_ratio = prediction_row.predicted_pe_multiple / industry_avg['pe']
    
    if pe_ratio > 1.3:
        return "PREMIUM"
    elif pe_ratio < 0.7:
        return "DISCOUNT"
    else:
        return "INLINE"

def extract_financial_key_phrases(text: str) -> List[str]:
    """Extract financial key phrases from text"""
    financial_terms = [
        'revenue growth', 'profit margin', 'cash flow', 'debt', 'equity',
        'market share', 'competitive advantage', 'risk factors', 'outlook',
        'guidance', 'earnings', 'ebitda', 'capex', 'working capital'
    ]
    
    text_lower = text.lower()
    found_phrases = []
    
    for term in financial_terms:
        if term in text_lower:
            found_phrases.append(term)
    
    return found_phrases[:10]  # Return top 10

def classify_sentiment(sentiment_score: float) -> str:
    """Classify sentiment score into label"""
    if sentiment_score > 0.3:
        return "POSITIVE"
    elif sentiment_score < -0.3:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def generate_risk_recommendations(risk_level: str, features: Dict[str, Any]) -> List[str]:
    """Generate risk-based recommendations"""
    recommendations = []
    
    if risk_level in ['HIGH', 'CRITICAL']:
        recommendations.append("Consider delaying IPO until risk factors are addressed")
        recommendations.append("Engage risk management consultants")
    
    if features.get('debt_to_equity', 0) > 2.0:
        recommendations.append("Reduce debt levels before going public")
    
    if features.get('current_ratio', 0) < 1.0:
        recommendations.append("Improve liquidity position")
    
    return recommendations

def generate_investment_recommendation(valuation: float, risk_assessment: Dict[str, Any]) -> str:
    """Generate investment recommendation"""
    risk_score = risk_assessment['risk_score']
    
    if risk_score > 0.7:
        return "HIGH_RISK"
    elif risk_score > 0.4:
        return "MODERATE_RISK"
    else:
        return "LOW_RISK"

def generate_request_id() -> str:
    """Generate unique request ID"""
    import uuid
    return str(uuid.uuid4())

def get_model_version(model_type: str) -> str:
    """Get model version for tracking"""
    # This would typically query model registry
    return "v1.0"

def log_request(request_id: str, model_type: str, company_id: str) -> None:
    """Log incoming request"""
    log_entry = {
        'request_id': request_id,
        'model_type': model_type,
        'company_id': company_id,
        'timestamp': datetime.utcnow().isoformat(),
        'severity': 'INFO'
    }
    logger.info(json.dumps(log_entry))

def log_prediction(request_id: str, model_type: str, result: Dict[str, Any]) -> None:
    """Log prediction result"""
    log_entry = {
        'request_id': request_id,
        'model_type': model_type,
        'prediction_timestamp': datetime.utcnow().isoformat(),
        'confidence_score': result.get('confidence_score'),
        'severity': 'INFO'
    }
    logger.info(json.dumps(log_entry))

def log_error(request_id: str, model_type: str, error_message: str) -> None:
    """Log error"""
    log_entry = {
        'request_id': request_id,
        'model_type': model_type,
        'error_message': error_message,
        'timestamp': datetime.utcnow().isoformat(),
        'severity': 'ERROR'
    }
    logger.error(json.dumps(log_entry))

def send_prediction_metric(model_type: str, success: bool) -> None:
    """Send custom metrics to Cloud Monitoring"""
    try:
        project_name = f"projects/PROJECT_ID"
        
        # Create time series data
        now = datetime.utcnow()
        seconds = int(now.timestamp())
        nanos = int((now.timestamp() - seconds) * 10**9)
        
        interval = monitoring_v3.TimeInterval({
            "end_time": {"seconds": seconds, "nanos": nanos}
        })
        
        point = monitoring_v3.Point({
            "interval": interval,
            "value": {"int64_value": 1 if success else 0}
        })
        
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/ipo_valuation/predictions"
        series.resource.type = "cloud_function"
        series.resource.labels["function_name"] = "ml-inference"
        series.metric.labels["model_type"] = model_type
        series.metric.labels["status"] = "success" if success else "error"
        series.points = [point]
        
        monitoring_client.create_time_series(
            name=project_name,
            time_series=[series]
        )
        
    except Exception as e:
        logger.error(f"Error sending metric: {str(e)}")
        # Don't raise - metrics failure shouldn't break inference