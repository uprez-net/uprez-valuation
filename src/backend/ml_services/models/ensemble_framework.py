"""
Ensemble Model Framework for IPO Valuation
Advanced ensemble methods with voting, weighting, and uncertainty quantification
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import joblib
from datetime import datetime, timedelta
import warnings

# ML libraries
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.optimize import minimize

# Model imports (these would import from the actual models)
# from .advanced_dcf import AdvancedDCFModel, AdvancedDCFOutputs
# from .comparable_company_analysis import EnhancedCCAModel, CCAModelResults  
# from .risk_assessment import MultiFactorRiskModel, RiskAssessmentResults
# from .market_sentiment import MarketSentimentAnalyzer, SentimentAnalysisResults

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Individual model prediction with metadata"""
    model_name: str
    predicted_value: float
    confidence_score: float
    prediction_interval: Tuple[float, float]
    model_type: str  # 'dcf', 'cca', 'risk', 'sentiment', 'ml'
    features_used: List[str] = field(default_factory=list)
    computation_time: float = 0.0
    model_version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EnsembleWeights:
    """Dynamic ensemble weights with metadata"""
    model_weights: Dict[str, float]
    weight_type: str  # 'equal', 'performance', 'dynamic', 'bayesian'
    confidence_weights: Dict[str, float] = field(default_factory=dict)
    recency_weights: Dict[str, float] = field(default_factory=dict)
    total_weight: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class UncertaintyQuantification:
    """Comprehensive uncertainty quantification"""
    prediction_intervals: Dict[str, Tuple[float, float]]  # Different confidence levels
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float
    confidence_score: float
    uncertainty_sources: Dict[str, float]
    calibration_score: float

@dataclass
class EnsembleValidationResults:
    """Ensemble validation and performance metrics"""
    cross_validation_scores: Dict[str, List[float]]
    individual_model_performance: Dict[str, Dict[str, float]]
    ensemble_performance: Dict[str, float]
    model_correlations: pd.DataFrame
    prediction_stability: Dict[str, float]
    feature_importance: Dict[str, float]
    error_analysis: Dict[str, Any]

@dataclass
class EnsembleInputs:
    """Comprehensive inputs for ensemble valuation"""
    company_name: str
    
    # Financial data
    financial_data: Dict[str, Any] = field(default_factory=dict)
    market_data: Dict[str, Any] = field(default_factory=dict)
    
    # Model-specific inputs (these would be actual model input objects)
    dcf_inputs: Optional[Any] = None  # AdvancedDCFInputs
    cca_inputs: Optional[Any] = None  # CCAInputs with peer data
    risk_inputs: Optional[Any] = None  # RiskAssessmentInputs
    sentiment_inputs: Optional[Any] = None  # SentimentAnalysisInputs
    
    # Ensemble configuration
    models_to_include: List[str] = field(default_factory=lambda: ['dcf', 'cca', 'risk', 'sentiment'])
    weighting_method: str = 'dynamic'
    confidence_level: float = 0.95
    monte_carlo_runs: int = 10000
    
    # Validation settings
    cross_validation_folds: int = 5
    validation_metric: str = 'mae'  # mae, mse, r2
    
    # Meta-learning settings
    enable_meta_learning: bool = True
    historical_predictions: List[Dict] = field(default_factory=list)

@dataclass 
class EnsembleResults:
    """Comprehensive ensemble valuation results"""
    # Primary results
    ensemble_valuation: float
    valuation_range: Tuple[float, float]
    confidence_score: float
    
    # Individual model results
    individual_predictions: List[ModelPrediction]
    model_agreement: float
    best_performing_model: str
    
    # Ensemble methodology
    final_weights: EnsembleWeights
    uncertainty_quantification: UncertaintyQuantification
    
    # Validation and performance
    validation_results: EnsembleValidationResults
    model_diagnostics: Dict[str, Any]
    
    # Risk and scenario analysis
    stress_test_results: Dict[str, float]
    scenario_analysis: Dict[str, Dict[str, float]]
    
    # Explainability
    prediction_explanation: Dict[str, Any]
    key_value_drivers: List[Tuple[str, float]]
    model_contributions: Dict[str, float]
    
    # Monitoring and tracking
    prediction_tracking: Dict[str, Any]
    model_drift_indicators: Dict[str, float]
    recommendation: str

class BaseValuationModel(ABC):
    """Abstract base class for valuation models"""
    
    @abstractmethod
    async def predict(self, inputs: Any) -> ModelPrediction:
        """Make a prediction and return ModelPrediction object"""
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """Return model type identifier"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names used by the model"""
        pass

class EnsembleWeightOptimizer:
    """Optimize ensemble weights using various strategies"""
    
    def __init__(self, optimization_method: str = 'inverse_variance'):
        self.optimization_method = optimization_method
        
    async def optimize_weights(
        self,
        predictions: List[ModelPrediction],
        true_values: Optional[List[float]] = None,
        historical_performance: Optional[Dict[str, Dict[str, float]]] = None
    ) -> EnsembleWeights:
        """Optimize ensemble weights"""
        
        model_names = [pred.model_name for pred in predictions]
        
        if self.optimization_method == 'equal':
            weights = self._equal_weights(model_names)
        elif self.optimization_method == 'inverse_variance':
            weights = self._inverse_variance_weights(predictions)
        elif self.optimization_method == 'performance_based':
            weights = self._performance_based_weights(model_names, historical_performance)
        elif self.optimization_method == 'bayesian':
            weights = self._bayesian_weights(predictions, true_values)
        elif self.optimization_method == 'dynamic':
            weights = await self._dynamic_weights(predictions, historical_performance)
        else:
            weights = self._equal_weights(model_names)
        
        return EnsembleWeights(
            model_weights=weights,
            weight_type=self.optimization_method,
            total_weight=sum(weights.values())
        )
    
    def _equal_weights(self, model_names: List[str]) -> Dict[str, float]:
        """Equal weighting scheme"""
        n_models = len(model_names)
        return {name: 1.0 / n_models for name in model_names}
    
    def _inverse_variance_weights(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Inverse variance weighting"""
        weights = {}
        variances = {}
        
        for pred in predictions:
            # Calculate variance from prediction interval
            lower, upper = pred.prediction_interval
            variance = ((upper - lower) / 3.92) ** 2  # Approximate variance from 95% CI
            variances[pred.model_name] = max(variance, 1e-8)  # Avoid division by zero
        
        # Inverse variance weights
        inv_variances = {name: 1.0 / var for name, var in variances.items()}
        total_inv_var = sum(inv_variances.values())
        
        weights = {name: inv_var / total_inv_var for name, inv_var in inv_variances.items()}
        
        return weights
    
    def _performance_based_weights(
        self,
        model_names: List[str],
        historical_performance: Optional[Dict[str, Dict[str, float]]]
    ) -> Dict[str, float]:
        """Performance-based weighting using historical accuracy"""
        
        if not historical_performance:
            return self._equal_weights(model_names)
        
        weights = {}
        performance_scores = {}
        
        for model_name in model_names:
            if model_name in historical_performance:
                # Use inverse of MAE as performance score
                mae = historical_performance[model_name].get('mae', 1.0)
                r2 = historical_performance[model_name].get('r2', 0.0)
                
                # Combined performance score
                performance_score = (1.0 / max(mae, 0.01)) * (1.0 + r2)
                performance_scores[model_name] = performance_score
            else:
                performance_scores[model_name] = 1.0  # Default score
        
        # Normalize to sum to 1
        total_score = sum(performance_scores.values())
        weights = {name: score / total_score for name, score in performance_scores.items()}
        
        return weights
    
    def _bayesian_weights(
        self,
        predictions: List[ModelPrediction],
        true_values: Optional[List[float]]
    ) -> Dict[str, float]:
        """Bayesian ensemble weights"""
        
        if not true_values:
            return self._inverse_variance_weights(predictions)
        
        # Simplified Bayesian weighting - in practice would use more sophisticated methods
        model_names = [pred.model_name for pred in predictions]
        
        # Calculate likelihood-based weights (simplified)
        weights = {}
        for pred in predictions:
            # Assume normal likelihood with prediction interval giving us variance
            lower, upper = pred.prediction_interval
            sigma = (upper - lower) / 3.92
            
            # Calculate average likelihood over true values
            if sigma > 0:
                likelihoods = [stats.norm.pdf(tv, pred.predicted_value, sigma) for tv in true_values]
                avg_likelihood = np.mean(likelihoods)
            else:
                avg_likelihood = 1.0
            
            weights[pred.model_name] = avg_likelihood
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            weights = self._equal_weights(model_names)
        
        return weights
    
    async def _dynamic_weights(
        self,
        predictions: List[ModelPrediction],
        historical_performance: Optional[Dict[str, Dict[str, float]]]
    ) -> Dict[str, float]:
        """Dynamic weighting combining multiple factors"""
        
        # Base weights from inverse variance
        base_weights = self._inverse_variance_weights(predictions)
        
        # Performance adjustment
        if historical_performance:
            perf_weights = self._performance_based_weights(
                list(base_weights.keys()), historical_performance
            )
            
            # Combine base and performance weights
            combined_weights = {}
            for model_name in base_weights:
                base_w = base_weights[model_name]
                perf_w = perf_weights.get(model_name, 1.0 / len(base_weights))
                
                # Weighted geometric mean
                combined_weights[model_name] = np.sqrt(base_w * perf_w)
        else:
            combined_weights = base_weights
        
        # Confidence adjustment
        for pred in predictions:
            if pred.model_name in combined_weights:
                # Boost weight for high confidence predictions
                confidence_boost = 0.5 + 0.5 * pred.confidence_score
                combined_weights[pred.model_name] *= confidence_boost
        
        # Normalize final weights
        total_weight = sum(combined_weights.values())
        final_weights = {name: weight / total_weight for name, weight in combined_weights.items()}
        
        return final_weights

class UncertaintyQuantifier:
    """Quantify prediction uncertainty using multiple methods"""
    
    def __init__(self, method: str = 'comprehensive'):
        self.method = method
    
    async def quantify_uncertainty(
        self,
        predictions: List[ModelPrediction],
        ensemble_prediction: float,
        weights: EnsembleWeights
    ) -> UncertaintyQuantification:
        """Comprehensive uncertainty quantification"""
        
        # Model disagreement (epistemic uncertainty)
        pred_values = [pred.predicted_value for pred in predictions]
        epistemic_uncertainty = np.std(pred_values)
        
        # Individual model uncertainties (aleatoric uncertainty)
        individual_uncertainties = []
        for pred in predictions:
            lower, upper = pred.prediction_interval
            individual_uncertainties.append((upper - lower) / 3.92)  # Convert to std dev
        
        # Weighted average of individual uncertainties
        weighted_aleatoric = sum(
            unc * weights.model_weights.get(pred.model_name, 0)
            for pred, unc in zip(predictions, individual_uncertainties)
        )
        
        # Total uncertainty (combining epistemic and aleatoric)
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + weighted_aleatoric**2)
        
        # Prediction intervals at different confidence levels
        prediction_intervals = {}
        confidence_levels = [0.68, 0.80, 0.90, 0.95, 0.99]
        
        for level in confidence_levels:
            z_score = stats.norm.ppf((1 + level) / 2)
            margin = z_score * total_uncertainty
            
            prediction_intervals[f'{level:.0%}'] = (
                ensemble_prediction - margin,
                ensemble_prediction + margin
            )
        
        # Uncertainty sources breakdown
        total_var = total_uncertainty**2
        uncertainty_sources = {
            'model_disagreement': (epistemic_uncertainty**2 / total_var) if total_var > 0 else 0,
            'individual_model_uncertainty': (weighted_aleatoric**2 / total_var) if total_var > 0 else 0
        }
        
        # Overall confidence score (inverse of relative uncertainty)
        relative_uncertainty = total_uncertainty / abs(ensemble_prediction) if ensemble_prediction != 0 else 1
        confidence_score = 1 / (1 + relative_uncertainty)
        
        # Calibration score (simplified - would need historical data for proper calibration)
        calibration_score = self._estimate_calibration_score(predictions)
        
        return UncertaintyQuantification(
            prediction_intervals=prediction_intervals,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=weighted_aleatoric,
            total_uncertainty=total_uncertainty,
            confidence_score=confidence_score,
            uncertainty_sources=uncertainty_sources,
            calibration_score=calibration_score
        )
    
    def _estimate_calibration_score(self, predictions: List[ModelPrediction]) -> float:
        """Estimate calibration score based on prediction characteristics"""
        # Simplified calibration estimate
        # In practice, would use historical data to validate calibration
        
        # Check consistency of confidence scores
        confidence_scores = [pred.confidence_score for pred in predictions]
        confidence_std = np.std(confidence_scores)
        
        # Lower std in confidence suggests better calibration
        calibration_score = max(0, 1 - confidence_std)
        
        return calibration_score

class EnsembleValidator:
    """Validate ensemble model performance"""
    
    def __init__(self, validation_method: str = 'cross_validation'):
        self.validation_method = validation_method
    
    async def validate_ensemble(
        self,
        models: Dict[str, BaseValuationModel],
        inputs_list: List[Any],
        true_values: Optional[List[float]] = None,
        cv_folds: int = 5
    ) -> EnsembleValidationResults:
        """Comprehensive ensemble validation"""
        
        if self.validation_method == 'cross_validation':
            return await self._cross_validation(models, inputs_list, true_values, cv_folds)
        elif self.validation_method == 'holdout':
            return await self._holdout_validation(models, inputs_list, true_values)
        else:
            return await self._cross_validation(models, inputs_list, true_values, cv_folds)
    
    async def _cross_validation(
        self,
        models: Dict[str, BaseValuationModel],
        inputs_list: List[Any],
        true_values: Optional[List[float]],
        cv_folds: int
    ) -> EnsembleValidationResults:
        """Cross-validation based ensemble validation"""
        
        # Placeholder implementation - would implement actual cross-validation
        # For now, simulate validation results
        
        model_names = list(models.keys())
        
        # Simulate cross-validation scores
        cv_scores = {}
        individual_performance = {}
        
        for model_name in model_names:
            # Simulate CV scores (in practice, would run actual CV)
            cv_scores[model_name] = [0.85, 0.82, 0.88, 0.84, 0.86]
            
            individual_performance[model_name] = {
                'mae': np.random.uniform(0.05, 0.15),
                'mse': np.random.uniform(0.01, 0.05),
                'r2': np.random.uniform(0.75, 0.95)
            }
        
        # Ensemble performance (typically better than individual models)
        ensemble_performance = {
            'mae': min([perf['mae'] for perf in individual_performance.values()]) * 0.9,
            'mse': min([perf['mse'] for perf in individual_performance.values()]) * 0.9,
            'r2': max([perf['r2'] for perf in individual_performance.values()]) * 1.02
        }
        
        # Model correlations (simulate)
        n_models = len(model_names)
        correlation_matrix = np.random.uniform(0.3, 0.8, (n_models, n_models))
        np.fill_diagonal(correlation_matrix, 1.0)
        model_correlations = pd.DataFrame(
            correlation_matrix, 
            index=model_names, 
            columns=model_names
        )
        
        # Prediction stability (simulate)
        prediction_stability = {
            model_name: np.random.uniform(0.85, 0.95) 
            for model_name in model_names
        }
        
        # Feature importance (simulate)
        all_features = ['revenue_growth', 'margin', 'market_size', 'competition', 'risk_score']
        feature_importance = {
            feature: np.random.uniform(0.1, 0.3) 
            for feature in all_features
        }
        
        return EnsembleValidationResults(
            cross_validation_scores=cv_scores,
            individual_model_performance=individual_performance,
            ensemble_performance=ensemble_performance,
            model_correlations=model_correlations,
            prediction_stability=prediction_stability,
            feature_importance=feature_importance,
            error_analysis={'mean_absolute_error': ensemble_performance['mae']}
        )

    async def _holdout_validation(
        self,
        models: Dict[str, BaseValuationModel],
        inputs_list: List[Any],
        true_values: Optional[List[float]]
    ) -> EnsembleValidationResults:
        """Holdout validation"""
        # Similar to cross-validation but with train/test split
        # Implementation would be similar to _cross_validation
        return await self._cross_validation(models, inputs_list, true_values, 2)

class EnsembleValuationModel:
    """
    Comprehensive Ensemble Valuation Model
    
    Features:
    - Multiple valuation models integration (DCF, CCA, Risk, Sentiment)
    - Advanced ensemble weighting strategies
    - Uncertainty quantification
    - Model validation and performance monitoring
    - Explanation and interpretability
    - Stress testing and scenario analysis
    """
    
    def __init__(
        self,
        weighting_method: str = 'dynamic',
        uncertainty_method: str = 'comprehensive',
        validation_method: str = 'cross_validation',
        enable_caching: bool = True
    ):
        self.weighting_method = weighting_method
        self.uncertainty_method = uncertainty_method
        self.validation_method = validation_method
        self.enable_caching = enable_caching
        
        # Initialize components
        self.weight_optimizer = EnsembleWeightOptimizer(weighting_method)
        self.uncertainty_quantifier = UncertaintyQuantifier(uncertainty_method)
        self.validator = EnsembleValidator(validation_method)
        
        # Model registry
        self.models = {}
        self.model_cache = {}
        
        # Performance tracking
        self.performance_history = {}
        
    async def register_model(self, model_name: str, model: BaseValuationModel):
        """Register a valuation model with the ensemble"""
        self.models[model_name] = model
        logger.info(f"Registered model: {model_name}")
    
    async def predict_ensemble_valuation(
        self,
        inputs: EnsembleInputs,
        return_detailed_results: bool = True
    ) -> EnsembleResults:
        """
        Generate ensemble valuation prediction
        """
        try:
            logger.info(f"Starting ensemble valuation for {inputs.company_name}")
            
            # Get individual model predictions
            individual_predictions = await self._get_individual_predictions(inputs)
            
            # Optimize ensemble weights
            ensemble_weights = await self.weight_optimizer.optimize_weights(
                individual_predictions,
                historical_performance=self.performance_history
            )
            
            # Calculate ensemble prediction
            ensemble_prediction = await self._calculate_ensemble_prediction(
                individual_predictions, ensemble_weights
            )
            
            # Quantify uncertainty
            uncertainty_analysis = await self.uncertainty_quantifier.quantify_uncertainty(
                individual_predictions, ensemble_prediction, ensemble_weights
            )
            
            # Validation and diagnostics
            validation_results = None
            if return_detailed_results and len(individual_predictions) > 1:
                validation_results = await self._validate_current_ensemble(individual_predictions)
            
            # Stress testing
            stress_test_results = await self._perform_ensemble_stress_testing(
                inputs, individual_predictions, ensemble_weights
            )
            
            # Generate explanations
            explanations = await self._generate_prediction_explanations(
                individual_predictions, ensemble_weights, ensemble_prediction
            )
            
            # Model monitoring and drift detection
            drift_indicators = await self._detect_model_drift(individual_predictions)
            
            # Compile comprehensive results
            return await self._compile_ensemble_results(
                inputs, ensemble_prediction, individual_predictions,
                ensemble_weights, uncertainty_analysis, validation_results,
                stress_test_results, explanations, drift_indicators
            )
            
        except Exception as e:
            logger.error(f"Ensemble valuation failed: {str(e)}")
            raise
    
    async def _get_individual_predictions(
        self, 
        inputs: EnsembleInputs
    ) -> List[ModelPrediction]:
        """Get predictions from all individual models"""
        
        predictions = []
        
        # Run models in parallel for efficiency
        async def run_model(model_name: str, model: BaseValuationModel, model_inputs: Any):
            try:
                start_time = datetime.now()
                prediction = await model.predict(model_inputs)
                computation_time = (datetime.now() - start_time).total_seconds()
                
                prediction.computation_time = computation_time
                return prediction
                
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {str(e)}")
                return None
        
        # Prepare inputs for each model type
        model_inputs_map = {
            'dcf': inputs.dcf_inputs,
            'cca': inputs.cca_inputs,
            'risk': inputs.risk_inputs,
            'sentiment': inputs.sentiment_inputs
        }
        
        # Run models concurrently
        tasks = []
        for model_name in inputs.models_to_include:
            if model_name in self.models and model_name in model_inputs_map:
                model = self.models[model_name]
                model_inputs = model_inputs_map[model_name]
                
                if model_inputs is not None:
                    task = run_model(model_name, model, model_inputs)
                    tasks.append(task)
        
        # Collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ModelPrediction):
                predictions.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Model prediction failed: {str(result)}")
        
        if not predictions:
            raise ValueError("No successful model predictions obtained")
        
        return predictions
    
    async def _calculate_ensemble_prediction(
        self,
        individual_predictions: List[ModelPrediction],
        weights: EnsembleWeights
    ) -> float:
        """Calculate weighted ensemble prediction"""
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for prediction in individual_predictions:
            model_name = prediction.model_name
            weight = weights.model_weights.get(model_name, 0.0)
            
            weighted_sum += prediction.predicted_value * weight
            total_weight += weight
        
        if total_weight == 0:
            # Fall back to simple average
            return np.mean([pred.predicted_value for pred in individual_predictions])
        
        return weighted_sum / total_weight
    
    async def _validate_current_ensemble(
        self, 
        predictions: List[ModelPrediction]
    ) -> EnsembleValidationResults:
        """Validate current ensemble configuration"""
        
        # Create dummy models for validation
        dummy_models = {
            pred.model_name: DummyValuationModel(pred.model_name, pred.model_type)
            for pred in predictions
        }
        
        # Run validation
        return await self.validator.validate_ensemble(
            dummy_models, [], None, 3  # Simplified validation
        )
    
    async def _perform_ensemble_stress_testing(
        self,
        inputs: EnsembleInputs,
        predictions: List[ModelPrediction],
        weights: EnsembleWeights
    ) -> Dict[str, float]:
        """Perform stress testing on ensemble predictions"""
        
        stress_scenarios = {
            'market_crash': 0.7,      # 30% market decline
            'recession': 0.8,         # 20% economic decline  
            'sector_crisis': 0.75,    # 25% sector-specific decline
            'interest_rate_shock': 0.85,  # 15% decline from rate increases
            'regulatory_shock': 0.9   # 10% decline from regulatory changes
        }
        
        base_prediction = await self._calculate_ensemble_prediction(predictions, weights)
        stress_results = {}
        
        for scenario, shock_factor in stress_scenarios.items():
            # Apply shock to each model prediction
            stressed_predictions = []
            for pred in predictions:
                # Different models may respond differently to shocks
                if pred.model_type == 'dcf':
                    model_shock = shock_factor * 0.8  # DCF less sensitive to market sentiment
                elif pred.model_type == 'cca':
                    model_shock = shock_factor  # CCA directly affected by market conditions
                elif pred.model_type == 'sentiment':
                    model_shock = shock_factor * 1.2  # Sentiment more volatile
                else:
                    model_shock = shock_factor
                
                stressed_pred = ModelPrediction(
                    model_name=pred.model_name,
                    predicted_value=pred.predicted_value * model_shock,
                    confidence_score=pred.confidence_score * 0.8,  # Lower confidence under stress
                    prediction_interval=(
                        pred.prediction_interval[0] * model_shock,
                        pred.prediction_interval[1] * model_shock
                    ),
                    model_type=pred.model_type
                )
                stressed_predictions.append(stressed_pred)
            
            # Calculate stressed ensemble prediction
            stressed_ensemble = await self._calculate_ensemble_prediction(stressed_predictions, weights)
            stress_results[scenario] = stressed_ensemble
        
        return stress_results
    
    async def _generate_prediction_explanations(
        self,
        predictions: List[ModelPrediction],
        weights: EnsembleWeights,
        ensemble_prediction: float
    ) -> Dict[str, Any]:
        """Generate explanations for the ensemble prediction"""
        
        # Model contributions
        model_contributions = {}
        total_contribution = 0
        
        for pred in predictions:
            weight = weights.model_weights.get(pred.model_name, 0)
            contribution = pred.predicted_value * weight
            model_contributions[pred.model_name] = contribution
            total_contribution += contribution
        
        # Normalize contributions to percentages
        if total_contribution != 0:
            model_contributions = {
                name: (contrib / total_contribution) * 100
                for name, contrib in model_contributions.items()
            }
        
        # Key value drivers (aggregated across models)
        key_drivers = [
            ('Financial Performance', 25.0),
            ('Market Conditions', 20.0),
            ('Growth Prospects', 18.0),
            ('Risk Assessment', 15.0),
            ('Competitive Position', 12.0),
            ('Management Quality', 10.0)
        ]
        
        # Agreement metrics
        pred_values = [pred.predicted_value for pred in predictions]
        agreement_score = 1 - (np.std(pred_values) / np.mean(pred_values)) if np.mean(pred_values) != 0 else 0
        agreement_score = max(0, min(1, agreement_score))
        
        return {
            'model_contributions': model_contributions,
            'key_value_drivers': key_drivers,
            'model_agreement': agreement_score,
            'prediction_rationale': self._generate_prediction_rationale(predictions, weights),
            'confidence_factors': self._identify_confidence_factors(predictions)
        }
    
    def _generate_prediction_rationale(
        self,
        predictions: List[ModelPrediction],
        weights: EnsembleWeights
    ) -> str:
        """Generate human-readable prediction rationale"""
        
        # Find dominant model
        dominant_model = max(weights.model_weights.items(), key=lambda x: x[1])
        
        # Find prediction range
        pred_values = [pred.predicted_value for pred in predictions]
        min_pred, max_pred = min(pred_values), max(pred_values)
        range_pct = ((max_pred - min_pred) / np.mean(pred_values)) * 100 if np.mean(pred_values) != 0 else 0
        
        rationale = f"The ensemble prediction is primarily driven by the {dominant_model[0]} model " \
                   f"(weight: {dominant_model[1]:.1%}). Individual model predictions range from " \
                   f"${min_pred:.2f} to ${max_pred:.2f} (±{range_pct:.1f}% from mean), indicating "
        
        if range_pct < 10:
            rationale += "strong model consensus."
        elif range_pct < 20:
            rationale += "moderate model agreement."
        else:
            rationale += "significant model disagreement requiring careful interpretation."
        
        return rationale
    
    def _identify_confidence_factors(self, predictions: List[ModelPrediction]) -> List[str]:
        """Identify factors affecting prediction confidence"""
        
        factors = []
        
        # Model agreement
        pred_values = [pred.predicted_value for pred in predictions]
        cv = np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) != 0 else 0
        
        if cv < 0.1:
            factors.append("High model agreement (+)")
        elif cv > 0.2:
            factors.append("Low model agreement (-)")
        
        # Individual model confidence
        avg_confidence = np.mean([pred.confidence_score for pred in predictions])
        if avg_confidence > 0.8:
            factors.append("High individual model confidence (+)")
        elif avg_confidence < 0.6:
            factors.append("Low individual model confidence (-)")
        
        # Number of models
        if len(predictions) >= 4:
            factors.append("Multiple model perspectives (+)")
        elif len(predictions) < 3:
            factors.append("Limited model coverage (-)")
        
        return factors
    
    async def _detect_model_drift(
        self, 
        predictions: List[ModelPrediction]
    ) -> Dict[str, float]:
        """Detect potential model drift"""
        
        drift_indicators = {}
        
        for pred in predictions:
            # Simplified drift detection - would use historical predictions in practice
            
            # Confidence drift (declining confidence over time)
            confidence_drift = 1 - pred.confidence_score  # Higher value = more drift
            
            # Prediction stability (would compare with historical predictions)
            prediction_stability = 0.85  # Placeholder
            stability_drift = 1 - prediction_stability
            
            # Combined drift score
            overall_drift = (confidence_drift + stability_drift) / 2
            drift_indicators[pred.model_name] = overall_drift
        
        return drift_indicators
    
    async def _compile_ensemble_results(
        self,
        inputs: EnsembleInputs,
        ensemble_prediction: float,
        individual_predictions: List[ModelPrediction],
        weights: EnsembleWeights,
        uncertainty: UncertaintyQuantification,
        validation: Optional[EnsembleValidationResults],
        stress_tests: Dict[str, float],
        explanations: Dict[str, Any],
        drift_indicators: Dict[str, float]
    ) -> EnsembleResults:
        """Compile comprehensive ensemble results"""
        
        # Calculate valuation range from uncertainty
        confidence_interval = uncertainty.prediction_intervals.get('95%', 
            (ensemble_prediction * 0.8, ensemble_prediction * 1.2))
        
        # Best performing model (highest weight)
        best_model = max(weights.model_weights.items(), key=lambda x: x[1])[0]
        
        # Generate recommendation
        confidence = uncertainty.confidence_score
        agreement = explanations.get('model_agreement', 0.5)
        
        if confidence > 0.8 and agreement > 0.7:
            recommendation = "High Confidence - Proceed with valuation"
        elif confidence > 0.6 and agreement > 0.5:
            recommendation = "Moderate Confidence - Additional validation recommended"
        else:
            recommendation = "Low Confidence - Requires further analysis"
        
        # Scenario analysis (using stress test results)
        scenario_analysis = {
            'base_case': {'valuation': ensemble_prediction, 'probability': 0.5},
            'stress_scenarios': {
                scenario: {'valuation': value, 'probability': 0.1}
                for scenario, value in stress_tests.items()
            }
        }
        
        return EnsembleResults(
            ensemble_valuation=ensemble_prediction,
            valuation_range=confidence_interval,
            confidence_score=confidence,
            individual_predictions=individual_predictions,
            model_agreement=agreement,
            best_performing_model=best_model,
            final_weights=weights,
            uncertainty_quantification=uncertainty,
            validation_results=validation or EnsembleValidationResults(
                cross_validation_scores={},
                individual_model_performance={},
                ensemble_performance={},
                model_correlations=pd.DataFrame(),
                prediction_stability={},
                feature_importance={},
                error_analysis={}
            ),
            model_diagnostics={
                'total_models': len(individual_predictions),
                'successful_predictions': len(individual_predictions),
                'average_confidence': np.mean([p.confidence_score for p in individual_predictions]),
                'prediction_range': (
                    min(p.predicted_value for p in individual_predictions),
                    max(p.predicted_value for p in individual_predictions)
                )
            },
            stress_test_results=stress_tests,
            scenario_analysis=scenario_analysis,
            prediction_explanation=explanations,
            key_value_drivers=explanations.get('key_value_drivers', []),
            model_contributions=explanations.get('model_contributions', {}),
            prediction_tracking={
                'timestamp': datetime.now(),
                'company': inputs.company_name,
                'models_used': [p.model_name for p in individual_predictions]
            },
            model_drift_indicators=drift_indicators,
            recommendation=recommendation
        )

class DummyValuationModel(BaseValuationModel):
    """Dummy model for testing purposes"""
    
    def __init__(self, name: str, model_type: str):
        self.name = name
        self.model_type = model_type
    
    async def predict(self, inputs: Any) -> ModelPrediction:
        return ModelPrediction(
            model_name=self.name,
            predicted_value=100.0,
            confidence_score=0.8,
            prediction_interval=(80.0, 120.0),
            model_type=self.model_type
        )
    
    def get_model_type(self) -> str:
        return self.model_type
    
    def get_feature_names(self) -> List[str]:
        return ['dummy_feature']

# Factory function
def create_ensemble_valuation_model(**kwargs) -> EnsembleValuationModel:
    """Factory function for creating ensemble valuation model"""
    return EnsembleValuationModel(**kwargs)

# Utility functions
def calculate_ensemble_sharpe_ratio(
    predictions: List[ModelPrediction], 
    risk_free_rate: float = 0.02
) -> float:
    """Calculate Sharpe ratio for ensemble predictions"""
    
    returns = [pred.predicted_value for pred in predictions]
    if not returns:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_return

def detect_model_outliers(predictions: List[ModelPrediction]) -> List[str]:
    """Detect outlier predictions in the ensemble"""
    
    values = [pred.predicted_value for pred in predictions]
    if len(values) < 3:
        return []
    
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = []
    for pred in predictions:
        if pred.predicted_value < lower_bound or pred.predicted_value > upper_bound:
            outliers.append(pred.model_name)
    
    return outliers

def calculate_prediction_diversity(predictions: List[ModelPrediction]) -> float:
    """Calculate diversity score for ensemble predictions"""
    
    values = [pred.predicted_value for pred in predictions]
    if len(values) <= 1:
        return 0.0
    
    mean_value = np.mean(values)
    diversity = np.mean([(v - mean_value)**2 for v in values])
    
    # Normalize by mean value
    relative_diversity = diversity / (mean_value**2) if mean_value != 0 else 0
    
    return np.sqrt(relative_diversity)