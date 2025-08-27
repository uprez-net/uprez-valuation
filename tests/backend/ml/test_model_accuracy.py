"""
ML Model Accuracy Testing Suite

Tests for validating the accuracy and performance of machine learning models
used in the IPO valuation platform, including DCF models, peer analysis,
sentiment analysis, and risk assessment models.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
import joblib
import json
import tempfile
import os

from tests.utils.test_data import TestDataManager, get_standard_test_scenario
from tests.utils.mocks import MockVertexAI, MockDocumentAI
from tests.utils.factories import CompanyFactory, MarketDataFactory, ValuationFactory


class TestMLModelAccuracy:
    """Test ML model accuracy against validation datasets."""
    
    def setup_method(self):
        """Set up test data and models for each test method."""
        self.test_data_manager = TestDataManager()
        self.validation_tolerance = 0.15  # 15% tolerance for predictions
        
        # Create synthetic validation dataset
        np.random.seed(42)  # For reproducible results
        self.validation_data = self.test_data_manager.create_ml_training_data(samples=200)
        
        # Mock model paths
        self.model_paths = {
            'valuation_model': 'models/valuation_rf_model.joblib',
            'risk_model': 'models/risk_xgb_model.joblib',
            'sentiment_model': 'models/sentiment_bert_model',
            'peer_matching_model': 'models/peer_similarity_model.joblib'
        }
    
    def test_valuation_model_accuracy(self):
        """Test valuation model accuracy against validation dataset."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        from src.backend.ml_services.predictive_models.valuation_predictor import ValuationPredictor
        
        # Load test model (would be actual trained model in production)
        with patch('joblib.load') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.uniform(0.5, 20.0, 100)
            mock_load.return_value = mock_model
            
            predictor = ValuationPredictor()
            
            # Test with synthetic company data
            test_companies = [CompanyFactory.build() for _ in range(100)]
            
            predictions = []
            actuals = []
            
            for company in test_companies:
                # Generate prediction
                features = predictor.extract_features(company)
                prediction = predictor.predict_valuation(features)
                predictions.append(prediction)
                
                # Generate "actual" valuation for comparison (in real scenario, this would be known IPO price)
                actual_valuation = self._generate_synthetic_actual_valuation(company)
                actuals.append(actual_valuation)
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate accuracy metrics
            mae = np.mean(np.abs(predictions - actuals))
            mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            
            # Accuracy thresholds
            assert mae < 2.0, f"Mean Absolute Error {mae} exceeds threshold of 2.0"
            assert mape < 25.0, f"Mean Absolute Percentage Error {mape}% exceeds threshold of 25%"
            assert rmse < 3.0, f"Root Mean Square Error {rmse} exceeds threshold of 3.0"
            
            # Correlation test
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            assert correlation > 0.7, f"Correlation {correlation} below minimum threshold of 0.7"
    
    def test_risk_model_classification_accuracy(self):
        """Test risk assessment model classification accuracy."""
        from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
        
        with patch('joblib.load') as mock_load:
            mock_model = Mock()
            # Mock predictions: 0=Low, 1=Medium, 2=High risk
            mock_predictions = np.random.choice([0, 1, 2], size=100, p=[0.3, 0.5, 0.2])
            mock_model.predict.return_value = mock_predictions
            mock_model.predict_proba.return_value = np.random.dirichlet(np.ones(3), size=100)
            mock_load.return_value = mock_model
            
            risk_model = RiskAssessmentModel()
            
            # Generate test cases with known risk levels
            test_cases = self._generate_risk_test_cases(100)
            
            correct_predictions = 0
            total_predictions = len(test_cases)
            
            for i, (company_profile, expected_risk) in enumerate(test_cases):
                prediction = risk_model.assess_overall_risk(company_profile)
                predicted_risk = prediction['risk_level']  # 'low', 'medium', 'high'
                
                if predicted_risk == expected_risk:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions
            
            # Risk model should achieve at least 70% accuracy
            assert accuracy >= 0.70, f"Risk model accuracy {accuracy:.2%} below 70% threshold"
            
            # Test prediction confidence
            high_confidence_predictions = 0
            for company_profile, _ in test_cases[:20]:  # Test subset for confidence
                prediction = risk_model.assess_overall_risk(company_profile)
                if prediction.get('confidence', 0) > 0.8:
                    high_confidence_predictions += 1
            
            confidence_rate = high_confidence_predictions / 20
            assert confidence_rate >= 0.6, f"Only {confidence_rate:.1%} of predictions had high confidence"
    
    def test_sentiment_analysis_accuracy(self):
        """Test sentiment analysis model accuracy on financial texts."""
        from src.backend.nlp_services.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
        
        # Create test dataset with labeled sentiment
        test_texts = [
            ("The company shows strong revenue growth and market position", "positive"),
            ("Significant regulatory risks and market headwinds ahead", "negative"),
            ("Mixed financial performance with both opportunities and challenges", "neutral"),
            ("Exceptional leadership team driving innovation and expansion", "positive"),
            ("Declining market share and increasing competition concerns", "negative"),
            ("Stable business model with predictable cash flows", "neutral"),
            ("Revolutionary technology with massive market potential", "positive"),
            ("Substantial debt levels and liquidity concerns", "negative"),
            ("Standard industry performance with no major issues", "neutral"),
            ("Outstanding growth trajectory and profitability", "positive")
        ]
        
        with patch.object(SentimentAnalyzer, '_load_model') as mock_load:
            # Mock sentiment model predictions
            mock_model = Mock()
            mock_model.predict.side_effect = self._mock_sentiment_predictions
            
            analyzer = SentimentAnalyzer()
            analyzer.model = mock_model
            
            correct_predictions = 0
            
            for text, expected_sentiment in test_texts:
                result = analyzer.analyze_text(text)
                predicted_sentiment = result['sentiment']
                
                if predicted_sentiment == expected_sentiment:
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(test_texts)
            
            # Sentiment analysis should achieve at least 75% accuracy
            assert accuracy >= 0.75, f"Sentiment analysis accuracy {accuracy:.2%} below 75% threshold"
    
    def test_peer_matching_accuracy(self):
        """Test peer company matching algorithm accuracy."""
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        
        # Create test companies with known similarities
        target_company = CompanyFactory.build(
            industry_sector='Technology',
            annual_revenue=Decimal('50000000'),
            employee_count=200
        )
        
        # Create peer companies with varying similarity levels
        high_similarity_peers = [
            CompanyFactory.build(
                industry_sector='Technology',
                annual_revenue=Decimal('45000000'),
                employee_count=180
            ),
            CompanyFactory.build(
                industry_sector='Technology', 
                annual_revenue=Decimal('55000000'),
                employee_count=220
            )
        ]
        
        medium_similarity_peers = [
            CompanyFactory.build(
                industry_sector='Technology',
                annual_revenue=Decimal('25000000'),
                employee_count=100
            ),
            CompanyFactory.build(
                industry_sector='Software & Services',
                annual_revenue=Decimal('50000000'),
                employee_count=200
            )
        ]
        
        low_similarity_peers = [
            CompanyFactory.build(
                industry_sector='Healthcare',
                annual_revenue=Decimal('50000000'),
                employee_count=200
            ),
            CompanyFactory.build(
                industry_sector='Materials',
                annual_revenue=Decimal('500000000'),
                employee_count=2000
            )
        ]
        
        all_peers = high_similarity_peers + medium_similarity_peers + low_similarity_peers
        
        cca_model = CCAModel()
        similarity_scores = cca_model.calculate_peer_similarity(target_company, all_peers)
        
        # High similarity peers should have highest scores
        high_sim_scores = similarity_scores[:2]
        medium_sim_scores = similarity_scores[2:4]
        low_sim_scores = similarity_scores[4:6]
        
        assert all(score > 0.7 for score in high_sim_scores), "High similarity peers should have scores > 0.7"
        assert all(0.3 < score < 0.7 for score in medium_sim_scores), "Medium similarity peers should have scores 0.3-0.7"
        assert all(score < 0.3 for score in low_sim_scores), "Low similarity peers should have scores < 0.3"
    
    def test_model_consistency_across_runs(self):
        """Test that models produce consistent results across multiple runs."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        test_company = CompanyFactory.build()
        
        # Run the same valuation multiple times
        results = []
        for _ in range(10):
            valuation_inputs = {
                'company_data': test_company,
                'financial_projections': {
                    'revenue_growth_rates': [0.20, 0.15, 0.12, 0.10, 0.08],
                    'ebitda_margins': [0.18, 0.20, 0.22, 0.23, 0.24],
                    'tax_rate': 0.25,
                    'capex_as_pct_revenue': 0.04,
                    'working_capital_change': 0.02
                },
                'valuation_assumptions': {
                    'wacc': 0.11,
                    'terminal_growth_rate': 0.03,
                    'forecast_years': 5
                },
                'net_debt': 5000000,
                'shares_outstanding': 50000000
            }
            
            result = dcf_model.calculate_valuation(valuation_inputs)
            results.append(result['price_per_share'])
        
        # All results should be identical (deterministic calculation)
        assert len(set(results)) == 1, f"DCF model produced inconsistent results: {results}"
        
        # Test with probabilistic models
        with patch('numpy.random.seed') as mock_seed:
            mock_seed.return_value = None
            
            # Test risk model consistency with same seed
            from src.backend.ml_services.valuation_engine.risk_model import RiskAssessmentModel
            
            risk_model = RiskAssessmentModel()
            company_profile = {
                'industry_sector': 'Technology',
                'revenue': 45000000,
                'employee_count': 150,
                'founded_year': 2018
            }
            
            # Set seed before each prediction to ensure consistency
            risk_results = []
            for i in range(5):
                np.random.seed(42)  # Same seed each time
                result = risk_model.assess_overall_risk(company_profile)
                risk_results.append(result['overall_score'])
            
            # With same seed, results should be very similar (allowing for small floating point differences)
            score_std = np.std(risk_results)
            assert score_std < 0.01, f"Risk model shows inconsistency across runs: std={score_std}"
    
    def test_model_performance_benchmarks(self):
        """Test that models meet performance benchmarks for inference time."""
        import time
        
        # Test DCF model performance
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        test_company = CompanyFactory.build()
        
        valuation_inputs = {
            'company_data': test_company,
            'financial_projections': {
                'revenue_growth_rates': [0.20, 0.15, 0.12, 0.10, 0.08],
                'ebitda_margins': [0.18, 0.20, 0.22, 0.23, 0.24],
                'tax_rate': 0.25,
                'capex_as_pct_revenue': 0.04,
                'working_capital_change': 0.02
            },
            'valuation_assumptions': {
                'wacc': 0.11,
                'terminal_growth_rate': 0.03,
                'forecast_years': 5
            },
            'net_debt': 5000000,
            'shares_outstanding': 50000000
        }
        
        # Measure DCF calculation time
        start_time = time.time()
        result = dcf_model.calculate_valuation(valuation_inputs)
        dcf_time = time.time() - start_time
        
        assert dcf_time < 1.0, f"DCF calculation took {dcf_time:.2f}s, should be under 1s"
        
        # Test peer analysis performance
        from src.backend.ml_services.valuation_engine.cca_model import CCAModel
        
        cca_model = CCAModel()
        peer_companies = [MarketDataFactory.build() for _ in range(100)]
        
        start_time = time.time()
        similarity_scores = cca_model.calculate_peer_similarity(test_company, peer_companies)
        cca_time = time.time() - start_time
        
        assert cca_time < 2.0, f"Peer analysis took {cca_time:.2f}s, should be under 2s for 100 peers"
        
        # Test sentiment analysis performance
        from src.backend.nlp_services.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
        
        with patch.object(SentimentAnalyzer, '_load_model'):
            analyzer = SentimentAnalyzer()
            analyzer.model = Mock()
            analyzer.model.predict.return_value = {'sentiment': 'positive', 'confidence': 0.85}
            
            test_text = "This company shows strong growth potential with excellent market positioning."
            
            start_time = time.time()
            result = analyzer.analyze_text(test_text)
            sentiment_time = time.time() - start_time
            
            assert sentiment_time < 0.5, f"Sentiment analysis took {sentiment_time:.2f}s, should be under 0.5s"
    
    def test_model_robustness_to_outliers(self):
        """Test model robustness when presented with outlier data."""
        from src.backend.ml_services.valuation_engine.dcf_model import DCFModel
        
        dcf_model = DCFModel()
        
        # Create company with extreme values
        outlier_company = CompanyFactory.build()
        
        # Test with extreme growth rates
        extreme_inputs = {
            'company_data': outlier_company,
            'financial_projections': {
                'revenue_growth_rates': [2.0, 1.5, 1.0, 0.5, 0.3],  # 200%, 150% growth
                'ebitda_margins': [0.50, 0.45, 0.40, 0.35, 0.30],  # Very high margins
                'tax_rate': 0.25,
                'capex_as_pct_revenue': 0.02,  # Very low capex
                'working_capital_change': 0.01
            },
            'valuation_assumptions': {
                'wacc': 0.25,  # Very high discount rate
                'terminal_growth_rate': 0.10,  # High terminal growth
                'forecast_years': 5
            },
            'net_debt': -100000000,  # Large net cash position
            'shares_outstanding': 1000000  # Small share count
        }
        
        # Model should handle extreme inputs gracefully
        try:
            result = dcf_model.calculate_valuation(extreme_inputs)
            
            # Result should be reasonable despite extreme inputs
            assert result['price_per_share'] > 0, "Price per share should be positive"
            assert result['price_per_share'] < 10000, "Price per share should be reasonable (<$10k)"
            assert 'enterprise_value' in result
            assert result['enterprise_value'] > 0
            
        except ValueError as e:
            # If model properly rejects extreme values, that's also acceptable
            assert "extreme" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_model_feature_importance(self):
        """Test that models correctly identify important features."""
        from src.backend.ml_services.predictive_models.feature_importance import FeatureImportanceAnalyzer
        
        with patch('joblib.load') as mock_load:
            # Mock a model with feature importance
            mock_model = Mock()
            mock_model.feature_importances_ = np.array([
                0.25,  # revenue
                0.20,  # revenue_growth
                0.15,  # profit_margin
                0.12,  # debt_to_equity
                0.10,  # market_share
                0.08,  # employee_growth
                0.05,  # r_and_d_spend
                0.03,  # market_size
                0.02   # other features...
            ])
            
            feature_names = [
                'revenue', 'revenue_growth', 'profit_margin', 'debt_to_equity',
                'market_share', 'employee_growth', 'r_and_d_spend', 'market_size',
                'other'
            ]
            
            mock_load.return_value = mock_model
            
            analyzer = FeatureImportanceAnalyzer()
            importance_scores = analyzer.get_feature_importance(feature_names)
            
            # Revenue and growth should be among top features
            top_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_feature_names = [name for name, _ in top_features]
            
            assert 'revenue' in top_feature_names, "Revenue should be a top feature"
            assert 'revenue_growth' in top_feature_names or 'profit_margin' in top_feature_names, \
                "Growth or profitability should be a top feature"
            
            # Feature importance should sum to approximately 1.0
            total_importance = sum(importance_scores.values())
            assert abs(total_importance - 1.0) < 0.01, f"Feature importance sum {total_importance} should be ~1.0"
    
    def _generate_synthetic_actual_valuation(self, company_data: Dict) -> float:
        """Generate synthetic 'actual' valuation for testing purposes."""
        # Simple synthetic valuation based on company characteristics
        base_multiple = 4.0  # Base P/S ratio
        
        # Adjust based on industry
        industry_multipliers = {
            'Technology': 1.5,
            'Healthcare': 1.3,
            'Financial Services': 0.8,
            'Materials': 0.6,
            'Energy': 0.7
        }
        
        industry_mult = industry_multipliers.get(company_data.get('industry_sector', 'Technology'), 1.0)
        revenue = float(company_data.get('annual_revenue', 50000000))
        shares = company_data.get('shares_to_be_issued', 50000000)
        
        # Add some randomness to simulate market conditions
        market_randomness = np.random.normal(1.0, 0.2)  # ±20% market variation
        
        valuation = (revenue * base_multiple * industry_mult * market_randomness) / shares
        return max(0.5, min(50.0, valuation))  # Clamp between $0.5 and $50
    
    def _generate_risk_test_cases(self, count: int) -> List[Tuple[Dict, str]]:
        """Generate test cases with known risk levels."""
        test_cases = []
        
        for i in range(count):
            if i % 3 == 0:  # Low risk
                company_profile = {
                    'industry_sector': 'Financial Services',
                    'revenue': np.random.uniform(100e6, 500e6),
                    'employee_count': np.random.randint(500, 2000),
                    'founded_year': np.random.randint(1990, 2010),
                    'debt_to_equity': np.random.uniform(0.1, 0.4),
                    'current_ratio': np.random.uniform(1.5, 3.0)
                }
                risk_level = 'low'
            elif i % 3 == 1:  # Medium risk
                company_profile = {
                    'industry_sector': 'Technology',
                    'revenue': np.random.uniform(20e6, 100e6),
                    'employee_count': np.random.randint(100, 500),
                    'founded_year': np.random.randint(2010, 2020),
                    'debt_to_equity': np.random.uniform(0.3, 0.8),
                    'current_ratio': np.random.uniform(1.0, 2.0)
                }
                risk_level = 'medium'
            else:  # High risk
                company_profile = {
                    'industry_sector': 'Biotech',
                    'revenue': np.random.uniform(1e6, 20e6),
                    'employee_count': np.random.randint(20, 100),
                    'founded_year': np.random.randint(2018, 2023),
                    'debt_to_equity': np.random.uniform(0.5, 2.0),
                    'current_ratio': np.random.uniform(0.5, 1.5)
                }
                risk_level = 'high'
            
            test_cases.append((company_profile, risk_level))
        
        return test_cases
    
    def _mock_sentiment_predictions(self, texts: List[str]) -> List[Dict]:
        """Mock sentiment predictions based on text content."""
        predictions = []
        
        for text in texts:
            text_lower = text.lower()
            
            # Simple keyword-based mock prediction
            positive_words = ['strong', 'growth', 'excellent', 'outstanding', 'exceptional', 'revolutionary']
            negative_words = ['risk', 'declining', 'concerns', 'debt', 'challenges', 'headwinds']
            
            positive_score = sum(1 for word in positive_words if word in text_lower)
            negative_score = sum(1 for word in negative_words if word in text_lower)
            
            if positive_score > negative_score:
                sentiment = 'positive'
                confidence = min(0.9, 0.6 + positive_score * 0.1)
            elif negative_score > positive_score:
                sentiment = 'negative'
                confidence = min(0.9, 0.6 + negative_score * 0.1)
            else:
                sentiment = 'neutral'
                confidence = 0.7
            
            predictions.append({
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': {'positive': positive_score, 'negative': negative_score, 'neutral': 1}
            })
        
        return predictions


@pytest.mark.ml
class TestDataDriftDetection:
    """Test data drift detection for production models."""
    
    def setup_method(self):
        """Set up drift detection tests."""
        self.reference_data = self._create_reference_dataset()
        self.drift_threshold = 0.1  # 10% drift threshold
    
    def test_feature_drift_detection(self):
        """Test detection of feature distribution drift."""
        from src.backend.ml_services.monitoring.drift_detector import DataDriftDetector
        
        detector = DataDriftDetector()
        
        # Create current data with slight drift
        current_data = self.reference_data.copy()
        current_data['revenue'] *= 1.2  # 20% increase in average revenue
        current_data['employee_count'] += 50  # Shift in employee count
        
        drift_report = detector.detect_drift(self.reference_data, current_data)
        
        # Should detect drift in revenue and employee_count
        assert 'revenue' in drift_report['drifted_features']
        assert 'employee_count' in drift_report['drifted_features']
        assert drift_report['overall_drift_score'] > self.drift_threshold
        
        # Should not detect drift in features that haven't changed
        stable_features = set(drift_report['stable_features'])
        assert len(stable_features) > 0
    
    def test_target_drift_detection(self):
        """Test detection of target variable drift."""
        from src.backend.ml_services.monitoring.drift_detector import DataDriftDetector
        
        detector = DataDriftDetector()
        
        reference_targets = np.random.normal(5.0, 2.0, 1000)  # Mean=5, std=2
        current_targets = np.random.normal(7.0, 2.5, 1000)    # Shifted mean and std
        
        drift_score = detector.detect_target_drift(reference_targets, current_targets)
        
        assert drift_score > self.drift_threshold, f"Target drift {drift_score} should exceed threshold {self.drift_threshold}"
    
    def test_concept_drift_detection(self):
        """Test detection of concept drift (relationship changes)."""
        from src.backend.ml_services.monitoring.drift_detector import DataDriftDetector
        
        detector = DataDriftDetector()
        
        # Create reference dataset with linear relationship
        X_ref = np.random.randn(1000, 5)
        y_ref = X_ref[:, 0] + 0.5 * X_ref[:, 1] + np.random.normal(0, 0.1, 1000)
        
        # Create current dataset with changed relationship
        X_curr = np.random.randn(1000, 5)
        y_curr = 2 * X_curr[:, 0] - X_curr[:, 1] + np.random.normal(0, 0.1, 1000)  # Different relationship
        
        concept_drift_score = detector.detect_concept_drift(
            X_ref, y_ref, X_curr, y_curr
        )
        
        assert concept_drift_score > self.drift_threshold, "Should detect concept drift in relationship"
    
    def test_drift_alert_system(self):
        """Test drift alerting and notification system."""
        from src.backend.ml_services.monitoring.drift_detector import DataDriftDetector
        from src.backend.ml_services.monitoring.alert_system import DriftAlertSystem
        
        detector = DataDriftDetector()
        alert_system = DriftAlertSystem()
        
        # Create severe drift scenario
        current_data = self.reference_data.copy()
        current_data['revenue'] *= 2.0  # 100% increase - severe drift
        
        drift_report = detector.detect_drift(self.reference_data, current_data)
        
        # Check if alert would be triggered
        should_alert = alert_system.should_trigger_alert(drift_report)
        assert should_alert, "Severe drift should trigger alert"
        
        # Test alert message generation
        alert_message = alert_system.generate_alert_message(drift_report)
        assert 'revenue' in alert_message
        assert 'drift detected' in alert_message.lower()
    
    def _create_reference_dataset(self) -> pd.DataFrame:
        """Create reference dataset for drift testing."""
        np.random.seed(42)
        
        data = {
            'revenue': np.random.lognormal(15, 1, 1000),  # Log-normal distribution
            'employee_count': np.random.randint(50, 1000, 1000),
            'founded_year': np.random.randint(1990, 2020, 1000),
            'debt_to_equity': np.random.gamma(2, 0.2, 1000),  # Gamma distribution
            'market_cap': np.random.exponential(1e8, 1000)
        }
        
        return pd.DataFrame(data)


@pytest.mark.ml
class TestModelExplainability:
    """Test model explainability and interpretability features."""
    
    def test_shap_explanations(self):
        """Test SHAP explanations for model predictions."""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP not available for explainability testing")
        
        from src.backend.ml_services.explainability.shap_explainer import SHAPExplainer
        
        # Mock model and data
        with patch('joblib.load') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([5.0, 3.5, 7.2])
            mock_load.return_value = mock_model
            
            explainer = SHAPExplainer()
            
            # Test data
            test_features = np.random.randn(3, 10)
            feature_names = [f'feature_{i}' for i in range(10)]
            
            explanations = explainer.explain_predictions(
                test_features, 
                feature_names=feature_names
            )
            
            # Verify explanation structure
            assert len(explanations) == 3  # One explanation per prediction
            
            for explanation in explanations:
                assert 'shap_values' in explanation
                assert 'expected_value' in explanation
                assert 'feature_contributions' in explanation
                
                # SHAP values should sum to prediction difference from expected
                shap_sum = np.sum(explanation['shap_values'])
                expected_diff = explanation['prediction'] - explanation['expected_value']
                assert abs(shap_sum - expected_diff) < 0.01
    
    def test_feature_importance_explanations(self):
        """Test feature importance explanations."""
        from src.backend.ml_services.explainability.feature_explainer import FeatureExplainer
        
        explainer = FeatureExplainer()
        
        # Mock feature importance data
        feature_importance = {
            'revenue': 0.25,
            'revenue_growth': 0.20,
            'profit_margin': 0.15,
            'debt_to_equity': 0.12,
            'market_share': 0.10,
            'employee_growth': 0.08,
            'r_and_d_spend': 0.05,
            'market_size': 0.03,
            'other': 0.02
        }
        
        explanation = explainer.generate_importance_explanation(feature_importance)
        
        # Verify explanation content
        assert 'top_features' in explanation
        assert 'feature_descriptions' in explanation
        assert 'importance_visualization' in explanation
        
        # Top features should be correctly identified
        top_features = explanation['top_features']
        assert top_features[0] == 'revenue'
        assert top_features[1] == 'revenue_growth'
        assert top_features[2] == 'profit_margin'
    
    def test_prediction_confidence_explanations(self):
        """Test confidence score explanations for predictions."""
        from src.backend.ml_services.explainability.confidence_explainer import ConfidenceExplainer
        
        explainer = ConfidenceExplainer()
        
        # Test different confidence scenarios
        high_confidence_prediction = {
            'prediction': 5.25,
            'model_uncertainty': 0.15,
            'data_quality_score': 0.95,
            'feature_coverage': 1.0,
            'peer_group_size': 25
        }
        
        low_confidence_prediction = {
            'prediction': 8.75,
            'model_uncertainty': 0.45,
            'data_quality_score': 0.65,
            'feature_coverage': 0.7,
            'peer_group_size': 3
        }
        
        # Test high confidence explanation
        high_conf_explanation = explainer.explain_confidence(high_confidence_prediction)
        assert high_conf_explanation['confidence_level'] == 'high'
        assert 'Strong data quality' in high_conf_explanation['explanation_text']
        
        # Test low confidence explanation
        low_conf_explanation = explainer.explain_confidence(low_confidence_prediction)
        assert low_conf_explanation['confidence_level'] == 'low'
        assert 'Limited peer group' in low_conf_explanation['explanation_text']
    
    def test_regulatory_compliance_explanations(self):
        """Test explanations that meet regulatory compliance requirements."""
        from src.backend.ml_services.explainability.regulatory_explainer import RegulatoryExplainer
        
        explainer = RegulatoryExplainer()
        
        valuation_result = {
            'prediction': 4.50,
            'methodology': {
                'dcf_weight': 0.6,
                'cca_weight': 0.4,
                'risk_adjustment': 0.85
            },
            'key_assumptions': {
                'wacc': 0.12,
                'terminal_growth': 0.03,
                'peer_group_size': 15
            },
            'data_sources': [
                'Company financial statements',
                'Market data (ASX)',
                'Peer company analysis',
                'Industry benchmarks'
            ]
        }
        
        compliance_explanation = explainer.generate_compliance_report(valuation_result)
        
        # Verify compliance report structure
        assert 'methodology_explanation' in compliance_explanation
        assert 'assumption_justifications' in compliance_explanation
        assert 'data_source_validation' in compliance_explanation
        assert 'limitations_and_risks' in compliance_explanation
        assert 'audit_trail' in compliance_explanation
        
        # Check that key regulatory requirements are met
        methodology_section = compliance_explanation['methodology_explanation']
        assert 'DCF' in methodology_section
        assert 'Comparable Company Analysis' in methodology_section
        assert 'risk adjustment' in methodology_section.lower()


@pytest.mark.ml 
@pytest.mark.slow
class TestModelPerformanceMonitoring:
    """Test production model performance monitoring."""
    
    def setup_method(self):
        """Set up performance monitoring tests."""
        self.performance_thresholds = {
            'accuracy_threshold': 0.75,
            'latency_threshold': 2.0,  # seconds
            'memory_threshold': 512,   # MB
            'error_rate_threshold': 0.05  # 5%
        }
    
    def test_model_performance_tracking(self):
        """Test model performance metrics tracking."""
        from src.backend.ml_services.monitoring.performance_monitor import ModelPerformanceMonitor
        
        monitor = ModelPerformanceMonitor()
        
        # Simulate model predictions with performance data
        prediction_logs = []
        for i in range(100):
            log_entry = {
                'timestamp': datetime.utcnow() - timedelta(minutes=i),
                'prediction_time': np.random.uniform(0.5, 3.0),
                'memory_usage': np.random.uniform(200, 600),
                'prediction_value': np.random.uniform(2.0, 8.0),
                'confidence': np.random.uniform(0.6, 0.95),
                'error': i % 20 == 0  # 5% error rate
            }
            prediction_logs.append(log_entry)
        
        # Calculate performance metrics
        metrics = monitor.calculate_performance_metrics(prediction_logs)
        
        # Verify metrics calculation
        assert 'avg_prediction_time' in metrics
        assert 'avg_memory_usage' in metrics
        assert 'error_rate' in metrics
        assert 'total_predictions' in metrics
        
        # Check if metrics meet thresholds
        performance_check = monitor.check_performance_thresholds(metrics, self.performance_thresholds)
        
        assert 'within_thresholds' in performance_check
        assert 'failing_metrics' in performance_check
        
        # Error rate should be around 5%
        assert abs(metrics['error_rate'] - 0.05) < 0.02
    
    def test_model_degradation_detection(self):
        """Test detection of model performance degradation."""
        from src.backend.ml_services.monitoring.degradation_detector import ModelDegradationDetector
        
        detector = ModelDegradationDetector()
        
        # Create historical performance baseline
        baseline_metrics = {
            'accuracy': 0.85,
            'avg_confidence': 0.82,
            'prediction_variance': 0.15
        }
        
        # Create current performance (degraded)
        current_metrics = {
            'accuracy': 0.72,  # Significant drop
            'avg_confidence': 0.68,  # Lower confidence
            'prediction_variance': 0.25  # Higher variance
        }
        
        degradation_report = detector.detect_degradation(baseline_metrics, current_metrics)
        
        assert degradation_report['degradation_detected'] == True
        assert 'accuracy' in degradation_report['degraded_metrics']
        assert degradation_report['severity'] in ['moderate', 'severe']
    
    def test_automated_model_retraining_trigger(self):
        """Test automated triggers for model retraining."""
        from src.backend.ml_services.monitoring.retraining_scheduler import RetrainingScheduler
        
        scheduler = RetrainingScheduler()
        
        # Simulate conditions that should trigger retraining
        trigger_conditions = {
            'accuracy_drop': 0.15,  # 15% accuracy drop
            'data_drift_score': 0.25,  # High drift
            'days_since_last_training': 90,  # 3 months
            'new_data_volume': 10000,  # Significant new data
            'error_rate_increase': 0.08  # Error rate increased
        }
        
        should_retrain = scheduler.should_trigger_retraining(trigger_conditions)
        
        assert should_retrain == True
        
        # Get retraining recommendations
        recommendations = scheduler.get_retraining_recommendations(trigger_conditions)
        
        assert 'priority' in recommendations
        assert 'recommended_data_size' in recommendations
        assert 'estimated_training_time' in recommendations
        assert recommendations['priority'] in ['low', 'medium', 'high', 'urgent']
    
    def test_model_a_b_testing(self):
        """Test A/B testing framework for model comparison."""
        from src.backend.ml_services.monitoring.ab_testing import ModelABTester
        
        ab_tester = ModelABTester()
        
        # Simulate results from two model versions
        model_a_results = np.random.normal(5.0, 1.5, 1000)  # Current model
        model_b_results = np.random.normal(5.3, 1.4, 1000)  # New model (slightly better)
        
        # Run A/B test
        ab_test_results = ab_tester.run_ab_test(
            model_a_results, 
            model_b_results,
            metric='accuracy',
            significance_level=0.05
        )
        
        assert 'statistical_significance' in ab_test_results
        assert 'confidence_interval' in ab_test_results
        assert 'recommendation' in ab_test_results
        
        # With slightly better performance, should recommend model B
        if ab_test_results['statistical_significance']:
            assert ab_test_results['recommendation'] == 'model_b'
    
    def test_production_model_health_check(self):
        """Test comprehensive health check for production models."""
        from src.backend.ml_services.monitoring.health_checker import ModelHealthChecker
        
        health_checker = ModelHealthChecker()
        
        # Simulate model status
        model_status = {
            'model_loaded': True,
            'last_prediction_time': datetime.utcnow() - timedelta(minutes=5),
            'cache_hit_rate': 0.85,
            'avg_response_time': 1.2,
            'error_count_24h': 12,
            'total_requests_24h': 5000,
            'memory_usage_percent': 45,
            'cpu_usage_percent': 30
        }
        
        health_report = health_checker.check_model_health(model_status)
        
        assert 'overall_health' in health_report
        assert 'component_health' in health_report
        assert 'recommendations' in health_report
        
        # Overall health should be good based on the status
        assert health_report['overall_health'] in ['excellent', 'good', 'fair', 'poor']
        
        # Check individual components
        components = health_report['component_health']
        assert 'model_availability' in components
        assert 'performance' in components
        assert 'resource_usage' in components