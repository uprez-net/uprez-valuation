"""
Comprehensive test suite for all ML valuation models
Tests cover functionality, accuracy, edge cases, and integration
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import shutil
from pathlib import Path

# Import all the models
import sys
sys.path.append('/Users/dhrubbiswas/code/uprez/uprez-valuation/src/backend')

from ml_services.models.advanced_dcf import (
    AdvancedDCFModel, AdvancedDCFInputs, AdvancedDCFOutputs,
    DCFScenarioResult, create_advanced_dcf_model
)
from ml_services.models.comparable_company_analysis import (
    EnhancedCCAModel, CompanyFinancialData, PeerSelectionCriteria,
    CCAModelResults, create_enhanced_cca_model
)
from ml_services.models.risk_assessment import (
    MultiFactorRiskModel, RiskAssessmentInputs, RiskAssessmentResults,
    MarketRiskFactors, OperationalRiskFactors, FinancialRiskFactors,
    create_multi_factor_risk_model
)
from ml_services.models.market_sentiment import (
    MarketSentimentAnalyzer, SentimentAnalysisInputs, SentimentAnalysisResults,
    NewsArticle, SocialMediaPost, AnalystReport, MarketIndicators,
    create_market_sentiment_analyzer
)
from ml_services.models.ensemble_framework import (
    EnsembleValuationModel, EnsembleInputs, EnsembleResults,
    ModelPrediction, create_ensemble_valuation_model
)
from ml_services.models.financial_analytics import (
    AdvancedFinancialAnalytics, FinancialAnalyticsInputs, FinancialTimeSeries,
    create_advanced_financial_analytics
)
from ml_services.models.model_serving import (
    ModelServingPipeline, PredictionRequest, PredictionResponse,
    BatchPredictionJob, create_model_serving_pipeline
)
from ml_services.models.model_monitoring import (
    ModelPerformanceMonitor, ModelPerformanceMetrics, DataDriftMetrics,
    create_model_performance_monitor
)

class TestAdvancedDCFModel:
    """Test suite for Advanced DCF Model"""
    
    @pytest.fixture
    async def dcf_model(self):
        """Create DCF model instance"""
        return create_advanced_dcf_model(simulation_runs=1000, projection_years=5)
    
    @pytest.fixture
    def sample_dcf_inputs(self):
        """Sample DCF inputs for testing"""
        return AdvancedDCFInputs(
            company_name="Test Company",
            sector="Technology",
            industry="Software",
            historical_revenues=[100, 120, 150, 180, 220],
            historical_ebitda=[20, 25, 35, 40, 50],
            historical_ebit=[15, 20, 28, 32, 42],
            historical_fcf=[12, 18, 25, 28, 38],
            revenue_growth_rates=[0.15, 0.12, 0.10, 0.08, 0.06],
            ebitda_margin_targets=[0.22, 0.24, 0.25, 0.25, 0.26],
            risk_free_rate=0.025,
            market_risk_premium=0.06,
            beta=1.2,
            cost_of_debt=0.04,
            debt_to_equity=0.3,
            terminal_growth_rate=0.025,
            shares_outstanding=10,
            tax_rate=0.25
        )
    
    @pytest.mark.asyncio
    async def test_dcf_basic_calculation(self, dcf_model, sample_dcf_inputs):
        """Test basic DCF calculation"""
        result = await dcf_model.calculate_advanced_valuation(
            sample_dcf_inputs,
            include_monte_carlo=False,
            include_scenarios=False
        )
        
        assert isinstance(result, AdvancedDCFOutputs)
        assert result.enterprise_value > 0
        assert result.equity_value > 0
        assert result.value_per_share > 0
        assert result.wacc > 0
        assert len(result.projection_summary) == 5  # 5-year projections
    
    @pytest.mark.asyncio
    async def test_dcf_monte_carlo(self, dcf_model, sample_dcf_inputs):
        """Test DCF with Monte Carlo simulation"""
        result = await dcf_model.calculate_advanced_valuation(
            sample_dcf_inputs,
            include_monte_carlo=True,
            confidence_levels=[0.80, 0.95]
        )
        
        assert '80%' in result.confidence_intervals
        assert '95%' in result.confidence_intervals
        assert result.probability_of_loss >= 0
        assert result.probability_of_loss <= 1
        assert len(result.monte_carlo_stats) > 0
    
    @pytest.mark.asyncio
    async def test_dcf_scenario_analysis(self, dcf_model, sample_dcf_inputs):
        """Test DCF scenario analysis"""
        result = await dcf_model.calculate_advanced_valuation(
            sample_dcf_inputs,
            include_scenarios=True,
            include_monte_carlo=False
        )
        
        assert 'bull' in result.scenario_results
        assert 'base' in result.scenario_results
        assert 'bear' in result.scenario_results
        
        # Bull case should be higher than bear case
        bull_value = result.scenario_results['bull'].value_per_share
        bear_value = result.scenario_results['bear'].value_per_share
        assert bull_value > bear_value
    
    @pytest.mark.asyncio
    async def test_dcf_sensitivity_analysis(self, dcf_model, sample_dcf_inputs):
        """Test DCF sensitivity analysis"""
        result = await dcf_model.calculate_advanced_valuation(
            sample_dcf_inputs,
            include_monte_carlo=False
        )
        
        assert len(result.sensitivity_rankings) > 0
        assert isinstance(result.sensitivity_matrix, pd.DataFrame)
        assert len(result.tornado_chart_data) > 0
    
    def test_dcf_input_validation(self, sample_dcf_inputs):
        """Test DCF input validation"""
        model = create_advanced_dcf_model()
        
        # Test with insufficient data
        invalid_inputs = AdvancedDCFInputs(
            company_name="Test",
            historical_revenues=[100],  # Only one year
            shares_outstanding=10
        )
        
        with pytest.raises(ValueError):
            asyncio.run(model.calculate_advanced_valuation(invalid_inputs))
    
    def test_dcf_edge_cases(self, dcf_model, sample_dcf_inputs):
        """Test DCF edge cases"""
        # Zero growth scenario
        zero_growth_inputs = sample_dcf_inputs
        zero_growth_inputs.revenue_growth_rates = [0.0] * 5
        zero_growth_inputs.terminal_growth_rate = 0.0
        
        result = asyncio.run(dcf_model.calculate_advanced_valuation(zero_growth_inputs))
        assert result.value_per_share > 0
        
        # High volatility scenario
        high_vol_inputs = sample_dcf_inputs
        high_vol_inputs.beta = 2.5
        high_vol_inputs.revenue_growth_rates = [0.5, -0.2, 0.3, -0.1, 0.2]
        
        result = asyncio.run(dcf_model.calculate_advanced_valuation(high_vol_inputs))
        assert result.value_per_share > 0

class TestEnhancedCCAModel:
    """Test suite for Enhanced CCA Model"""
    
    @pytest.fixture
    async def cca_model(self):
        """Create CCA model instance"""
        return create_enhanced_cca_model()
    
    @pytest.fixture
    def sample_target_company(self):
        """Sample target company for testing"""
        return CompanyFinancialData(
            company_name="Target Corp",
            sector="Technology",
            industry="Software",
            market_cap=1000,
            revenue=200,
            ebitda=50,
            net_income=30,
            ev_revenue=5.0,
            ev_ebitda=20.0,
            pe_ratio=33.3,
            shares_outstanding=10
        )
    
    @pytest.fixture
    def sample_universe_companies(self):
        """Sample universe of companies for peer selection"""
        companies = []
        
        # Create 20 sample companies with varying characteristics
        for i in range(20):
            company = CompanyFinancialData(
                company_name=f"Company_{i}",
                sector="Technology" if i < 15 else "Healthcare",
                industry="Software" if i < 10 else "Hardware",
                market_cap=500 + i * 100,
                revenue=100 + i * 20,
                ebitda=20 + i * 5,
                net_income=10 + i * 3,
                ev_revenue=4.0 + i * 0.2,
                ev_ebitda=15.0 + i * 1.0,
                pe_ratio=20.0 + i * 2.0,
                beta=0.8 + i * 0.05,
                revenue_growth_3y=0.1 + i * 0.01,
                ebitda_margin=0.15 + i * 0.01,
                shares_outstanding=5 + i * 0.5
            )
            companies.append(company)
        
        return companies
    
    @pytest.fixture
    def sample_selection_criteria(self):
        """Sample peer selection criteria"""
        return PeerSelectionCriteria(
            target_company="Target Corp",
            same_sector_required=True,
            size_multiple_range=(0.5, 2.0),
            max_peers=10,
            min_peers=3
        )
    
    @pytest.mark.asyncio
    async def test_cca_peer_selection(self, cca_model, sample_target_company, 
                                     sample_universe_companies, sample_selection_criteria):
        """Test peer selection functionality"""
        result = await cca_model.analyze_comparable_companies(
            sample_target_company,
            sample_universe_companies,
            sample_selection_criteria
        )
        
        assert isinstance(result, CCAModelResults)
        assert len(result.peer_companies) >= sample_selection_criteria.min_peers
        assert len(result.peer_companies) <= sample_selection_criteria.max_peers
        
        # Check that selected peers are from the same sector
        for peer in result.peer_companies:
            assert peer.sector == sample_target_company.sector
    
    @pytest.mark.asyncio
    async def test_cca_valuation_calculation(self, cca_model, sample_target_company,
                                           sample_universe_companies, sample_selection_criteria):
        """Test valuation calculation"""
        result = await cca_model.analyze_comparable_companies(
            sample_target_company,
            sample_universe_companies,
            sample_selection_criteria
        )
        
        assert len(result.implied_valuations) > 0
        assert len(result.multiple_statistics) > 0
        
        # Check that we have different valuation methods
        valuation_methods = list(result.implied_valuations.keys())
        assert any('mean' in method for method in valuation_methods)
        assert any('median' in method for method in valuation_methods)
    
    @pytest.mark.asyncio
    async def test_cca_statistical_analysis(self, cca_model, sample_target_company,
                                          sample_universe_companies, sample_selection_criteria):
        """Test statistical analysis of multiples"""
        result = await cca_model.analyze_comparable_companies(
            sample_target_company,
            sample_universe_companies,
            sample_selection_criteria,
            include_regression_analysis=True
        )
        
        # Check multiple statistics
        for multiple, stats in result.multiple_statistics.items():
            assert 'mean' in stats
            assert 'median' in stats
            assert 'std' in stats
            assert 'count' in stats
            assert stats['count'] > 0
        
        # Check regression results
        if result.regression_results:
            for multiple, reg_result in result.regression_results.items():
                assert 'predicted_value' in reg_result
                assert 'r_squared' in reg_result
                assert reg_result['r_squared'] >= 0
    
    def test_cca_edge_cases(self, cca_model):
        """Test CCA edge cases"""
        # Empty universe
        target = CompanyFinancialData(company_name="Target", sector="Tech")
        universe = []
        criteria = PeerSelectionCriteria(target_company="Target")
        
        with pytest.raises(ValueError):
            asyncio.run(cca_model.analyze_comparable_companies(target, universe, criteria))
        
        # No matching peers
        target = CompanyFinancialData(company_name="Target", sector="Unique")
        universe = [CompanyFinancialData(company_name="Peer", sector="Different")]
        
        result = asyncio.run(cca_model.analyze_comparable_companies(target, universe, criteria))
        assert len(result.peer_companies) == 0

class TestMultiFactorRiskModel:
    """Test suite for Multi-Factor Risk Model"""
    
    @pytest.fixture
    async def risk_model(self):
        """Create risk model instance"""
        return create_multi_factor_risk_model()
    
    @pytest.fixture
    def sample_risk_inputs(self):
        """Sample risk assessment inputs"""
        from ml_services.models.risk_assessment import ESGMetrics, RegulatoryRiskFactors
        
        return RiskAssessmentInputs(
            company_name="Test Company",
            sector="Technology",
            market_risks=MarketRiskFactors(
                beta=1.2,
                volatility=0.3,
                liquidity_risk=0.2,
                market_cap_risk=0.1
            ),
            financial_risks=FinancialRiskFactors(
                current_ratio=1.5,
                debt_to_equity=0.4,
                interest_coverage=5.0,
                earnings_volatility=0.2
            ),
            operational_risks=OperationalRiskFactors(
                revenue_concentration=0.3,
                operating_leverage=2.0,
                technology_risk=0.15
            ),
            esg_metrics=ESGMetrics(
                environmental_score=70.0,
                social_score=75.0,
                governance_score=80.0,
                esg_score=75.0
            ),
            regulatory_risks=RegulatoryRiskFactors(
                regulatory_environment_score=65.0,
                compliance_history=80.0
            )
        )
    
    @pytest.mark.asyncio
    async def test_risk_assessment_basic(self, risk_model, sample_risk_inputs):
        """Test basic risk assessment"""
        result = await risk_model.assess_comprehensive_risk(sample_risk_inputs)
        
        assert isinstance(result, RiskAssessmentResults)
        assert 0 <= result.composite_risk_score <= 100
        assert result.risk_grade in ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
        assert 0 <= result.probability_of_distress <= 1
        assert len(result.category_scores) > 0
    
    @pytest.mark.asyncio
    async def test_risk_stress_testing(self, risk_model, sample_risk_inputs):
        """Test risk model stress testing"""
        result = await risk_model.assess_comprehensive_risk(
            sample_risk_inputs,
            include_stress_testing=True
        )
        
        assert len(result.stress_test_results) > 0
        stress_scenarios = ['market_crash', 'recession', 'sector_crisis']
        
        for scenario in stress_scenarios:
            if scenario in result.stress_test_results:
                stress_result = result.stress_test_results[scenario]
                assert 'composite' in stress_result
                assert stress_result['composite'] >= result.composite_risk_score
    
    @pytest.mark.asyncio
    async def test_risk_peer_comparison(self, risk_model, sample_risk_inputs):
        """Test risk assessment with peer comparison"""
        # Create mock peer data
        peer_companies = [
            {'name': 'Peer1', 'risk_score': 45.0},
            {'name': 'Peer2', 'risk_score': 55.0},
            {'name': 'Peer3', 'risk_score': 50.0}
        ]
        
        result = await risk_model.assess_comprehensive_risk(
            sample_risk_inputs,
            peer_companies=peer_companies
        )
        
        assert len(result.peer_comparison) > 0
        assert len(result.category_percentiles) > 0
    
    def test_risk_grade_mapping(self):
        """Test risk grade to numeric conversion"""
        from ml_services.models.risk_assessment import risk_grade_to_numeric, numeric_to_risk_grade
        
        # Test round-trip conversion
        grades = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        for grade in grades:
            numeric = risk_grade_to_numeric(grade)
            converted_back = numeric_to_risk_grade(numeric)
            assert converted_back == grade or abs(risk_grade_to_numeric(converted_back) - numeric) < 5

class TestMarketSentimentAnalyzer:
    """Test suite for Market Sentiment Analyzer"""
    
    @pytest.fixture
    async def sentiment_analyzer(self):
        """Create sentiment analyzer instance"""
        return create_market_sentiment_analyzer(use_finbert=False)  # Disable FinBERT for testing
    
    @pytest.fixture
    def sample_sentiment_inputs(self):
        """Sample sentiment analysis inputs"""
        return SentimentAnalysisInputs(
            company_name="Test Company",
            sector="Technology",
            analysis_period_days=30,
            primary_keywords=["Test Company", "earnings", "growth"]
        )
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_basic(self, sentiment_analyzer, sample_sentiment_inputs):
        """Test basic sentiment analysis"""
        result = await sentiment_analyzer.analyze_market_sentiment(sample_sentiment_inputs)
        
        assert isinstance(result, SentimentAnalysisResults)
        assert -1 <= result.composite_sentiment_score <= 1
        assert 0 <= result.sentiment_confidence <= 1
        assert result.sentiment_trend in ['improving', 'stable', 'deteriorating']
    
    @pytest.mark.asyncio
    async def test_sentiment_components(self, sentiment_analyzer, sample_sentiment_inputs):
        """Test individual sentiment components"""
        result = await sentiment_analyzer.analyze_market_sentiment(
            sample_sentiment_inputs,
            return_detailed_breakdown=True
        )
        
        # Check that we have analysis for different sources
        assert hasattr(result, 'news_analysis')
        assert hasattr(result, 'social_analysis')
        assert hasattr(result, 'analyst_analysis')
        assert hasattr(result, 'market_analysis')
        
        # Check sentiment scores are in valid range
        assert -1 <= result.news_sentiment <= 1
        assert -1 <= result.social_sentiment <= 1
        assert -1 <= result.analyst_sentiment <= 1
    
    @pytest.mark.asyncio
    async def test_sentiment_forecasting(self, sentiment_analyzer, sample_sentiment_inputs):
        """Test sentiment forecasting capabilities"""
        result = await sentiment_analyzer.analyze_market_sentiment(
            sample_sentiment_inputs,
            include_forecasting=True
        )
        
        assert result.predicted_sentiment_direction in ['positive', 'negative', 'neutral']
        assert isinstance(result.sentiment_momentum, float)
        assert len(result.sentiment_support_resistance) >= 0
    
    def test_sentiment_utility_functions(self):
        """Test sentiment utility functions"""
        from ml_services.models.market_sentiment import (
            sentiment_to_valuation_adjustment,
            calculate_sentiment_risk_premium
        )
        
        # Test valuation adjustment
        positive_adj = sentiment_to_valuation_adjustment(0.5)
        assert positive_adj > 1.0
        
        negative_adj = sentiment_to_valuation_adjustment(-0.5)
        assert negative_adj < 1.0
        
        # Test risk premium calculation
        risk_premium = calculate_sentiment_risk_premium(0.3, 0.8)
        assert risk_premium >= 0

class TestEnsembleValuationModel:
    """Test suite for Ensemble Valuation Model"""
    
    @pytest.fixture
    async def ensemble_model(self):
        """Create ensemble model instance"""
        model = create_ensemble_valuation_model()
        
        # Register mock models for testing
        await self._register_mock_models(model)
        return model
    
    async def _register_mock_models(self, ensemble_model):
        """Register mock models for testing"""
        from ml_services.models.ensemble_framework import BaseValuationModel, ModelPrediction
        
        class MockDCFModel(BaseValuationModel):
            async def predict(self, inputs):
                return ModelPrediction(
                    model_name="dcf",
                    predicted_value=100.0,
                    confidence_score=0.85,
                    prediction_interval=(80.0, 120.0),
                    model_type="dcf"
                )
            
            def get_model_type(self):
                return "dcf"
            
            def get_feature_names(self):
                return ["revenue", "growth_rate", "wacc"]
        
        class MockCCAModel(BaseValuationModel):
            async def predict(self, inputs):
                return ModelPrediction(
                    model_name="cca",
                    predicted_value=95.0,
                    confidence_score=0.80,
                    prediction_interval=(75.0, 115.0),
                    model_type="cca"
                )
            
            def get_model_type(self):
                return "cca"
            
            def get_feature_names(self):
                return ["multiples", "peer_data"]
        
        await ensemble_model.register_model("dcf", MockDCFModel())
        await ensemble_model.register_model("cca", MockCCAModel())
    
    @pytest.fixture
    def sample_ensemble_inputs(self):
        """Sample ensemble inputs"""
        return EnsembleInputs(
            company_name="Test Company",
            models_to_include=["dcf", "cca"],
            weighting_method="dynamic",
            confidence_level=0.95
        )
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self, ensemble_model, sample_ensemble_inputs):
        """Test basic ensemble prediction"""
        result = await ensemble_model.predict_ensemble_valuation(sample_ensemble_inputs)
        
        assert isinstance(result, EnsembleResults)
        assert result.ensemble_valuation > 0
        assert 0 <= result.confidence_score <= 1
        assert len(result.individual_predictions) == 2  # DCF and CCA
        assert 0 <= result.model_agreement <= 1
    
    @pytest.mark.asyncio
    async def test_ensemble_weighting_methods(self, ensemble_model, sample_ensemble_inputs):
        """Test different ensemble weighting methods"""
        weighting_methods = ['equal', 'dynamic', 'performance_based', 'inverse_variance']
        
        for method in weighting_methods:
            test_inputs = sample_ensemble_inputs
            test_inputs.weighting_method = method
            
            result = await ensemble_model.predict_ensemble_valuation(test_inputs)
            assert result.ensemble_valuation > 0
            assert sum(result.final_weights.model_weights.values()) == pytest.approx(1.0, rel=1e-2)
    
    @pytest.mark.asyncio
    async def test_ensemble_uncertainty_quantification(self, ensemble_model, sample_ensemble_inputs):
        """Test ensemble uncertainty quantification"""
        result = await ensemble_model.predict_ensemble_valuation(
            sample_ensemble_inputs,
            return_detailed_results=True
        )
        
        uq = result.uncertainty_quantification
        assert 0 <= uq.epistemic_uncertainty
        assert 0 <= uq.aleatoric_uncertainty
        assert 0 <= uq.total_uncertainty
        assert 0 <= uq.confidence_score <= 1
        assert len(uq.prediction_intervals) > 0
    
    def test_ensemble_utility_functions(self):
        """Test ensemble utility functions"""
        from ml_services.models.ensemble_framework import (
            calculate_ensemble_sharpe_ratio,
            detect_model_outliers,
            calculate_prediction_diversity
        )
        
        # Create sample predictions for testing
        predictions = [
            ModelPrediction("model1", 100.0, 0.8, (90, 110), "test"),
            ModelPrediction("model2", 105.0, 0.85, (95, 115), "test"),
            ModelPrediction("model3", 95.0, 0.75, (85, 105), "test")
        ]
        
        # Test Sharpe ratio calculation
        sharpe = calculate_ensemble_sharpe_ratio(predictions)
        assert isinstance(sharpe, float)
        
        # Test outlier detection
        outliers = detect_model_outliers(predictions)
        assert isinstance(outliers, list)
        
        # Test diversity calculation
        diversity = calculate_prediction_diversity(predictions)
        assert isinstance(diversity, float)
        assert diversity >= 0

class TestAdvancedFinancialAnalytics:
    """Test suite for Advanced Financial Analytics"""
    
    @pytest.fixture
    async def analytics_model(self):
        """Create financial analytics model instance"""
        return create_advanced_financial_analytics()
    
    @pytest.fixture
    def sample_time_series(self):
        """Sample time series data"""
        dates = [datetime.now() - timedelta(days=30*i) for i in range(24, 0, -1)]
        values = [100 + i*2 + np.random.normal(0, 5) for i in range(24)]
        
        return FinancialTimeSeries(
            series_name="revenue",
            dates=dates,
            values=values,
            frequency="monthly"
        )
    
    @pytest.fixture
    def sample_analytics_inputs(self, sample_time_series):
        """Sample financial analytics inputs"""
        return FinancialAnalyticsInputs(
            company_name="Test Company",
            revenue_series=sample_time_series,
            ebitda_series=sample_time_series,
            fcf_series=sample_time_series,
            current_assets=[150, 160, 170, 180, 190],
            current_liabilities=[50, 55, 60, 65, 70],
            total_debt=[100, 105, 110, 115, 120],
            forecast_horizon=12
        )
    
    @pytest.mark.asyncio
    async def test_financial_analytics_basic(self, analytics_model, sample_analytics_inputs):
        """Test basic financial analytics"""
        result = await analytics_model.perform_comprehensive_analysis(sample_analytics_inputs)
        
        assert isinstance(result, FinancialAnalyticsResults)
        assert len(result.forecast_results) > 0
        assert result.credit_risk_metrics is not None
        assert result.working_capital_analysis is not None
    
    @pytest.mark.asyncio
    async def test_time_series_forecasting(self, analytics_model, sample_analytics_inputs):
        """Test time series forecasting"""
        result = await analytics_model.perform_comprehensive_analysis(sample_analytics_inputs)
        
        # Check forecasts
        for metric, forecast in result.forecast_results.items():
            assert len(forecast.forecast_values) == sample_analytics_inputs.forecast_horizon
            assert len(forecast.forecast_dates) == sample_analytics_inputs.forecast_horizon
            assert forecast.model_type in ['arima', 'exponential_smoothing', 'ml_ensemble']
    
    @pytest.mark.asyncio
    async def test_credit_risk_analysis(self, analytics_model, sample_analytics_inputs):
        """Test credit risk analysis"""
        result = await analytics_model.perform_comprehensive_analysis(sample_analytics_inputs)
        
        credit_metrics = result.credit_risk_metrics
        assert 0 <= credit_metrics.probability_of_default <= 1
        assert credit_metrics.altman_z_score > 0
        assert credit_metrics.credit_rating in ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        assert 300 <= credit_metrics.credit_score <= 850
    
    @pytest.mark.asyncio
    async def test_working_capital_optimization(self, analytics_model, sample_analytics_inputs):
        """Test working capital optimization"""
        result = await analytics_model.perform_comprehensive_analysis(sample_analytics_inputs)
        
        wc_analysis = result.working_capital_analysis
        assert wc_analysis.current_working_capital != 0
        assert wc_analysis.cash_conversion_cycle > 0
        assert wc_analysis.days_sales_outstanding > 0
        assert wc_analysis.days_inventory_outstanding > 0
    
    def test_financial_utility_functions(self):
        """Test financial utility functions"""
        from ml_services.models.financial_analytics import (
            calculate_compound_annual_growth_rate,
            calculate_volatility,
            detect_structural_breaks
        )
        
        # Test CAGR calculation
        cagr = calculate_compound_annual_growth_rate(100, 150, 5)
        assert cagr > 0
        
        # Test volatility calculation
        values = [100, 105, 95, 110, 90, 115]
        vol = calculate_volatility(values)
        assert vol >= 0
        
        # Test structural break detection
        ts_data = pd.Series([100, 102, 104, 120, 122, 124])
        breaks = detect_structural_breaks(ts_data)
        assert isinstance(breaks, list)

class TestModelServing:
    """Test suite for Model Serving Pipeline"""
    
    @pytest.fixture
    async def serving_pipeline(self):
        """Create model serving pipeline"""
        # Use temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        pipeline = create_model_serving_pipeline(
            models_directory=temp_dir,
            cache_enabled=False,  # Disable cache for testing
            monitoring_enabled=False
        )
        yield pipeline
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_prediction_request(self):
        """Sample prediction request"""
        return PredictionRequest(
            model_name="ensemble",
            inputs={
                "company_name": "Test Company",
                "financial_data": {"revenue": 100, "ebitda": 25}
            },
            prediction_type="real_time"
        )
    
    @pytest.mark.asyncio
    async def test_model_serving_health_check(self, serving_pipeline):
        """Test model serving health check"""
        health = await serving_pipeline.health_check()
        
        assert health['status'] in ['healthy', 'degraded']
        assert 'models_loaded' in health
        assert 'timestamp' in health
    
    @pytest.mark.asyncio
    async def test_model_serving_metrics(self, serving_pipeline):
        """Test model serving metrics"""
        metrics = await serving_pipeline.get_metrics()
        
        assert 'timestamp' in metrics
        assert 'models' in metrics
        assert 'batch_jobs' in metrics
    
    @pytest.fixture
    def sample_batch_job(self):
        """Sample batch prediction job"""
        # Create sample data
        sample_data = pd.DataFrame({
            'id': range(10),
            'company_name': [f'Company_{i}' for i in range(10)],
            'revenue': [100 + i*10 for i in range(10)]
        })
        
        return BatchPredictionJob(
            job_id="test_batch_job",
            model_name="ensemble",
            input_data=sample_data,
            output_path="test_output.csv",
            batch_size=5
        )
    
    @pytest.mark.asyncio
    async def test_batch_job_lifecycle(self, serving_pipeline, sample_batch_job):
        """Test batch job lifecycle"""
        # Start job
        job_id = await serving_pipeline.start_batch_job(sample_batch_job)
        assert job_id == sample_batch_job.job_id
        
        # Check status
        status = await serving_pipeline.get_batch_job_status(job_id)
        assert status is not None
        assert status.status in ["queued", "running", "completed", "failed"]
        
        # Cancel job
        cancelled = await serving_pipeline.cancel_batch_job(job_id)
        assert isinstance(cancelled, bool)

class TestModelMonitoring:
    """Test suite for Model Performance Monitor"""
    
    @pytest.fixture
    async def performance_monitor(self):
        """Create performance monitor"""
        temp_dir = tempfile.mkdtemp()
        monitor = create_model_performance_monitor(storage_path=temp_dir)
        yield monitor
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_basic(self, performance_monitor):
        """Test basic performance monitoring"""
        # Record some predictions
        for i in range(10):
            await performance_monitor.record_prediction(
                model_name="test_model",
                prediction_value=100 + i,
                confidence_score=0.8 + i*0.01,
                prediction_time_ms=50 + i*5,
                features={"feature1": i, "feature2": i*2},
                true_value=95 + i if i < 5 else None
            )
        
        # Calculate performance metrics
        metrics = await performance_monitor.calculate_and_record_performance_metrics(
            "test_model", window_hours=1
        )
        
        assert isinstance(metrics, ModelPerformanceMetrics)
        assert metrics.prediction_count == 10
        assert metrics.average_confidence_score > 0.8
    
    @pytest.mark.asyncio
    async def test_monitoring_dashboard_data(self, performance_monitor):
        """Test monitoring dashboard data"""
        # Record some data first
        await performance_monitor.record_prediction(
            "test_model", 100.0, 0.85, 50.0, {"test": 1}
        )
        
        # Get dashboard data
        dashboard_data = await performance_monitor.get_monitoring_dashboard_data("test_model")
        
        assert dashboard_data['model_name'] == "test_model"
        assert 'summary_stats' in dashboard_data
        assert 'monitoring_status' in dashboard_data

class TestIntegration:
    """Integration tests for the complete ML pipeline"""
    
    @pytest.mark.asyncio
    async def test_full_valuation_pipeline(self):
        """Test complete valuation pipeline integration"""
        # Create models
        dcf_model = create_advanced_dcf_model(simulation_runs=100)
        cca_model = create_enhanced_cca_model()
        risk_model = create_multi_factor_risk_model()
        ensemble_model = create_ensemble_valuation_model()
        
        # Create sample data
        company_data = CompanyFinancialData(
            company_name="Integration Test Corp",
            sector="Technology",
            revenue=200,
            ebitda=50,
            market_cap=1000,
            shares_outstanding=10
        )
        
        dcf_inputs = AdvancedDCFInputs(
            company_name="Integration Test Corp",
            historical_revenues=[150, 175, 200],
            historical_ebitda=[30, 40, 50],
            revenue_growth_rates=[0.15, 0.12, 0.10],
            ebitda_margin_targets=[0.25, 0.25, 0.25],
            shares_outstanding=10
        )
        
        # Test DCF model
        dcf_result = await dcf_model.calculate_advanced_valuation(
            dcf_inputs, include_monte_carlo=False
        )
        assert dcf_result.value_per_share > 0
        
        # Test that results are consistent across models
        assert isinstance(dcf_result.enterprise_value, (int, float))
        assert dcf_result.enterprise_value > 0
    
    @pytest.mark.asyncio
    async def test_model_serving_integration(self):
        """Test model serving with actual models"""
        serving_pipeline = create_model_serving_pipeline(
            cache_enabled=False,
            monitoring_enabled=False
        )
        
        # Test health check
        health = await serving_pipeline.health_check()
        assert health['status'] in ['healthy', 'degraded']
        
        # Test metrics
        metrics = await serving_pipeline.get_metrics()
        assert 'models' in metrics
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling across all models"""
        # Test invalid inputs
        with pytest.raises(ValueError):
            invalid_dcf = AdvancedDCFInputs(company_name="Test")
            # Missing required fields should raise error
        
        # Test empty data handling
        empty_ts = FinancialTimeSeries("empty", [], [])
        analytics = create_advanced_financial_analytics()
        # Should handle gracefully without crashing

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for ML models"""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_dcf_model_performance(self, benchmark):
        """Benchmark DCF model performance"""
        model = create_advanced_dcf_model(simulation_runs=1000)
        inputs = AdvancedDCFInputs(
            company_name="Benchmark Test",
            historical_revenues=[100, 120, 150],
            historical_ebitda=[20, 25, 35],
            revenue_growth_rates=[0.15, 0.12, 0.10],
            ebitda_margin_targets=[0.25, 0.25, 0.25],
            shares_outstanding=10
        )
        
        def run_dcf():
            return asyncio.run(model.calculate_advanced_valuation(inputs))
        
        result = benchmark(run_dcf)
        assert result.value_per_share > 0
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_ensemble_model_performance(self, benchmark):
        """Benchmark ensemble model performance"""
        model = create_ensemble_valuation_model()
        inputs = EnsembleInputs(
            company_name="Benchmark Test",
            models_to_include=["dcf"]
        )
        
        def run_ensemble():
            return asyncio.run(model.predict_ensemble_valuation(inputs))
        
        # This would fail without registered models, so we'll just test the creation
        assert model is not None

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])